"""
dcase_ae_baseline_runner.py — CLI runner for the DCASE autoencoder baseline.

Trains and evaluates the DCASE AE anomaly-detection baseline on one or more
machine entity IDs.  Each entity is processed independently: split → train →
evaluate (MSE + Mahalanobis) → log to Weights & Biases.

If the requested data is not present locally the runner will attempt to
download and preprocess it via the :mod:`download` package.

Usage examples
--------------
# Single entity, all defaults (DCASE baseline, 10 epochs):
    uv run python dcase_ae_baseline_runner.py mimii id_00

# Multiple entities, fan entity type:
    uv run python dcase_ae_baseline_runner.py mimii id_00,id_02,id_04 \\
        --entity-type fan

# Custom config, named W&B project:
    uv run python dcase_ae_baseline_runner.py mimii id_00 \\
        --experiment-name my-sweep              \\
        --epochs 50 --lr 5e-4 --bottleneck-dim 32
"""

from __future__ import annotations

import argparse
import dataclasses
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import DataSplit, MelFrameDataset, make_combined_split, make_train_test_split
from evaluation.labels import extract_labels
from evaluation.metrics import compute_metrics
from evaluation.report import print_report
from inference.file_scorer import FileScorer, make_mel_dataset_factory
from inference.frame_scorer import MahalanobisFrameScorer, ReconstructionFrameScorer
from inference.threshold import calibrate_threshold
from models.autoencoder import Autoencoder
from models.checkpoint import load_from_checkpoint
from tracking.wandb import WandbTracker
from training.config import TrainingConfig
from training.trainer import PtTrainer
from utilities import DEFAULT_DATA_DIR, DEFAULT_MEL_CACHE_DIR, detect_device, ensure_data, entity_dir
from features.hpss import HpssTransform, resolve_signal_transform


# ── CLI ───────────────────────────────────────────────────────────────────

def _load_toml_defaults() -> dict:
    """
    Load ``[tool.dcase-baseline]`` from the project's ``pyproject.toml``.

    Returns an empty dict when the file or the section is absent so the
    function is always safe to call.  Path-valued keys (``checkpoint_dir``,
    ``data_dir``) are converted to :class:`~pathlib.Path` objects so they
    match the types that :mod:`argparse` would produce from CLI input.
    """
    toml_path = Path(__file__).parent / "pyproject.toml"
    if not toml_path.exists():
        return {}
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    defaults = data.get("tool", {}).get("dcase-baseline", {})
    for key in ("checkpoint_dir", "data_dir"):
        if key in defaults:
            defaults[key] = Path(defaults[key])
    return defaults


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dcase_ae_baseline_runner",
        description="DCASE autoencoder baseline — anomaly detection runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── positional (can be omitted when set in pyproject.toml) ──
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name registered in download.registry (e.g. 'mimii')",
    )
    parser.add_argument(
        "entity_id",
        metavar="entity-id",
        nargs="?",
        default=None,
        help="Comma-separated entity IDs to evaluate (e.g. 'id_00' or 'id_00,id_02')",
    )

    # ── optional: data ───────────────────────────────────────
    parser.add_argument(
        "--entity-type",
        default="fan",
        help="Entity type within the dataset",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=(
            "Root directory for preprocessed audio data.  "
            "Defaults to the platform cache directory "
            f"(currently: {DEFAULT_DATA_DIR}).  "
            "Note: on macOS this directory is NOT cleared on reboot — only /tmp is.  "
            "Pass an explicit path (e.g. 'audio_data/') to control exactly "
            "where data is stored."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory for model checkpoints (one sub-folder per entity)",
    )
    parser.add_argument(
        "--mel-cache-dir",
        type=lambda p: None if p.lower() == "none" else Path(p),
        default=DEFAULT_MEL_CACHE_DIR,
        metavar="DIR",
        help=(
            "Directory for the on-disk mel-spectrogram cache.  "
            "Each spectrogram is stored as a .npy file keyed by a hash of "
            "the source path, mtime, feature params, and transform config — "
            "so stale entries are never served.  "
            f"Defaults to the platform cache dir ({DEFAULT_MEL_CACHE_DIR}).  "
            "Pass 'none' to disable caching entirely."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split",
    )
    parser.add_argument(
        "--mode",
        choices=["per-entity", "combined"],
        default="per-entity",
        help=(
            "Experiment mode. "
            "'per-entity' trains one model per entity ID. "
            "'combined' trains a single model on the merged data from all supplied IDs."
        ),
    )
    parser.add_argument(
        "--per-entity-eval",
        action="store_true",
        default=False,
        help=(
            "When using --mode combined, also evaluate the trained model "
            "separately on each entity ID's test split and log results to "
            "W&B under '<scorer>/<entity_id>/' prefixes.  "
            "Ignored in per-entity mode."
        ),
    )

    # ── optional: tracking ───────────────────────────────────
    parser.add_argument(
        "--experiment-name",
        default=None,
        metavar="NAME",
        help=(
            "W&B project name.  "
            "Defaults to '<dataset>-<entity-type>-anomaly-detection'"
        ),
    )

    # ── optional: dataset metadata overrides ─────────────────
    parser.add_argument(
        "--noise-level-db",
        type=int,
        default=None,
        metavar="DB",
        help=(
            "SNR level in dB to use, for datasets that provide multiple "
            "noise levels (e.g. MIMII: -6, 0, 6).  Only used with the MIMII dataset."
            "When omitted the data directory is expected to contain a flat "
            "'{entity_id}/' layout with no noise-level subdirectory."
        ),
    )
    parser.add_argument(
        "--hpss-component",
        default=None,
        metavar="COMPONENT",
        help=(
            "HPSS component to use as the training signal: 'harmonic', 'percussive', "
            "or 'none' to disable HPSS.  When omitted (or empty) the entity-type "
            "default from features.hpss.ENTITY_TYPE_COMPONENT is used as a fallback."
        ),
    )

    # ── optional: DCASE baseline config params ───────────────
    g = parser.add_argument_group(
        "model / training config (DCASE baseline defaults)"
    )
    g.add_argument("--epochs",       type=int,   default=10)
    g.add_argument("--lr",           type=float, default=1e-3,
                   help="Adam learning rate")
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--batch-size",   type=int,   default=512)
    g.add_argument(
        "--hidden-dims",
        default="128,128,128,128",
        help="Comma-separated encoder hidden-layer widths",
    )
    g.add_argument("--bottleneck-dim", type=int, default=8)
    g.add_argument("--early-stopping-patience", type=int, default=None,
                   metavar="N",
                   help="Stop if val loss does not improve for N epochs (None = off)")
    g.add_argument("--n-mels",          type=int, default=64)
    g.add_argument("--n-fft",           type=int, default=1_024)
    g.add_argument("--mel-hop-length",  type=int, default=512)
    g.add_argument("--n-frames",        type=int, default=5,
                   help="Context window width in mel time steps (P)")
    g.add_argument("--sample-rate",     type=int, default=16_000)
    g.add_argument("--max-fpr",         type=float, default=0.1,
                   help="Maximum false-positive rate for the partial AUC-ROC computation")

    # ── pyproject.toml defaults (override hardcoded values above) ────────
    # Precedence: CLI args  >  pyproject.toml [tool.dcase-baseline]  >  hardcoded fallbacks
    # Note: 'dataset' and 'entity_id' are positional args with nargs='?' and
    # are intentionally excluded here — set_defaults() does not reliably
    # override parsed positional values.  They are handled in main() instead.
    toml = _load_toml_defaults()
    parser.set_defaults(**{k: v for k, v in toml.items()
                           if k not in ("dataset", "entity_id")})

    return parser


# ── data availability ─────────────────────────────────────────────────────


# ── per-entity experiment ─────────────────────────────────────────────────

def _run_experiment(
    entity_ids:      list[str],
    entity_type:     str,
    data_dir:        Path,
    cfg:             TrainingConfig,
    hidden_dims:     tuple[int, ...],
    bottleneck_dim:  int,
    mel_factory,
    wandb_project:   str,
    seed:            int,
    noise_level_db:  int | None = None,
    per_entity_eval: bool = False,
    signal_transform = None,
    mel_cache_dir:   Path | None = DEFAULT_MEL_CACHE_DIR,
    max_fpr:         float = 0.1,
) -> None:
    """Train and evaluate the DCASE AE baseline for one or more entity IDs.

    When *entity_ids* contains a single element the behaviour is identical to
    the previous per-entity mode.  When it contains multiple elements the data
    from all directories is merged via :func:`data.make_combined_split` before
    training, and a single model is produced for the combined set.

    :param per_entity_eval: When ``True`` and ``len(entity_ids) > 1``, run an
        additional evaluation pass for each individual entity ID after the
        global combined evaluation.  Results are logged to W&B under
        ``<scorer>/<entity_id>/`` prefixes.  Has no effect when there is only
        one entity (which would duplicate the global result).
    """
    input_dim  = cfg.n_frames * cfg.n_mels      # 5 × 64 = 320
    entity_tag = "+".join(entity_ids)           # "id_00" or "id_00+id_02+id_04"

    noise_tag = f"  Noise      : {noise_level_db} dB" if noise_level_db is not None else ""
    print(f"\n{'═' * 64}")
    print(f"  Entity     : {entity_tag}  ({entity_type})")
    if noise_tag:
        print(noise_tag)
    print(f"  Mode       : {'combined' if len(entity_ids) > 1 else 'per-entity'}")
    print(f"  Device     : {cfg.device}   input_dim: {input_dim}")
    print(f"  Bottleneck : {bottleneck_dim}   lr: {cfg.learning_rate}   epochs: {cfg.epochs}")
    print(f"{'═' * 64}")

    # ── split ─────────────────────────────────────────────────
    entity_paths = [
        entity_dir(data_dir, entity_type, eid)
        for eid in entity_ids
    ]
    if len(entity_paths) == 1:
        split = make_train_test_split(entity_paths[0], seed=seed)
    else:
        split = make_combined_split(entity_paths, seed=seed)

    print(f"  train: {len(split.train_paths):>4} files")
    print(f"  test : {len(split.test_paths):>4} files  "
          f"({len(split.test_normal_paths)} normal / "
          f"{len(split.test_abnormal_paths)} abnormal)")

    # ── dataset & loader ──────────────────────────────────────
    ds_train = MelFrameDataset(
        file_paths=split.train_paths,
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        mel_hop_length=cfg.mel_hop_length,
        n_mels=cfg.n_mels,
        n_frames=cfg.n_frames,
        signal_transform=signal_transform,
        cache_dir=mel_cache_dir,
    )
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # ── model & optimiser ─────────────────────────────────────
    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        bottleneck_dim=bottleneck_dim,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ── W&B tracker ───────────────────────────────────────────
    tracker = WandbTracker(
        project=wandb_project,
        config={
            **dataclasses.asdict(cfg),
            "entity_id":       entity_tag,
            "entity_type":     entity_type,
            "noise_level_db":  noise_level_db,
            "hidden_dims":     list(hidden_dims),
            "bottleneck_dim":  bottleneck_dim,
            "input_dim":       input_dim,
            "mode":            "combined" if len(entity_ids) > 1 else "per-entity",
            "per_entity_eval": per_entity_eval and len(entity_ids) > 1,
            "max_fpr":         max_fpr,
            **(
                signal_transform.to_config()
                if isinstance(signal_transform, HpssTransform)
                else {"signal_transform": "none"}
            ),
        },
    )

    # ── train ─────────────────────────────────────────────────
    trainer = PtTrainer(model, optimizer, cfg, tracker=tracker)
    trainer.train(train_loader)

    # ── load best checkpoint ──────────────────────────────────
    model = load_from_checkpoint(
        cfg.checkpoint_dir / "best.pt", model, device=cfg.device
    )

    mse_scorer = ReconstructionFrameScorer(model, device=cfg.device)
    #mah_scorer = MahalanobisFrameScorer.fit(
    #    model.encoder, train_loader, device=cfg.device
    #)

    # ── global evaluation ─────────────────────────────────────
    def _evaluate(prefix: str, frame_scorer, eval_split: DataSplit | None = None) -> None:
        s = eval_split or split
        if not s.test_paths:
            print(f"\n── {prefix.upper()} evaluation ── SKIPPED (no test data)")
            return
        if not s.test_abnormal_paths:
            print(f"\n── {prefix.upper()} evaluation ── SKIPPED (no abnormal test files — AUC-ROC undefined)")
            return
        file_scorer = FileScorer(frame_scorer, dataset_factory=mel_factory)
        threshold   = calibrate_threshold(s.train_paths, file_scorer)
        scores      = file_scorer.score_files(s.test_paths)
        result      = compute_metrics(scores, extract_labels(s.test_paths), threshold, max_fpr=max_fpr)

        tracker.log_metric(f"{prefix}/auc_roc",        result.auc_roc)
        tracker.log_metric(f"{prefix}/partial_auc_roc", result.partial_auc_roc)
        tracker.log_metric(f"{prefix}/precision",      result.precision)
        tracker.log_metric(f"{prefix}/recall",         result.recall)
        tracker.log_metric(f"{prefix}/f1_score",       result.f1_score)
        tracker.log_metric(f"{prefix}/accuracy",       result.accuracy)
        tracker.log_param(f"{prefix}_threshold",       result.threshold)

        print(f"\n── {prefix.upper()} evaluation ---")
        print_report(result)

    _evaluate("mse",         mse_scorer)
    #_evaluate("mahalanobis", mah_scorer)

    # ── per-entity evaluation (combined mode only) ────────────
    if per_entity_eval and len(entity_ids) > 1:
        print(f"\n{'─' * 64}")
        print("  Per-entity breakdown")
        print(f"{'─' * 64}")
        for eid, epath in zip(entity_ids, entity_paths):
            entity_split = make_train_test_split(epath, seed=seed)
            print(f"\n  ▸ {eid}  "
                  f"(train {len(entity_split.train_paths)}, "
                  f"test {len(entity_split.test_paths)})")
            _evaluate(f"mse/{eid}",         mse_scorer, eval_split=entity_split)
            #_evaluate(f"mahalanobis/{eid}", mah_scorer, eval_split=entity_split)
    elif per_entity_eval and len(entity_ids) == 1:
        print("\n  ℹ  --per-entity-eval has no effect with a single entity "
              "(result is identical to the global evaluation).")

    tracker.finish()


# ── entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Positional defaults from TOML — applied manually because set_defaults()
    # does not reliably override nargs='?' positionals when a CLI value is given.
    toml = _load_toml_defaults()
    if args.dataset is None:
        args.dataset = toml.get("dataset")
    if args.entity_id is None:
        args.entity_id = toml.get("entity_id")

    # Validate — raise a clear error when neither CLI nor TOML provides the value.
    if args.dataset is None:
        parser.error(
            "'dataset' was not provided on the CLI and is not set in "
            "[tool.dcase-baseline] in pyproject.toml."
        )
    if args.entity_id is None:
        parser.error(
            "'entity-id' was not provided on the CLI and is not set in "
            "[tool.dcase-baseline] in pyproject.toml."
        )

    # Resolve the effective data directory via ensure_data, which applies the
    # noise-level suffix only when the dataset actually supports it (and warns
    # or raises otherwise).  Use the returned path as the authoritative data_dir
    # for all downstream lookups — do NOT call effective_data_dir independently.
    device     = detect_device()
    entity_ids = [eid.strip() for eid in args.entity_id.split(",")]
    base_data_dir = args.data_dir
    args.data_dir = ensure_data(
        args.dataset, args.entity_type, entity_ids,
        base_data_dir, noise_level_db=args.noise_level_db,
    )
    hidden_dims = tuple(int(d) for d in args.hidden_dims.split(","))
    wandb_project = (
        args.experiment_name
        or f"{args.dataset}-{args.entity_type}-anomaly-detection"
    )

    # Resolve HPSS signal transform (CLI override → entity-type default → None)
    signal_transform = resolve_signal_transform(
        getattr(args, "hpss_component", None),
        entity_type=args.entity_type,
    )
    if signal_transform is not None:
        print(f"HPSS     : {signal_transform.component} component")
    else:
        print("HPSS     : disabled")

    print(f"Device  : {device}")
    print(f"Dataset : {args.dataset}  entity type: {args.entity_type}  "
          f"IDs: {entity_ids}  mode: {args.mode}")
    print(f"Data dir: {args.data_dir}"
          + (" (platform cache — persists across reboots)" if args.data_dir == DEFAULT_DATA_DIR else ""))


    # ── shared mel factory ────────────────────────────────────
    mel_factory = make_mel_dataset_factory(
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        mel_hop_length=args.mel_hop_length,
        n_frames=args.n_frames,
        sample_rate=args.sample_rate,
        signal_transform=signal_transform,
        cache_dir=args.mel_cache_dir,
    )

    def _make_cfg(checkpoint_subdir: str) -> TrainingConfig:
        return TrainingConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=device,
            checkpoint_dir=args.checkpoint_dir / checkpoint_subdir,
            early_stopping_patience=args.early_stopping_patience,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            mel_hop_length=args.mel_hop_length,
            n_frames=args.n_frames,
        )

    common = dict(
        entity_type=args.entity_type,
        data_dir=args.data_dir,
        hidden_dims=hidden_dims,
        bottleneck_dim=args.bottleneck_dim,
        mel_factory=mel_factory,
        wandb_project=wandb_project,
        seed=args.seed,
        noise_level_db=args.noise_level_db,
        per_entity_eval=args.per_entity_eval,
        signal_transform=signal_transform,
        mel_cache_dir=args.mel_cache_dir,
        max_fpr=args.max_fpr,
    )

    if args.mode == "combined":
        # ── single experiment across all IDs ──────────────────
        entity_tag = "+".join(entity_ids)
        _run_experiment(
            entity_ids=entity_ids,
            cfg=_make_cfg(entity_tag),
            **common,
        )
    else:
        # ── one experiment per entity ID (default) ────────────
        if args.per_entity_eval:
            print("  ℹ  --per-entity-eval has no effect in per-entity mode.")
        for entity_id in entity_ids:
            _run_experiment(
                entity_ids=[entity_id],
                cfg=_make_cfg(entity_id),
                **common,
            )

    print("\n✓ All experiments complete.")


if __name__ == "__main__":
    main()

