#!/usr/bin/env bash
# run_experiments.sh
#
# Runs the DCASE AE baseline sequentially for specified entity types and noise levels.
#
# For example, with "pump" and "slider" as entity types, "combined" and "per-entity" as modes, and 6 and -6 dB as noise
# levels, the script runs the following 8 experiments:
#
#   entity_type × mode          × noise_level_db  →  4 runs per entity type
#   ─────────────────────────────────────────────────────────────────────────
#   pump        combined         6
#   pump        combined        -6
#   pump        per-entity       6
#   pump        per-entity      -6
#   slider      combined         6
#   slider      combined        -6
#   slider      per-entity       6
#   slider      per-entity      -6
#
# All other parameters (entity_id, epochs, lr, batch_size, …) are taken from
# [tool.dcase-baseline] in pyproject.toml.
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
#
# The script aborts immediately if any single run fails (set -e).

set -euo pipefail

RUNNER="uv run python dcase_ae_baseline_runner.py"

# ── experiment matrix ─────────────────────────────────────────────────────
ENTITY_TYPES=("fan")
MODES=("per-entity")
NOISE_LEVELS=(-6 6)

total=$(( ${#ENTITY_TYPES[@]} * ${#MODES[@]} * ${#NOISE_LEVELS[@]} ))
run=0

for entity_type in "${ENTITY_TYPES[@]}"; do
  for mode in "${MODES[@]}"; do
    for noise_level_db in "${NOISE_LEVELS[@]}"; do
      run=$(( run + 1 ))

      # Build a readable checkpoint sub-directory, e.g.
      #   checkpoints/pump/combined/6db
      noise_tag="${noise_level_db}db"
      checkpoint_subdir="${entity_type}/${mode}/${noise_tag}"

      echo ""
      echo "════════════════════════════════════════════════════════════════"
      echo "  Run ${run}/${total}"
      echo "  entity_type   : ${entity_type}"
      echo "  mode          : ${mode}"
      echo "  noise_level_db: ${noise_level_db}"
      echo "  checkpoint_dir: checkpoints/${checkpoint_subdir}"
      echo "════════════════════════════════════════════════════════════════"

      $RUNNER \
        --entity-type    "${entity_type}"    \
        --mode           "${mode}"           \
        --noise-level-db "${noise_level_db}" \
        --checkpoint-dir "checkpoints/${checkpoint_subdir}"

    done
  done
done

echo ""
echo "✓ All ${total} experiments complete."

