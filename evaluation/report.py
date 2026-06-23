"""
evaluation/report.py — human-readable formatting of EvaluationResult.

:func:`format_report` converts an :class:`~evaluation.metrics.EvaluationResult`
into a structured text table suitable for logging, printing, or writing to a
file.  :func:`print_report` is a convenience wrapper that writes the result
directly to stdout.
"""

from __future__ import annotations

from evaluation.metrics import EvaluationResult

# Column width used for aligned output
_COL_WIDTH = 12


def format_report(result: EvaluationResult) -> str:
    """
    Format an :class:`~evaluation.metrics.EvaluationResult` as a text table.

    Example output::

        ┌────────────────────────────────┐
        │     Anomaly Detection Report   │
        ├──────────────────┬─────────────┤
        │ Threshold        │      0.0421 │
        │ AUC-ROC          │      0.9456 │
        │ pAUC-ROC         │      0.9230 │
        │ Precision        │      0.8667 │
        │ Recall           │      0.9286 │
        │ F1 Score         │      0.8966 │
        │ Accuracy         │      0.9000 │
        └──────────────────┴─────────────┘

    :param result: Metrics produced by
        :func:`~evaluation.metrics.compute_metrics`.
    :type result: EvaluationResult
    :returns: Formatted multi-line string; does **not** include a trailing
        newline.
    :rtype: str
    """
    rows: list[tuple[str, str]] = [
        ("Threshold", f"{result.threshold:.4f}"),
        ("AUC-ROC",   f"{result.auc_roc:.4f}"),
        ("pAUC-ROC",  f"{result.partial_auc_roc:.4f}"),
        ("Precision", f"{result.precision:.4f}"),
        ("Recall",    f"{result.recall:.4f}"),
        ("F1 Score",  f"{result.f1_score:.4f}"),
        ("Accuracy",  f"{result.accuracy:.4f}"),
    ]

    label_w = max(len(label) for label, _ in rows) + 2
    value_w = max(len(value) for _, value in rows) + 2
    total_w = label_w + value_w + 3  # borders + separator

    title = "Anomaly Detection Report"
    title_line = f"│ {title:^{total_w - 4}} │"
    divider_top    = "┌" + "─" * (total_w - 2) + "┐"
    divider_header = "├" + "─" * label_w + "┬" + "─" * value_w + "┤"
    divider_bottom = "└" + "─" * label_w + "┴" + "─" * value_w + "┘"

    lines: list[str] = [divider_top, title_line, divider_header]
    for label, value in rows:
        lines.append(f"│ {label:<{label_w - 2}} │ {value:>{value_w - 2}} │")
    lines.append(divider_bottom)

    return "\n".join(lines)


def print_report(result: EvaluationResult) -> None:
    """
    Print an :class:`~evaluation.metrics.EvaluationResult` to stdout.

    Convenience wrapper around :func:`format_report`.

    :param result: Metrics produced by
        :func:`~evaluation.metrics.compute_metrics`.
    :type result: EvaluationResult
    """
    print(format_report(result))

