"""
gen_tables_figs.py — figure and table generators for the CSCI-535 paper.

Each public function builds one figure (or table) type and writes its
output under `turn_taking_analysis/paper/figures/` (figures) or
`turn_taking_analysis/paper/tables/` (LaTeX tables).

Modifications to a figure or table belong here — call sites elsewhere
should not duplicate plotting/tabulation logic. Add a new figure type by
writing a new public function and (when reusable) a shared private
helper.

Usage as a script (regenerates every default figure):

    python turn_taking_analysis/scripts/gen_tables_figs.py

Usage as a module:

    import gen_tables_figs as gtf
    gtf.figure_per_class_f1_vs_tau(top_n=3)
    gtf.figure_per_class_recall_vs_tau(top_n=5)
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Path / style constants
# ---------------------------------------------------------------------------

_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent                      # turn_taking_analysis/
PAPER_DIR    = PROJECT_ROOT / "paper"
SWEEP_CSV    = PAPER_DIR / "data_for_figs_tau_sweep.csv"
T400_CSV     = PAPER_DIR / "data_for_figs_tau_400.csv"
FIGURES_DIR  = PAPER_DIR / "figures"

DEFAULT_DPI       = 300
DEFAULT_FIGSIZE   = (13, 4.2)
COLOR_CYCLE       = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
TAU_TICKS_MS      = [100, 200, 400, 800, 1600]
TAU_TRAIN_MS      = 400  # vertical reference line (training horizon)
PANEL_TITLES      = ["HOLD", "YIELD", "BCHAN"]
PANEL_PREFIXES    = ["h",    "y",     "b"]


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_sweep_csv() -> list[dict]:
    with SWEEP_CSV.open() as f:
        return list(csv.DictReader(f))


def _parse_or_none(s) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _per_experiment_summary(sweep_rows: list[dict]) -> list[dict]:
    """One row per experiment, with per-experiment scalars (incl. AUCs).

    Used for ranking experiments by a per-experiment metric before
    selecting which experiments appear in a figure.
    """
    by_exp: dict[str, list[dict]] = defaultdict(list)
    for r in sweep_rows:
        by_exp[r["experiment"]].append(r)

    summary = []
    for exp_key, rows in by_exp.items():
        first = rows[0]
        row_at_400 = next(
            (r for r in rows if int(r["tau_ms"]) == TAU_TRAIN_MS), None
        )
        summary.append({
            "experiment":          exp_key,
            "experiment_name":     first["experiment_name"],
            "ablation":            first["ablation"],
            "arch":                first["arch"],
            "auc_macro_f1":        _parse_or_none(first.get("auc_macro_f1")),
            "auc_macro_recall":    _parse_or_none(first.get("auc_macro_recall")),
            "macro_f1_at_400":     _parse_or_none(row_at_400["macro_f1"])     if row_at_400 else None,
            "macro_recall_at_400": _parse_or_none(row_at_400["macro_recall"]) if row_at_400 else None,
        })
    return summary


# ---------------------------------------------------------------------------
# Shared plotting helpers
# ---------------------------------------------------------------------------

def _plot_per_class_vs_tau(
    top_rows: list[dict],
    metric_prefix: str,
    metric_label: str,
    out_png: Path,
    fig_title: str,
    dpi: int = DEFAULT_DPI,
    figsize: tuple = DEFAULT_FIGSIZE,
) -> Path:
    """
    3-panel per-class metric vs τ plot, one line per experiment in top_rows.

    Single-τ experiments (Coordination ablation) plot as scatter dots, not
    lines, with a "(τ=400 only)" suffix in the legend.
    """
    sweep_rows = _load_sweep_csv()
    cols = [f"{p}_{metric_prefix}" for p in PANEL_PREFIXES]

    exp_keys  = [r["experiment"] for r in top_rows]
    # Two-line legend label: arch on line 1, experiment_name on line 2.
    # matplotlib centers the handle icon vertically against the label box.
    exp_label = {
        r["experiment"]: f"{r['arch']}\n{r['experiment_name']}" for r in top_rows
    }

    series: dict[str, list[dict]] = {}
    for r in sweep_rows:
        if r["experiment"] not in exp_keys:
            continue
        series.setdefault(r["experiment"], []).append({
            "tau": int(r["tau_ms"]),
            "h":   float(r[cols[0]]),
            "y":   float(r[cols[1]]),
            "b":   float(r[cols[2]]),
        })
    for k in series:
        series[k].sort(key=lambda d: d["tau"])

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    color_for = {ek: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, ek in enumerate(exp_keys)}

    for ax, panel_col, panel_title in zip(axes, PANEL_PREFIXES, PANEL_TITLES):
        for ek in exp_keys:
            pts   = series[ek]
            taus  = [p["tau"]        for p in pts]
            vals  = [p[panel_col]    for p in pts]
            label = exp_label[ek]
            if len(pts) == 1:
                ax.scatter(taus, vals, s=80, color=color_for[ek],
                           edgecolors="black", linewidths=0.8,
                           label=f"{label} (τ=400 only)", zorder=3)
            else:
                ax.plot(taus, vals, marker="o", color=color_for[ek],
                        label=label, linewidth=1.6, markersize=5)
        ax.set_xscale("log")
        ax.set_xticks(TAU_TICKS_MS)
        ax.set_xticklabels([str(t) for t in TAU_TICKS_MS])
        ax.minorticks_off()
        ax.axvline(TAU_TRAIN_MS, color="grey", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_xlabel(r"$\tau$ (ms)")
        ax.set_ylabel(metric_label)
        ax.set_title(panel_title)
        ax.grid(True, alpha=0.25, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(top_rows),
               bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9,
               handletextpad=0.6, columnspacing=2.5)
    fig.suptitle(fig_title, fontsize=12, y=1.02)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {out_png}")
    return out_png


# ---------------------------------------------------------------------------
# Public figure generators (one function per figure type)
# ---------------------------------------------------------------------------

def figure_per_class_f1_vs_tau(
    top_n: int = 3,
    out_png: Optional[Path] = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Per-class F1 vs τ for the top-N experiments ranked by AUC of macro-F1.

    3-panel layout: HOLD / YIELD / BCHAN. One line per experiment, with
    markers at every τ in the grid. Vertical dotted line at τ=400 (training
    horizon). Coordination experiments (single-τ) are skipped from the
    ranking pool because their AUC is undefined.

    Default output: paper/figures/per_class_f1_top{N}_by_auc_macro_f1.png
    """
    sweep_rows = _load_sweep_csv()
    summary    = _per_experiment_summary(sweep_rows)
    ranked     = sorted(
        [s for s in summary if s["auc_macro_f1"] is not None],
        key=lambda s: -s["auc_macro_f1"],
    )[:top_n]

    out_png = out_png or (FIGURES_DIR / f"per_class_f1_top{top_n}_by_auc_macro_f1.png")
    return _plot_per_class_vs_tau(
        top_rows=ranked,
        metric_prefix="f1",
        metric_label="F1",
        out_png=out_png,
        fig_title=rf"Per-class F1 vs $\tau$ — Top {top_n} experiments by AUC of macro-F1",
        dpi=dpi,
    )


def figure_per_class_recall_vs_tau(
    top_n: int = 3,
    out_png: Optional[Path] = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Per-class Recall vs τ for the top-N experiments ranked by AUC of
    macro-recall (Macro-TPR).

    3-panel layout: HOLD / YIELD / BCHAN. One line per experiment, with
    markers at every τ in the grid. Vertical dotted line at τ=400 (training
    horizon). Coordination experiments (single-τ) are skipped from the
    ranking pool because their AUC is undefined.

    Default output: paper/figures/per_class_recall_top{N}_by_auc_macro_recall.png
    """
    sweep_rows = _load_sweep_csv()
    summary    = _per_experiment_summary(sweep_rows)
    ranked     = sorted(
        [s for s in summary if s["auc_macro_recall"] is not None],
        key=lambda s: -s["auc_macro_recall"],
    )[:top_n]

    out_png = out_png or (FIGURES_DIR / f"per_class_recall_top{top_n}_by_auc_macro_recall.png")
    return _plot_per_class_vs_tau(
        top_rows=ranked,
        metric_prefix="recall",
        metric_label="Recall",
        out_png=out_png,
        fig_title=rf"Per-class Recall vs $\tau$ — Top {top_n} experiments by AUC of Macro-TPR",
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# CLI: regenerate every default figure
# ---------------------------------------------------------------------------

def regenerate_all() -> list[Path]:
    """Regenerate every default figure with default args."""
    return [
        figure_per_class_f1_vs_tau(),
        figure_per_class_recall_vs_tau(),
    ]


if __name__ == "__main__":
    regenerate_all()
