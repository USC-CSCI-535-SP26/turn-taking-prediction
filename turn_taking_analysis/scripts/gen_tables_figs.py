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
TABLES_DIR                = PAPER_DIR / "tables"
TABLES_TAU400_OVERALL_DIR = TABLES_DIR / "tau_400" / "overall"
TABLES_TAU400_PER_AB_DIR  = TABLES_DIR / "tau_400" / "per_ablation_block"

DEFAULT_DPI       = 300
DEFAULT_FIGSIZE   = (13, 4.2)
COLOR_CYCLE       = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
TAU_TICKS_MS      = [100, 200, 400, 800, 1600]
TAU_TRAIN_MS      = 400  # vertical reference line (training horizon)
PANEL_TITLES      = ["HOLD", "YIELD", "BCHAN"]
PANEL_PREFIXES    = ["h",    "y",     "b"]

# Per-ablation metadata for table generation: notebook iteration order
# (matches the dict-literal order in fusion_experiments.ipynb so tables
# row-order is consistent with the notebook), and the dir-name → CSV
# ablation-label map.
NOTEBOOK_ORDER_BY_ABLATION_DIR: dict[str, list[str]] = {
    "standard": [
        "cpc_both_early", "cpc_both_neural_concat",
        "cpc_speaker_of_listener", "cpc_speaker_of_dyad", "full_dyad",
    ],
    "ssa": [
        "cpc_both_attention", "of_speaker_of_listener_attention",
        "cpc_speaker_of_listener_attention", "cpc_speaker_of_dyad_attention",
        "full_dyad_attention",
    ],
    "sca": [
        "cpc_speaker_of_listener_cross_attention",
        "cpc_speaker_of_listener_self_cross_attention",
    ],
    "coordination": [
        "cpc_speaker_plus_coordination",
        "openface_with_coord_neural_concat",
        "openface_BOTH_with_coord_neural_concat",
        "cpc_speaker_of_listener_coord",
        "cpc_speaker_both_faces_coord",
        "full_dyad_plus_coordination",
    ],
    "csa": [
        "cpc_speaker_coord_self_attention",
        "cpc_speaker_cpc_listener_coord_self_attention",
        "cpc_speaker_of_speaker_coord_self_attention",
        "openface_coord_self_attention",
        "cpc_speaker_of_listener_coord_self_attention",
        "full_dyad_plus_coordination_self_attention",
    ],
}

ABLATION_DIR_TO_LABEL: dict[str, str] = {
    "standard":     "Standard",
    "ssa":          "Standard + Self-Attention",
    "sca":          "Standard + Cross-Attention",
    "coordination": "Coordination",
    "csa":          "Coordination + Self-Attention",
}


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_sweep_csv() -> list[dict]:
    with SWEEP_CSV.open() as f:
        return list(csv.DictReader(f))


def _load_t400_csv() -> list[dict]:
    with T400_CSV.open() as f:
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
        fig_title=rf"Per-class F1 vs $\tau$ — Top {top_n} experiments by AUC of Macro-F1",
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
# LaTeX table helpers
# ---------------------------------------------------------------------------

def _fmt_metric_with_max_bold(value: float, is_row_max: bool, places: int = 3) -> str:
    """`\\textbf{…}` if this is the column's row-wise max, else plain."""
    s = f"{value:.{places}f}"
    return f"\\textbf{{{s}}}" if is_row_max else s


def _render_per_ablation_table(
    *,
    ablation_dir: str,
    table_filename: str,
    caption_template: str,
    label_prefix: str,
    metric_columns: list[tuple[str, str]],
) -> Path:
    """
    Generic builder for per-ablation τ=400 tables.

    Renders a booktabs LaTeX table to
    `paper/tables/tau_400/per_ablation_block/<ablation_dir>/<table_filename>`,
    with `Configuration` (= experiment_name) as the row-label column followed
    by one centered metric column per `(csv_col, display_header)` entry in
    `metric_columns`. Row-wise max in each metric column is bolded. Rows are
    ordered by the notebook iteration order for that ablation.

    Reused by `tables_per_ablation_macro`, `tables_per_ablation_f1_per_class`,
    and `tables_per_ablation_recall_per_class` — they differ only in metric
    columns, caption template, and output filename.
    """
    ablation_label = ABLATION_DIR_TO_LABEL[ablation_dir]
    notebook_order = NOTEBOOK_ORDER_BY_ABLATION_DIR[ablation_dir]
    order_idx = {k: i for i, k in enumerate(notebook_order)}

    rows = [r for r in _load_t400_csv() if r["ablation"] == ablation_label]
    rows.sort(key=lambda r: order_idx[r["experiment"]])

    col_maxes = {
        csv_col: max(float(r[csv_col]) for r in rows)
        for csv_col, _ in metric_columns
    }

    # Configuration column uses ragged-right p{4cm} so long experiment_name
    # values wrap to multiple lines without justified spacing. Requires
    # \usepackage{array} in the document preamble. Metric columns stay
    # centered. \small + \arraystretch{1.20} keep the table inside one
    # acmart sigconf column with comfortable inter-row spacing.
    column_spec = r">{\raggedright\arraybackslash}p{4cm}" + "c" * len(metric_columns)
    headers     = ["Configuration"] + [hdr for _, hdr in metric_columns]

    lines: list[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(rf"\caption{{{caption_template.format(ablation_label=ablation_label)}}}")
    lines.append(rf"\label{{{label_prefix}_{ablation_dir}}}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.20}")
    lines.append(rf"\begin{{tabular}}{{{column_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        cells = [r["experiment_name"]]
        for csv_col, _ in metric_columns:
            v = float(r[csv_col])
            cells.append(_fmt_metric_with_max_bold(v, v == col_maxes[csv_col]))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines) + "\n"

    out_tex = TABLES_TAU400_PER_AB_DIR / ablation_dir / table_filename
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex)
    print(f"wrote: {out_tex}")
    return out_tex


# ---------------------------------------------------------------------------
# Public table generators (one function per table type)
# ---------------------------------------------------------------------------

def tables_per_ablation_macro(
    ablation_dirs: Optional[list[str]] = None,
) -> list[Path]:
    """
    Per-ablation `macro.tex` tables — one row per experiment in the ablation,
    columns: Configuration, Macro-F1, Macro-TPR. Row-wise max in each metric
    column is bolded. Rows in notebook iteration order.

    `ablation_dirs` defaults to all 5 ablation block dir names.
    Each output lands at:
        paper/tables/tau_400/per_ablation_block/<dir>/macro.tex
    """
    ablation_dirs = ablation_dirs or list(NOTEBOOK_ORDER_BY_ABLATION_DIR.keys())
    return [
        _render_per_ablation_table(
            ablation_dir=ab,
            table_filename="macro.tex",
            caption_template=(
                r"Macro F1 and Recall at $\tau$=400~ms Across "
                "{ablation_label} Configurations"
            ),
            label_prefix="tab:macro",
            metric_columns=[
                ("macro_f1",     "Macro-F1"),
                ("macro_recall", "Macro-TPR"),
            ],
        )
        for ab in ablation_dirs
    ]


def tables_per_ablation_f1_per_class(
    ablation_dirs: Optional[list[str]] = None,
) -> list[Path]:
    """
    Per-ablation `f1_per_class.tex` tables — one row per experiment in the
    ablation, columns: Configuration, HOLD, YIELD, BCHAN (per-class F1).
    Row-wise max in each metric column is bolded. Rows in notebook
    iteration order.

    `ablation_dirs` defaults to all 5 ablation block dir names.
    Each output lands at:
        paper/tables/tau_400/per_ablation_block/<dir>/f1_per_class.tex
    """
    ablation_dirs = ablation_dirs or list(NOTEBOOK_ORDER_BY_ABLATION_DIR.keys())
    return [
        _render_per_ablation_table(
            ablation_dir=ab,
            table_filename="f1_per_class.tex",
            caption_template=(
                r"Per-Class F1 at $\tau$=400~ms Across "
                "{ablation_label} Configurations"
            ),
            label_prefix="tab:f1_per_class",
            metric_columns=[
                ("h_f1", "HOLD"),
                ("y_f1", "YIELD"),
                ("b_f1", "BCHAN"),
            ],
        )
        for ab in ablation_dirs
    ]


def tables_per_ablation_recall_per_class(
    ablation_dirs: Optional[list[str]] = None,
) -> list[Path]:
    """
    Per-ablation `recall_per_class.tex` tables — one row per experiment in
    the ablation, columns: Configuration, HOLD, YIELD, BCHAN (per-class
    recall). Row-wise max in each metric column is bolded. Rows in
    notebook iteration order.

    `ablation_dirs` defaults to all 5 ablation block dir names.
    Each output lands at:
        paper/tables/tau_400/per_ablation_block/<dir>/recall_per_class.tex
    """
    ablation_dirs = ablation_dirs or list(NOTEBOOK_ORDER_BY_ABLATION_DIR.keys())
    return [
        _render_per_ablation_table(
            ablation_dir=ab,
            table_filename="recall_per_class.tex",
            caption_template=(
                r"Per-Class Recall at $\tau$=400~ms Across "
                "{ablation_label} Configurations"
            ),
            label_prefix="tab:recall_per_class",
            metric_columns=[
                ("h_recall", "HOLD"),
                ("y_recall", "YIELD"),
                ("b_recall", "BCHAN"),
            ],
        )
        for ab in ablation_dirs
    ]


def table_overall_ablation_suite(out_tex: Optional[Path] = None) -> Path:
    """
    Cross-ablation 'study design' table listing all 24 experiments.

    Three columns from the τ=400 CSV: Block (= `ablation`), Streams (=
    `experiment_name`), Architecture (= `arch`). The Block name appears once
    per ablation (first row of the block); subsequent rows in the same block
    leave it blank. A `\\midrule` separates each ablation block from the next.

    Rows are emitted in notebook iteration order — ablations follow the
    `NOTEBOOK_ORDER_BY_ABLATION_DIR` key order (standard → ssa → sca →
    coordination → csa); experiments within each ablation follow the per-key
    list order. This is a study-design table, not a results table — no
    metric values appear.

    Default output: paper/tables/tau_400/overall/ablation_suite.tex
    """
    by_exp = {r["experiment"]: r for r in _load_t400_csv()}

    lines: list[str] = []
    lines.append(r"\begin{table*}[h]")
    lines.append(r"\caption{Ablation suite: 24 experiments across five ablation blocks.}")
    lines.append(r"\label{tab:ablations}")
    lines.append(r"\begin{tabular}{l l l}")
    lines.append(r"\toprule")
    lines.append(r"Block & Streams & Architecture \\")
    lines.append(r"\midrule")

    ablation_dirs = list(NOTEBOOK_ORDER_BY_ABLATION_DIR.keys())
    for i, ab_dir in enumerate(ablation_dirs):
        ab_label = ABLATION_DIR_TO_LABEL[ab_dir]
        exp_keys = NOTEBOOK_ORDER_BY_ABLATION_DIR[ab_dir]
        for j, exp_key in enumerate(exp_keys):
            r = by_exp[exp_key]
            block_cell = ab_label if j == 0 else ""
            lines.append(
                f"{block_cell} & {r['experiment_name']} & {r['arch']} " + r"\\"
            )
        # \midrule between blocks, but not after the last block (the
        # \bottomrule serves as the closer there).
        if i < len(ablation_dirs) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    tex = "\n".join(lines) + "\n"

    out_tex = out_tex or (TABLES_TAU400_OVERALL_DIR / "ablation_suite.tex")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex)
    print(f"wrote: {out_tex}")
    return out_tex


def table_overall_macro_leaderboard(out_tex: Optional[Path] = None) -> Path:
    """
    Cross-ablation leaderboard at τ=400~ms with one row per experiment,
    sorted by macro-F1 descending. Columns: Architecture, Configuration,
    Macro-F1, Macro-TPR. Row-wise maxima in the two metric columns are
    bolded.

    Default output: paper/tables/tau_400/overall/macro_leaderboard.tex
    """
    rows = _load_t400_csv()

    # Sort descending by macro_f1.
    rows.sort(key=lambda r: -float(r["macro_f1"]))

    f1s = [float(r["macro_f1"])     for r in rows]
    rec = [float(r["macro_recall"]) for r in rows]
    max_f1, max_rec = max(f1s), max(rec)

    lines: list[str] = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\caption{Macro F1 and Recall at $\tau$=400~ms Across All Configurations}")
    lines.append(r"\label{tab:overall_macro_leaderboard}")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"Architecture & Configuration & Macro-F1 & Macro-TPR \\")
    lines.append(r"\midrule")
    for r, f1, rc in zip(rows, f1s, rec):
        arch = r["arch"]
        cfg  = r["experiment_name"]
        lines.append(
            f"{arch} & {cfg} & "
            f"{_fmt_metric_with_max_bold(f1, f1 == max_f1)} & "
            f"{_fmt_metric_with_max_bold(rc, rc == max_rec)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    tex = "\n".join(lines) + "\n"

    out_tex = out_tex or (TABLES_TAU400_OVERALL_DIR / "macro_leaderboard.tex")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex)
    print(f"wrote: {out_tex}")
    return out_tex


# ---------------------------------------------------------------------------
# CLI: regenerate every default figure / table
# ---------------------------------------------------------------------------

def regenerate_all() -> list[Path]:
    """Regenerate every default figure and table with default args."""
    return [
        figure_per_class_f1_vs_tau(),
        figure_per_class_recall_vs_tau(),
        table_overall_ablation_suite(),
        table_overall_macro_leaderboard(),
        *tables_per_ablation_macro(),
        *tables_per_ablation_f1_per_class(),
        *tables_per_ablation_recall_per_class(),
    ]


if __name__ == "__main__":
    regenerate_all()
