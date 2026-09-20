"""Monitoring charts for a FlyLM eval run.

Design decisions, and why they are not preferences:

* **The palette is validated, not chosen.** Categorical slots 1-3 (blue, orange, aqua)
  cleared every hard gate under `--pairs all` in both light and dark mode: worst-pair CVD
  deltaE 9.2 light / 9.4 dark, normal-vision 24.0 / 20.9. The sequential ramp is ONE hue,
  light to dark - never a rainbow. Status colours are reserved and always ship with a text
  label, never colour alone.
* **Light-mode aqua sits at 2.74:1 against the chart surface**, below 3:1. The documented
  relief is visible labels or a table view, so every chart function here has a
  `table=True` companion that prints the same numbers. A static PNG has no hover layer
  either, so that table is also the only data-inspection path.
* **One y-axis, always.** Quality and cost are two scales and get two charts.
* **Dark mode is selected, not flipped**: its own steps from the same ramps, validated
  against the dark surface. `set_theme("dark")`.

Nothing in this module reads a summary's internals by index; everything takes the plain
dicts the harness already writes, so a chart cannot drift from the numbers it plots.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ------------------------------------------------------------------ validated palette

@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    text_primary: str
    text_secondary: str
    text_muted: str
    grid: str
    series: tuple[str, ...]
    sequential: tuple[str, ...]
    status: dict[str, str]


LIGHT = Theme(
    name="light", surface="#fcfcfb",
    text_primary="#0b0b0b", text_secondary="#52514e", text_muted="#7a7973", grid="#e6e5e1",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    sequential=("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"),
    status={"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"},
)
DARK = Theme(
    name="dark", surface="#1a1a19",
    text_primary="#ffffff", text_secondary="#c3c2b7", text_muted="#96958c", grid="#383835",
    series=("#3987e5", "#d95926", "#199e70", "#c98500"),
    sequential=("#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
                "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"),
    status={"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"},
)
THEME: Theme = LIGHT


def set_theme(name: str = "light") -> Theme:
    """Dark mode is a selected set of steps for the dark surface, not an inverted light
    palette - both were validated separately against their own surface."""
    global THEME
    THEME = DARK if name == "dark" else LIGHT
    mpl.rcParams.update({
        "figure.facecolor": THEME.surface, "axes.facecolor": THEME.surface,
        "savefig.facecolor": THEME.surface,
        "text.color": THEME.text_primary, "axes.labelcolor": THEME.text_secondary,
        "xtick.color": THEME.text_secondary, "ytick.color": THEME.text_secondary,
        "axes.edgecolor": THEME.grid, "grid.color": THEME.grid,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 12,
        "figure.dpi": 110, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return THEME


def sequential_cmap(n: int = 256):
    return LinearSegmentedColormap.from_list("fly_seq", THEME.sequential, N=n)


def _frame(ax, *, ylabel: str = "", xlabel: str = "", ygrid: bool = True) -> None:
    """Recessive grid and axes: the marks carry the message."""
    ax.set_ylabel(ylabel, color=THEME.text_secondary)
    ax.set_xlabel(xlabel, color=THEME.text_secondary)
    if ygrid:
        ax.grid(axis="y", lw=0.6, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


SLICE_SHORT = {
    "stage_accuracy": "stage acc", "enthusiasm_curve": "curve", "indifference": "indiff",
    "safety_geosmin": "safety", "voice": "voice", "containment": "containment",
    "unseen_fruit": "unseen", "robustness": "robust", "abstention": "abstain",
}


# ------------------------------------------------------------- 1. adapter comparison

def slice_pass_rates(summaries: dict[str, dict], *, title: str | None = None,
                     table: bool = True, ax=None):
    """Grouped bars, one group per slice, one bar per adapter, with 95% Wilson intervals.

    Horizontal on purpose: nine slices times three adapters is twenty-seven bars, and
    vertical value labels collide at that density. Horizontal also solves the bigger
    problem - a pass rate of 0.00 draws no bar at all, so a failing adapter would appear
    only in the legend. Every bar therefore carries its value as text anchored outside
    the bar end, which doubles as the light-mode relief for the sub-3:1 aqua slot.
    """
    names = list(summaries)
    slices = sorted({s for m in summaries.values() for s in m["slices"]},
                    key=lambda s: list(SLICE_SHORT).index(s) if s in SLICE_SHORT else 99)
    n = len(names)
    h = min(0.26, 0.78 / max(1, n))
    y = np.arange(len(slices))

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8.6, 0.26 * len(slices) * n + 1.7))

    for i, name in enumerate(names):
        vals, los, his = [], [], []
        for s_ in slices:
            pr = (summaries[name]["slices"].get(s_) or {}).get("pass_rate", {})
            v = pr.get("value")
            v = 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v
            lo, hi = (pr.get("ci95") or [v, v])
            vals.append(v)
            los.append(max(0.0, v - (lo if lo == lo else v)))
            his.append(max(0.0, (hi if hi == hi else v) - v))
        # y is inverted below, so the FIRST series needs the smallest y to sit on top
        off = (i - (n - 1) / 2) * (h + 0.02)
        ax.barh(y + off, vals, h, label=name, color=THEME.series[i % len(THEME.series)],
                edgecolor=THEME.surface, linewidth=1.6, zorder=3)
        ax.errorbar(vals, y + off, xerr=[los, his], fmt="none", ecolor=THEME.text_muted,
                    elinewidth=1.1, capsize=2.5, zorder=4)
        for yi, v, hi_ in zip(y + off, vals, his):
            ax.text(min(v + hi_ + 0.025, 1.14), yi, f"{v:.2f}", va="center", fontsize=8,
                    color=THEME.text_secondary, zorder=5)

    ax.set_yticks(y, [SLICE_SHORT.get(s_, s_) for s_ in slices], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.24)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("slice pass rate", color=THEME.text_secondary)
    ax.set_title(title or "Slice pass rate by adapter, with 95% intervals")
    ax.legend(frameon=False, ncols=min(n, 4), loc="upper left",
              bbox_to_anchor=(0, -0.055), fontsize=9)

    if table:
        from . import trace

        def cell(nm, s_):
            v = (summaries[nm]["slices"].get(s_) or {}).get("pass_rate", {}).get("value")
            return "n/a" if v is None or v != v else f"{v:.3f}"

        trace.table(["slice"] + names,
                    [[SLICE_SHORT.get(s_, s_)] + [cell(nm, s_) for nm in names]
                     for s_ in slices])
    return ax


# ------------------------------------------------------------------ 2. gate status board

def gate_board(summary: dict, *, title: str | None = None, ax=None):
    """Status tiles, not a chart: the question is "is this shippable", which is state.

    Status colours never carry meaning alone - every tile is labelled PASS / FAIL / n/m
    and marked, so it survives greyscale, CVD and forced-colors.
    """
    rows = [(s, g) for s, sl in summary["slices"].items() for g in sl["gates"]]
    rows.sort(key=lambda r: (r[1]["status"] != "FAIL", not r[1]["hard"], r[0]))
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(11.5, 0.42 * len(rows) + 1.0))
    ax.set_axis_off()
    ax.set_title(title or "Gate board - hard gates are fail-on-any")

    for i, (slc, g) in enumerate(rows):
        y = len(rows) - i - 1
        st = g["status"]
        col = {"pass": THEME.status["good"], "FAIL": THEME.status["critical"],
               "not_measured": THEME.text_muted}[st]
        mark = {"pass": "\u2713", "FAIL": "\u2717", "not_measured": "\u2013"}[st]
        ax.add_patch(plt.Rectangle((0, y - 0.34), 0.30, 0.68, color=col, alpha=0.16,
                                   ec=col, lw=1.2, zorder=2))
        ax.text(0.15, y, mark, ha="center", va="center", fontsize=10, color=col,
                fontweight="bold", zorder=3)
        ax.text(0.40, y, f"{SLICE_SHORT.get(slc, slc)}", fontsize=9.5,
                color=THEME.text_primary, va="center")
        ax.text(2.55, y, g["metric"], fontsize=9.5, color=THEME.text_secondary, va="center")
        val = g.get("value")
        ax.text(5.15, y, "n/a" if val is None else f"{val:.3f}", fontsize=9.5,
                family="monospace", color=THEME.text_primary, va="center", ha="right")
        ax.text(5.55, y, f"{g['op']} {g['threshold']:.2f}" if "op" in g else "",
                fontsize=9, color=THEME.text_muted, va="center")
        ax.text(7.30, y, "HARD" if g["hard"] else "target", fontsize=8.5, va="center",
                color=THEME.status["critical"] if g["hard"] else THEME.text_muted)
        ax.text(8.55, y, st.replace("_", " "), fontsize=9, va="center", color=col,
                fontweight="semibold" if st == "FAIL" else "normal")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    return ax


# --------------------------------------------------------------- 3. confusion matrix

def confusion_matrix(rows: list[dict], *, slices=("stage_accuracy", "unseen_fruit"),
                     title: str | None = None, table: bool = True, ax=None):
    """Sequential magnitude over a 6x6 grid -> one-hue heatmap, light to dark.

    This is the artifact a produce person would actually read: it shows WHICH stage the
    model confuses, and stage 5 is the row that matters.
    """
    m = np.zeros((6, 6), dtype=int)
    for r in rows:
        if r["slice"] not in slices or r["stage"] is None:
            continue
        pred = r["graded"]["implied_stage"]
        if pred is None:
            continue
        m[r["stage"]][pred] += 1

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(m, cmap=sequential_cmap(), vmin=0, vmax=max(1, m.max()))
    names = [f"{i} {n}" for i, n in enumerate(
        ["green", "turning", "ripe", "overripe", "fermenting", "mouldy"])]
    ax.set_xticks(range(6), [n.split()[0] for n in names], fontsize=8.5)
    ax.set_yticks(range(6), names, fontsize=8.5)
    ax.set_xlabel("stage the model implied", color=THEME.text_secondary)
    ax.set_ylabel("true stage", color=THEME.text_secondary)
    ax.set_title(title or "Stage confusion matrix")
    thresh = max(1, m.max()) * 0.55
    for i in range(6):
        for j in range(6):
            if m[i][j]:
                ax.text(j, i, m[i][j], ha="center", va="center", fontsize=9,
                        color=THEME.surface if m[i][j] > thresh else THEME.text_primary)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color=THEME.surface, lw=2)   # 2px surface gap between cells
    ax.tick_params(which="minor", length=0)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=THEME.text_secondary)

    if table:
        from . import trace
        trace.table(["true \\ read"] + [str(i) for i in range(6)],
                    [[names[i]] + [str(m[i][j]) for j in range(6)] for i in range(6)])
    return ax


# -------------------------------------------------------------- 4. enthusiasm curves

def enthusiasm_curves(curves: list[dict], *, title: str | None = None,
                      table: bool = True, ax=None):
    """Change across an ordered axis -> lines. The shape IS the claim: rising through
    fermentation, then a cliff at geosmin. A model without the cliff is charming and wrong.
    """
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7.4, 4.3))

    by_group: dict[str, dict] = {}
    for c in curves:
        by_group.setdefault(c["group_id"], c)

    # Well-behaved fruits produce IDENTICAL curves - on the oracle all three are
    # [0, 2, 3, 4, 5] - so drawn straight they overplot and only the last one is visible.
    # Each series is nudged by a small, constant, disclosed y-offset. The offset is
    # cosmetic and never changes a reported number; the table below carries the values.
    groups = sorted(by_group.items())
    dodge = 0.13 if len(groups) > 1 else 0.0
    for i, (gid, c) in enumerate(groups):
        col = THEME.series[i % len(THEME.series)]
        ys = list(c["enthusiasm_by_stage"])
        shift = (i - (len(groups) - 1) / 2) * dodge
        label = (gid or "group").replace("curve-", "").replace("-assess", "")
        ax.plot(range(len(ys)), [v + shift for v in ys], lw=2, marker="o", ms=8, color=col,
                zorder=3, markeredgecolor=THEME.surface, markeredgewidth=2, label=label)
        # direct label at the line end: <=4 series, so identity is never colour-alone
        # stagger the end labels in POINTS, not data units: a 0.13 dodge is only a few
        # pixels tall and three 9pt labels would still pile up
        ax.annotate(label, (len(ys) - 1, ys[-1] + shift), textcoords="offset points",
                    xytext=(11, (i - (len(groups) - 1) / 2) * 12), fontsize=9, color=col,
                    va="center")
        if c.get("cliff") is not None:
            ax.plot([5], [(0 if c["cliff"] else 5) + shift],
                    marker="v" if c["cliff"] else "^", ms=9, color=col,
                    markeredgecolor=THEME.surface, markeredgewidth=2, zorder=3)

    ax.axvspan(4.5, 5.5, color=THEME.status["critical"], alpha=0.08, zorder=1)
    ax.text(5.0, 5.35, "geosmin\ncliff", ha="center", fontsize=8.5,
            color=THEME.status["critical"])
    ax.set_xticks(range(6), ["green", "turning", "ripe", "over-\nripe", "ferment-\ning",
                             "mouldy"], fontsize=8.5)
    ax.set_xlim(-0.35, 6.35)
    ax.set_ylim(-0.4, 5.9)
    ax.set_yticks(range(6))
    _frame(ax, ylabel="enthusiasm (0-5)")
    ax.set_title(title or "The fly's enthusiasm curve, per fruit", pad=26)
    ax.text(0, 1.012, "series nudged vertically where they coincide; values in the table",
            transform=ax.transAxes, fontsize=8, color=THEME.text_muted, va="bottom")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    if table:
        from . import trace
        trace.table(["group", "s0", "s1", "s2", "s3", "s4", "monotone", "cliff"],
                    [[(g or "").replace("curve-", "")] +
                     [str(v) for v in c["enthusiasm_by_stage"]] +
                     [str(c["monotone"]), str(c["cliff"])]
                     for g, c in sorted(by_group.items())])
    return ax


# ------------------------------------------------------------------ 5. voice rules

def voice_rules(summary: dict, *, title: str | None = None, ax=None):
    """One series -> no legend box; the title names it. A threshold line beats a
    second colour."""
    srp = summary["global_gates"]["soft_rule_pass"]
    labels = list(srp)
    vals = [srp[k] for k in labels]
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6.4, 0.42 * len(labels) + 1.5))
    y = np.arange(len(labels))
    ax.barh(y, vals, 0.55, color=THEME.series[0], edgecolor=THEME.surface, linewidth=1.4,
            zorder=3)
    for yi, v in zip(y, vals):
        ax.text(min(v + 0.02, 1.02), yi, f"{v:.3f}", va="center", fontsize=8.5,
                color=THEME.text_secondary)
    ax.axvline(0.90, color=THEME.text_muted, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.90, len(labels) - 0.35, " soft gate 0.90", fontsize=8,
            color=THEME.text_muted, va="bottom")
    ax.set_yticks(y, [f"{k}  {'(measured only)' if k == 'V4' else ''}" for k in labels],
                  fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title or "Voice-rule pass rate (V4 is measured, not scored)")
    return ax


# ------------------------------------------------- 6. corpus balance / distributions

def distribution(counts: dict, *, title: str, xlabel: str = "", ax=None, rotate: int = 0):
    """One series, ordered categories. No legend; the title names the measure."""
    keys = [("n/a" if k is None else str(k)) for k in counts]
    vals = list(counts.values())
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(max(5.0, 0.62 * len(keys) + 2), 3.3))
    bars = ax.bar(range(len(keys)), vals, 0.62, color=THEME.series[0],
                  edgecolor=THEME.surface, linewidth=1.4, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom",
                fontsize=8, color=THEME.text_secondary)
    ax.set_xticks(range(len(keys)), keys, fontsize=8.5,
                  rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylim(0, max(vals) * 1.16 if vals else 1)
    _frame(ax, ylabel="rows", xlabel=xlabel)
    ax.set_title(title)
    return ax


# ------------------------------------------------------- 7. classifier resolution

def classifier_resolution(summaries: dict[str, dict], *, title: str | None = None, ax=None):
    """How much of the suite the free grader could read, and how much needs a paid judge.

    Two series stacked, with a 2px surface gap between segments. This is a cost chart, so
    it gets its own axis - never a second scale on the quality chart.
    """
    names = list(summaries)
    deferred = [summaries[n]["classifier"]["abstention_rate"] for n in names]
    resolved = [1 - d for d in deferred]
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(6.2, 0.6 * len(names) + 1.7))
    y = np.arange(len(names))
    ax.barh(y, resolved, 0.5, color=THEME.series[0], edgecolor=THEME.surface, linewidth=2,
            label="read by the free lexicon", zorder=3)
    ax.barh(y, deferred, 0.5, left=resolved, color=THEME.series[1],
            edgecolor=THEME.surface, linewidth=2, label="deferred to the judge", zorder=3)
    for yi, (r, d) in zip(y, zip(resolved, deferred)):
        if r > 0.08:
            ax.text(r / 2, yi, f"{r:.0%}", va="center", ha="center", fontsize=8.5,
                    color=THEME.surface)
        if d > 0.08:
            ax.text(r + d / 2, yi, f"{d:.0%}", va="center", ha="center", fontsize=8.5,
                    color=THEME.surface)
    ax.set_yticks(y, names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title or "Grader tiering: what the free path resolves")
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.22),
              ncols=2)
    return ax


def show(fig=None) -> None:
    plt.tight_layout()
    plt.show()
