"""TPR at fixed FPR operating points, one panel per dataset.

x = FPR operating point (10 / 5 / 1 / 0.1 %), y = TPR at that point.
Color encodes the METHOD (fixed order, never cycled); marker + line style
encode the paraphrasing ATTACK model. Only the two attack models the user
asked for (gpt-5-mini, gpt-oss-20b) are plotted; gemma-4-12b-it is dropped.

One figure per source table in RUNS -> outputs/figures/tpr_by_fpr_<key>.{pdf,png}
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# embed real TrueType outlines in the PDF (Type 3 is rejected by most
# paper submission systems)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")

# FPR operating points, plotted on evenly spaced categorical slots (the real
# spacing is decades apart; a linear axis would crush 1% and 0.1% together).
FPRS = ["10%", "5%", "1%", "0.1%"]
X = list(range(len(FPRS)))

# Slots 1/2/3 of the categorical palette — the subset documented as clearing
# the all-pairs CVD + normal-vision floors, so the three curves stay separable
# for colorblind readers. Fixed order, never cycled; OURS is listed last so it
# draws on top of the baselines. (SWEET is still in RUNS below — drop a method
# from this list and it simply stops being plotted.)
METHODS = [
    {"name": "SpARK-R (soft)", "color": "#2a78d6"},
    {"name": "IE τ=2.2",       "color": "#1baf7a"},
    # ours: the warm hue against two cool baselines, plus a heavier stroke
    {"name": "OURS",           "color": "#eb6834", "emphasis": True},
]

# Second channel: shape (+ line style as reinforcement), so identity never
# rests on color alone.
ATTACKS = [
    {"name": "gpt-5-mini",  "marker": "o", "ls": "-"},
    {"name": "gpt-oss-20b", "marker": "s", "ls": (0, (5, 2))},
]

NA = float("nan")  # cell missing/unusable in the source table -> gap in the line

# Panel titles. RUNS keeps the table's own short dataset keys; this maps them to
# the names the paper uses.
DATASET_LABELS = {
    "c4": "C4",
    "daily mail": "CNN Daily Mail",
    "wm16": "wmt16",
}

# TPR @ FPR = 10 / 5 / 1 / 0.1 %, transcribed from the result tables.
RUNS = {
    "table1": {
        "c4": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.5900, 0.4950, 0.2300, 0.0500],
                "gpt-oss-20b": [0.4250, 0.3500, 0.1250, 0.0750],
            },
            "SWEET": {
                "gpt-5-mini":  [0.8200, 0.7550, 0.5800, 0.3850],
                "gpt-oss-20b": [0.7750, 0.7100, 0.5550, 0.4400],
            },
            "IE τ=2.2": {
                "gpt-5-mini":  [0.8000, 0.7400, 0.5600, 0.1550],
                "gpt-oss-20b": [0.7450, 0.6950, 0.5800, 0.2900],
            },
            "OURS": {
                "gpt-5-mini":  [0.8780, 0.7700, 0.7350, 0.6120],
                "gpt-oss-20b": [0.8960, 0.8050, 0.7550, 0.6880],
            },
        },
        "daily mail": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.2200, 0.1300, 0.0450, 0.0050],
                "gpt-oss-20b": [0.2150, 0.1600, 0.0500, 0.0000],
            },
            "SWEET": {
                "gpt-5-mini":  [0.4850, 0.4400, 0.3050, 0.2500],
                "gpt-oss-20b": [0.4850, 0.4450, 0.2950, 0.2350],
            },
            "IE τ=2.2": {
                "gpt-5-mini":  [0.5600, 0.5350, 0.4400, 0.4250],
                "gpt-oss-20b": [0.5350, 0.5000, 0.4200, 0.4200],
            },
            "OURS": {
                "gpt-5-mini":  [0.2480, 0.1530, 0.0250, 0.0010],
                "gpt-oss-20b": [0.1430, 0.0780, 0.0150, 0.0050],
            },
        },
        "wm16": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.5900, 0.4500, 0.2800, 0.1980],
                "gpt-oss-20b": [0.6350, 0.5000, 0.3400, 0.2400],
            },
            "SWEET": {
                "gpt-5-mini":  [0.8800, 0.8600, 0.7450, 0.6150],
                "gpt-oss-20b": [0.9050, 0.9000, 0.8150, 0.7400],
            },
            "IE τ=2.2": {
                "gpt-5-mini":  [0.9500, 0.8550, 0.7500, 0.6650],
                "gpt-oss-20b": [0.8800, 0.8050, 0.7700, 0.6850],
            },
            "OURS": {
                "gpt-5-mini":  [0.9050, 0.8500, 0.7050, 0.5210],
                "gpt-oss-20b": [0.9100, 0.8500, 0.7550, 0.6850],
            },
        },
    },
    "table2": {
        "c4": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.5600, 0.4250, 0.1000, 0.0250],
                "gpt-oss-20b": [0.6700, 0.5450, 0.2550, 0.0750],
            },
            "SWEET": {
                "gpt-5-mini":  [0.8000, 0.7500, 0.5750, 0.5300],
                "gpt-oss-20b": [0.8150, 0.7550, 0.5750, 0.5560],
            },
            "IE τ=2.2": {
                "gpt-5-mini":  [0.8500, 0.6900, 0.5450, 0.4300],
                "gpt-oss-20b": [0.8150, 0.7150, 0.5800, 0.4900],
            },
            "OURS": {
                "gpt-5-mini":  [0.8367, 0.7575, 0.5950, 0.4320],
                "gpt-oss-20b": [0.8500, 0.7850, 0.6300, 0.4860],
            },
        },
        "daily mail": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.2550, 0.1350, 0.0450, 0.0350],
                "gpt-oss-20b": [0.2500, 0.1250, 0.0600, 0.0500],
            },
            "SWEET": {
                "gpt-5-mini":  [0.5200, 0.4650, 0.3200, 0.2850],
                "gpt-oss-20b": [0.4550, 0.4200, 0.3350, 0.2950],
            },
            "IE τ=2.2": {
                "gpt-5-mini":  [0.6350, 0.5950, 0.4850, 0.3600],
                "gpt-oss-20b": [0.5900, 0.5650, 0.4700, 0.3500],
            },
            "OURS": {
                "gpt-5-mini":  [0.8250, 0.7110, 0.6200, 0.3560],
                "gpt-oss-20b": [0.8337, 0.7370, 0.6750, 0.4360],
            },
        },
        "wm16": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.7950, 0.6400, 0.4150, 0.1750],
                "gpt-oss-20b": [0.8550, 0.7600, 0.5950, 0.3060],
            },
            "SWEET": {
                "gpt-5-mini":  [0.9100, 0.8950, 0.8300, 0.7950],
                "gpt-oss-20b": [0.9300, 0.9000, 0.8700, 0.8450],
            },
            "IE τ=2.2": {
                # NOTE: in the source table this gpt-oss-20b row sits far below
                # both its gpt-5-mini and gemma rows (AUROC 0.8082 vs 0.9585 /
                # 0.9562) — check whether the oss/gemma rows were swapped.
                "gpt-5-mini":  [0.9150, 0.8800, 0.8250, 0.5150],
                "gpt-oss-20b": [0.5550, 0.4050, 0.3050, 0.0700],
            },
            "OURS": {
                "gpt-5-mini":  [0.8055, 0.7550, 0.6900, 0.5575],
                "gpt-oss-20b": [0.8100, 0.7700, 0.7050, 0.5845],
            },
        },
    },
    "table3": {
        "c4": {
            "SpARK-R (soft)": {
                # NOTE: the 0.1% cell of this row reads 1.3584 in the source
                # table — not a TPR (>1); the row looks shifted one column
                # right (its z / p cells both read 0.3433). Left as a gap.
                "gpt-5-mini":  [0.3150, 0.1950, 0.0200, NA],
                "gpt-oss-20b": [0.3800, 0.2325, 0.1600, 0.0300],
            },
            "SWEET": {
                "gpt-5-mini":  [0.9150, 0.8800, 0.8300, 0.6150],
                "gpt-oss-20b": [0.9050, 0.8850, 0.7950, 0.6100],
            },
            "IE τ=2.2": {
                "gpt-5-mini":  [0.8450, 0.7550, 0.5650, 0.3750],
                "gpt-oss-20b": [0.7800, 0.7300, 0.5150, 0.3950],
            },
            "OURS": {
                "gpt-5-mini":  [0.8783, 0.8600, 0.7150, 0.2090],
                "gpt-oss-20b": [0.9000, 0.8888, 0.7600, 0.3450],
            },
        },
        "daily mail": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.1900, 0.1300, 0.0500, 0.0000],
                "gpt-oss-20b": [0.2250, 0.1500, 0.0400, 0.0100],
            },
            "SWEET": {
                "gpt-5-mini":  [0.4500, 0.4050, 0.3400, 0.2050],
                "gpt-oss-20b": [0.4900, 0.4350, 0.3350, 0.1800],
            },
            "IE τ=2.2": {
                "gpt-5-mini":  [0.5400, 0.5100, 0.4500, 0.3550],
                "gpt-oss-20b": [0.4700, 0.4350, 0.3950, 0.3450],
            },
            "OURS": {
                "gpt-5-mini":  [0.2630, 0.1840, 0.0450, 0.0160],
                "gpt-oss-20b": [0.1610, 0.0810, 0.0130, 0.0020],
            },
        },
        "wm16": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.8000, 0.7050, 0.4500, 0.3000],
                "gpt-oss-20b": [0.8450, 0.7500, 0.5750, 0.4620],
            },
            "SWEET": {
                "gpt-5-mini":  [0.9450, 0.8850, 0.8200, 0.6950],
                "gpt-oss-20b": [0.9200, 0.8850, 0.8400, 0.7550],
            },
            "IE τ=2.2": {
                "gpt-5-mini":  [0.8750, 0.8350, 0.5150, 0.0250],
                "gpt-oss-20b": [0.8650, 0.8050, 0.5850, 0.0605],
            },
            "OURS": {
                "gpt-5-mini":  [0.9083, 0.8150, 0.6950, 0.5320],
                # NOTE: 5% (0.5400) < 1% (0.7750) in the source table — TPR
                # cannot rise as FPR tightens; transcribed as written.
                "gpt-oss-20b": [0.8900, 0.5400, 0.7750, 0.6700],
            },
        },
    },
    "table4": {
        "c4": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.5300, 0.4050, 0.2050, 0.0400],
                "gpt-oss-20b": [0.6200, 0.4650, 0.2900, 0.0710],
            },
            "SWEET": {
                "gpt-5-mini":  [0.8650, 0.7850, 0.5500, 0.4800],
                "gpt-oss-20b": [0.8450, 0.7500, 0.5750, 0.4850],
            },
            "IE τ=2.2": {
                # NOTE: same oss-below-gemma pattern as table2's wm16 IE row
                # (AUROC 0.8051 vs gemma 0.9300) — possible row swap.
                "gpt-5-mini":  [0.8600, 0.7550, 0.6500, 0.5850],
                "gpt-oss-20b": [0.5300, 0.3650, 0.2500, 0.1550],
            },
            "OURS": {
                "gpt-5-mini":  [0.8755, 0.7825, 0.5700, 0.5020],
                "gpt-oss-20b": [0.8250, 0.7525, 0.5950, 0.4220],
            },
        },
        "daily mail": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.2600, 0.1300, 0.0450, 0.0100],
                "gpt-oss-20b": [0.2600, 0.1750, 0.0450, 0.0250],
            },
            "SWEET": {
                "gpt-5-mini":  [0.4750, 0.4300, 0.2900, 0.1950],
                "gpt-oss-20b": [0.4500, 0.3950, 0.2850, 0.2150],
            },
            "IE τ=2.2": {
                "gpt-5-mini":  [0.4750, 0.4300, 0.3500, 0.2850],
                "gpt-oss-20b": [0.4750, 0.4050, 0.3700, 0.3100],
            },
            "OURS": {
                "gpt-5-mini":  [0.8038, 0.7450, 0.5750, 0.4970],
                "gpt-oss-20b": [0.8458, 0.8000, 0.6900, 0.6260],
            },
        },
        "wm16": {
            "SpARK-R (soft)": {
                "gpt-5-mini":  [0.7400, 0.6650, 0.5150, 0.4520],
                "gpt-oss-20b": [0.8750, 0.8100, 0.6400, 0.5700],
            },
            "SWEET": {
                "gpt-5-mini":  [0.9000, 0.8550, 0.8150, 0.7050],
                "gpt-oss-20b": [0.9250, 0.8750, 0.8700, 0.7850],
            },
            "IE τ=2.2": {
                "gpt-5-mini":  [0.8050, 0.6600, 0.4050, 0.1950],
                "gpt-oss-20b": [0.8350, 0.7550, 0.4900, 0.3100],
            },
            "OURS": {
                "gpt-5-mini":  [0.8811, 0.8233, 0.7500, 0.6810],
                "gpt-oss-20b": [0.8900, 0.8400, 0.8000, 0.7400],
            },
        },
    },
}

# No tinted canvas: the figure and axes are plain white, so the plot drops onto
# a paper page without a visible panel behind it. SURFACE is only used for the
# 2px ring that keeps overlapping markers legible.
SURFACE = "#ffffff"
TEXT = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Type sizes — pushed as large as the panels allow while every label still
# clears its neighbour.
FS_TITLE = 22
FS_AXIS_LABEL = 20
FS_TICK = 18
FS_LEGEND = 19


def draw(ax, results, dataset):
    for m in METHODS:
        ours = m.get("emphasis", False)
        for a in ATTACKS:
            ax.plot(X, results[dataset][m["name"]][a["name"]],
                    color=m["color"], lw=3.5 if ours else 2.5, ls=a["ls"],
                    marker=a["marker"], ms=12 if ours else 10, mfc=m["color"],
                    mec=SURFACE, mew=2, solid_capstyle="round",
                    zorder=4 if ours else 3, clip_on=False)

    ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=FS_TITLE,
                 color=TEXT, pad=12, loc="center")
    ax.set_xticks(X)
    ax.set_xticklabels(FPRS)
    ax.set_xlim(-0.25, len(X) - 0.75)
    # small bottom margin so the near-zero daily-mail points sit above the
    # baseline instead of on top of the x tick labels
    ax.set_ylim(-0.035, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("FPR", fontsize=FS_AXIS_LABEL, color=TEXT, labelpad=8)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=FS_TICK, length=0, pad=7)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(TEXT)


def render(key, results):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), sharey=True)
    for ax, dataset in zip(axes, results):
        draw(ax, results, dataset)
    axes[0].set_ylabel("TPR", fontsize=FS_AXIS_LABEL, color=TEXT, labelpad=8)

    # Both channels legended together — hue = method, shape = attack model —
    # parked inside the last panel's empty lower-left corner so the header band
    # is free and the panels get the height instead. Opaque box: where a line
    # does run under it, the labels still read.
    method_keys = [Line2D([], [], color=m["color"],
                          lw=4 if m.get("emphasis") else 3, label=m["name"])
                   for m in METHODS]
    attack_keys = [Line2D([], [], color=MUTED, lw=2.5, ls=a["ls"], marker=a["marker"],
                          ms=10, mfc=MUTED, mec=SURFACE, mew=1.5, label=a["name"])
                   for a in ATTACKS]

    leg = axes[-1].legend(handles=method_keys + attack_keys, loc="lower left",
                          bbox_to_anchor=(-0.01, -0.02), ncol=1, fontsize=FS_LEGEND,
                          frameon=True, facecolor=SURFACE, edgecolor="none",
                          framealpha=0.92, borderpad=0.5, labelspacing=0.35,
                          handlelength=2.2, handletextpad=0.6)
    leg.set_zorder(5)
    for t in leg.get_texts():
        t.set_color(TEXT)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"tpr_by_fpr_{key}.{ext}")
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print("wrote", path)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for key, results in RUNS.items():
        render(key, results)


if __name__ == "__main__":
    main()
