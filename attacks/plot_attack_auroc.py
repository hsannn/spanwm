import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")

RATIOS = [10, 20, 30, 40, 50]

# Color encodes the METHOD (fixed order, never cycled); the panel encodes the
# attack. This 4-hue set is the only one from the categorical palette that
# clears the all-pairs CVD + normal-vision floors (magenta/yellow/red all
# collide with the orange slot). Aqua sits under 3:1 on white, so the
# end-of-line direct labels below are mandatory relief, not decoration.
METHODS = [
    {"name": "SpanWM v7",      "short": "SpanWM v7", "color": "#2a78d6", "slug": "spanwm_v7"},
    {"name": "SpARK-R (soft)", "short": "SpARK-R",   "color": "#eb6834", "slug": "sparkr_soft"},
    {"name": "EWD",            "short": "EWD",       "color": "#1baf7a", "slug": "ewd"},
    {"name": "IE τ=2.2",  "short": "IE τ=2.2", "color": "#4a3aa7", "slug": "ie_tau2.2"},
]

RESULTS = {
    "Synonym Substitution": {
        "slug": "substitution",
        "auroc": {
            "SpanWM v7":      [0.9425, 0.8648, 0.7935, 0.7184, 0.7006],
            "SpARK-R (soft)": [0.9276, 0.9115, 0.8852, 0.8630, 0.8317],
            "EWD":            [0.9942, 0.9925, 0.9854, 0.9750, 0.9465],
            "IE τ=2.2":  [0.9620, 0.9617, 0.9403, 0.9064, 0.8705],
        },
    },
    "Word Deletion": {
        "slug": "deletion",
        "auroc": {
            "SpanWM v7":      [0.9330, 0.8361, 0.7444, 0.6635, 0.5980],
            "SpARK-R (soft)": [0.9346, 0.9246, 0.9166, 0.9023, 0.8872],
            "EWD":            [0.9945, 0.9942, 0.9909, 0.9824, 0.9606],
            "IE τ=2.2":  [0.9647, 0.9488, 0.9225, 0.8875, 0.8221],
        },
    },
}

TEXT = "#0b0b0b"
MUTED = "#8a8a85"
GRID = "#e6e6e2"
YLIM = (0.45, 1.0)
XLIM = (5, 72)          # right of 50 is the direct-label gutter
LABEL_X = 52
MIN_LABEL_GAP = 0.034   # data units; keeps stacked end labels legible


def _declutter(values):
    """Nudge end-label y positions apart, keeping their top-to-bottom order."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    placed = dict()
    prev = None
    for i in order:
        y = values[i]
        if prev is not None and prev - y < MIN_LABEL_GAP:
            y = prev - MIN_LABEL_GAP
        placed[i] = y
        prev = y
    return placed


def draw(ax, name, spec, legend=True):
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("chance", (XLIM[0], 0.5), textcoords="offset points", xytext=(4, 5),
                ha="left", fontsize=9, color=MUTED)

    for m in METHODS:
        color = m["color"]
        ax.plot(RATIOS, spec["auroc"][m["name"]], color=color, lw=2, marker="o", ms=8,
                mfc=color, mec="white", mew=2, zorder=3, clip_on=False,
                label=m["name"])

    # direct labels at the line ends — identity is never carried by color alone
    ends = [spec["auroc"][m["name"]][-1] for m in METHODS]
    label_y = _declutter(ends)
    for i, m in enumerate(METHODS):
        ax.annotate(f"{m['short']}  {ends[i]:.3f}", (LABEL_X, label_y[i]),
                    va="center", ha="left", fontsize=9, color=TEXT, zorder=4,
                    annotation_clip=False)

    ax.set_title(name, fontsize=12, color=TEXT, pad=34 if legend else 14, loc="left")
    ax.set_xlabel("attack ratio (%)", fontsize=10, color=TEXT)
    ax.set_xticks(RATIOS)
    ax.set_ylim(*YLIM)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlim(*XLIM)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(TEXT)

    if legend:
        # header row between title and plot — the plot area itself has no
        # reliably empty corner (the chance line and labels use them all).
        leg = ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0,
                        ncol=len(METHODS), frameon=False, fontsize=9,
                        handlelength=1.4, columnspacing=1.2, handletextpad=0.5)
        for t in leg.get_texts():
            t.set_color(TEXT)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # combined: two panels, shared y-scale, one shared legend
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (name, spec) in zip(axes, RESULTS.items()):
        draw(ax, name, spec, legend=False)
    axes[0].set_ylabel("AUROC", fontsize=10, color=TEXT)
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.045, 0.95),
                     ncol=len(METHODS), frameon=False, fontsize=10, handlelength=1.6,
                     columnspacing=1.8)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.suptitle("Detection AUROC under attack (C4, n=200)",
                 fontsize=13, color=TEXT, x=0.046, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = os.path.join(OUT_DIR, "v7_attack_auroc.png")
    fig.savefig(path, dpi=200, facecolor="white")
    print("wrote", path)
    plt.close(fig)

    # standalone panels
    for name, spec in RESULTS.items():
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        draw(ax, name, spec)
        ax.set_ylabel("AUROC", fontsize=10, color=TEXT)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"v7_attack_auroc_{spec['slug']}.png")
        fig.savefig(path, dpi=200, facecolor="white")
        print("wrote", path)
        plt.close(fig)


if __name__ == "__main__":
    main()
