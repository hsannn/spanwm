import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")

RATIOS = [10, 20, 30, 40, 50]
RESULTS = {
    "Synonym Substitution": {
        "auroc": [0.9425, 0.8648, 0.7935, 0.7184, 0.7006],
        "color": "#2a78d6",
        "slug": "substitution",
    },
    "Word Deletion": {
        "auroc": [0.9330, 0.8361, 0.7444, 0.6635, 0.5980],
        "color": "#eb6834",
        "slug": "deletion",
    },
}

TEXT = "#0b0b0b"
MUTED = "#8a8a85"
GRID = "#e6e6e2"
YLIM = (0.45, 1.0)


def draw(ax, name, spec):
    y = spec["auroc"]
    color = spec["color"]

    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("chance", (5, 0.5), textcoords="offset points", xytext=(4, 5),
                ha="left", fontsize=9, color=MUTED)
    ax.plot(RATIOS, y, color=color, lw=2, marker="o", ms=7,
            mfc=color, mec="white", mew=1.5, zorder=3, clip_on=False)

    # selective direct labels: endpoints only
    for i in (0, len(RATIOS) - 1):
        ax.annotate(f"{y[i]:.3f}", (RATIOS[i], y[i]),
                    textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=9, color=TEXT)

    ax.set_title(name, fontsize=12, color=TEXT, pad=14, loc="left")
    ax.set_xlabel("attack ratio (%)", fontsize=10, color=TEXT)
    ax.set_xticks(RATIOS)
    ax.set_ylim(*YLIM)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlim(5, 55)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(TEXT)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # combined: two panels, shared y-scale
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (name, spec) in zip(axes, RESULTS.items()):
        draw(ax, name, spec)
    axes[0].set_ylabel("AUROC", fontsize=10, color=TEXT)
    fig.suptitle("SpanWM v7 — detection AUROC under attack (C4, n=200)",
                 fontsize=13, color=TEXT, x=0.055, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(OUT_DIR, "v7_attack_auroc.png")
    fig.savefig(path, dpi=200, facecolor="white")
    print("wrote", path)
    plt.close(fig)

    # standalone panels
    for name, spec in RESULTS.items():
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        draw(ax, name, spec)
        ax.set_ylabel("AUROC", fontsize=10, color=TEXT)
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"v7_attack_auroc_{spec['slug']}.png")
        fig.savefig(path, dpi=200, facecolor="white")
        print("wrote", path)
        plt.close(fig)


if __name__ == "__main__":
    main()
