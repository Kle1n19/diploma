import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

COLORS = {
    "Grid": "#7f7f7f",
    "Random": "#1f77b4",
    "ε-greedy":"#ff7f0e",
    "UCB1":"#2ca02c",
    "Softmax":"#d62728",
    "Thompson":"#9467bd",
    "TPE":"#8c564b",
    "SAC":"#e377c2",
    "GRPO":"#17becf",
}

LINE_STYLES = {
    "Grid": (0, (4, 2)),
    "Random": (0, (2, 1)),
    "ε-greedy": "solid",
    "UCB1": "solid",
    "Softmax": "solid",
    "Thompson": "solid",
    "TPE": "solid",
    "SAC": "solid",
    "GRPO": "solid",
}


def set_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.3)


def save_fig(fig, path: Path, dpi: int = 150):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")
