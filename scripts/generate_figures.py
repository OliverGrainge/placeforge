"""Generate publication-quality figures from CureVPR ablation results.

Usage:
    python scripts/generate_figures.py

Reads metric YAML files from checkpoints/train/ablations/ and produces
bar charts saved to assets/.
"""

import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = ROOT / "checkpoints" / "train" / "ablations"
ASSETS_DIR = ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "pitts30k_test": "#4C72B0",
    "nordland_test": "#DD8452",
    "average": "#55A868",
}


def load_metrics(name: str) -> dict:
    path = ABLATION_DIR / name / "best.test_metrics.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data


# ---------------------------------------------------------------------------
# 1. Component ablation bar chart
# ---------------------------------------------------------------------------
def plot_component_ablation():
    experiments = [
        ("Raw GPS", "raw_gps"),
        ("Filter Only", "filter_only"),
        ("SG Only", "sg_only"),
        ("CureVPR (Full)", "full"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(experiments))
    width = 0.25

    for i, (dataset, color) in enumerate([
        ("pitts30k_test", COLORS["pitts30k_test"]),
        ("nordland_test", COLORS["nordland_test"]),
        ("average", COLORS["average"]),
    ]):
        vals = []
        for label, key in experiments:
            data = load_metrics(key)
            if dataset == "average":
                val = data["average"]["R@1"] * 100
            else:
                val = data["metrics"][dataset]["R@1"] * 100
            vals.append(val)

        bars = ax.bar(x + i * width, vals, width, label=dataset.replace("_", " ").title(),
                      color=color, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Recall@1 (%)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([e[0] for e in experiments])
    ax.set_ylim(75, 97)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Component Ablation — CureVPR Pipeline", fontweight="bold")

    fig.savefig(ASSETS_DIR / "component_ablation.png")
    plt.close(fig)
    print(f"Saved {ASSETS_DIR / 'component_ablation.png'}")


# ---------------------------------------------------------------------------
# 2. Threshold sensitivity
# ---------------------------------------------------------------------------
def plot_threshold_sensitivity():
    taus = [
        (0.2, "tau02"),
        (0.3, "full"),   # tau=0.3 is the default (full pipeline)
        (0.4, "tau04"),
    ]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for dataset, color, marker in [
        ("pitts30k_test", COLORS["pitts30k_test"], "o"),
        ("nordland_test", COLORS["nordland_test"], "s"),
        ("average", COLORS["average"], "D"),
    ]:
        vals = []
        xs = []
        for tau_val, key in taus:
            data = load_metrics(key)
            if dataset == "average":
                val = data["average"]["R@1"] * 100
            else:
                val = data["metrics"][dataset]["R@1"] * 100
            vals.append(val)
            xs.append(tau_val)

        ax.plot(xs, vals, marker=marker, color=color, linewidth=2, markersize=7,
                label=dataset.replace("_", " ").title())

    ax.set_xlabel("Cosine Similarity Threshold (τ)")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_xticks([0.2, 0.3, 0.4])
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Threshold Sensitivity", fontweight="bold")

    fig.savefig(ASSETS_DIR / "threshold_sensitivity.png")
    plt.close(fig)
    print(f"Saved {ASSETS_DIR / 'threshold_sensitivity.png'}")


# ---------------------------------------------------------------------------
# 3. Supergroup size sensitivity
# ---------------------------------------------------------------------------
def plot_supergroup_sensitivity():
    sizes = [
        (512, "sg512"),
        (1024, "full"),   # sg=1024 is the default (full pipeline)
        (2048, "sg2048"),
    ]

    fig, ax = plt.subplots(figsize=(5, 3.5))

    for dataset, color, marker in [
        ("pitts30k_test", COLORS["pitts30k_test"], "o"),
        ("nordland_test", COLORS["nordland_test"], "s"),
        ("average", COLORS["average"], "D"),
    ]:
        vals = []
        xs = []
        for size_val, key in sizes:
            data = load_metrics(key)
            if dataset == "average":
                val = data["average"]["R@1"] * 100
            else:
                val = data["metrics"][dataset]["R@1"] * 100
            vals.append(val)
            xs.append(size_val)

        ax.plot(xs, vals, marker=marker, color=color, linewidth=2, markersize=7,
                label=dataset.replace("_", " ").title())

    ax.set_xlabel("Supergroup Size")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_xticks([512, 1024, 2048])
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Supergroup Size Sensitivity", fontweight="bold")

    fig.savefig(ASSETS_DIR / "supergroup_sensitivity.png")
    plt.close(fig)
    print(f"Saved {ASSETS_DIR / 'supergroup_sensitivity.png'}")


# ---------------------------------------------------------------------------
# 4. Combined sensitivity (side by side)
# ---------------------------------------------------------------------------
def plot_sensitivity_combined():
    """Threshold and supergroup size plots side-by-side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    # -- Threshold --
    taus = [(0.2, "tau02"), (0.3, "full"), (0.4, "tau04")]
    for dataset, color, marker in [
        ("pitts30k_test", COLORS["pitts30k_test"], "o"),
        ("nordland_test", COLORS["nordland_test"], "s"),
        ("average", COLORS["average"], "D"),
    ]:
        vals = []
        for tau_val, key in taus:
            data = load_metrics(key)
            val = data["average"]["R@1"] * 100 if dataset == "average" else data["metrics"][dataset]["R@1"] * 100
            vals.append(val)
        ax1.plot([0.2, 0.3, 0.4], vals, marker=marker, color=color, linewidth=2,
                 markersize=7, label=dataset.replace("_", " ").title())

    ax1.set_xlabel("Cosine Similarity Threshold (τ)")
    ax1.set_ylabel("Recall@1 (%)")
    ax1.set_xticks([0.2, 0.3, 0.4])
    ax1.set_title("(a) Threshold Sensitivity", fontweight="bold")
    ax1.legend(frameon=False, fontsize=9)

    # -- Supergroup size --
    sizes = [(512, "sg512"), (1024, "full"), (2048, "sg2048")]
    for dataset, color, marker in [
        ("pitts30k_test", COLORS["pitts30k_test"], "o"),
        ("nordland_test", COLORS["nordland_test"], "s"),
        ("average", COLORS["average"], "D"),
    ]:
        vals = []
        for size_val, key in sizes:
            data = load_metrics(key)
            val = data["average"]["R@1"] * 100 if dataset == "average" else data["metrics"][dataset]["R@1"] * 100
            vals.append(val)
        ax2.plot([512, 1024, 2048], vals, marker=marker, color=color, linewidth=2,
                 markersize=7, label=dataset.replace("_", " ").title())

    ax2.set_xlabel("Supergroup Size")
    ax2.set_ylabel("Recall@1 (%)")
    ax2.set_xticks([512, 1024, 2048])
    ax2.set_title("(b) Supergroup Size Sensitivity", fontweight="bold")
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout(w_pad=3)
    fig.savefig(ASSETS_DIR / "sensitivity_combined.png")
    plt.close(fig)
    print(f"Saved {ASSETS_DIR / 'sensitivity_combined.png'}")


if __name__ == "__main__":
    plot_component_ablation()
    plot_threshold_sensitivity()
    plot_supergroup_sensitivity()
    plot_sensitivity_combined()
    print("\nAll figures generated in assets/")
