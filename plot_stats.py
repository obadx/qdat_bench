import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D
from datasets import load_dataset, Dataset
import numpy as np

from qdat_bench.data_models import NoonMoshaddahLen, NoonMokhfahLen, Qalqalah


def group_avg_metrics_for_violin(
    metric_names: list[str],
) -> dict[str, list[str]]:
    """
    Group avg metrics into categories for violin plot visualization.

    Args:
        metric_names: List of all metric names from avg_metrics

    Returns:
        Dictionary with three categories:
        - per_metrics: Phoneme Error Rate metrics
        - rmse_metrics: Root Mean Squared Error metrics
        - percentage_metrics: Metrics like recall, precision, f1, accuracy
    """
    per_metrics = []
    rmse_metrics = []
    percentage_metrics = []

    for name in metric_names:
        name_lower = name.lower()

        if "per" in name_lower:
            per_metrics.append(name)
        elif "rmse" in name_lower:
            rmse_metrics.append(name)
        elif any(x in name_lower for x in ["f1", "acc"]):
            percentage_metrics.append(name)

    return {
        "per_metrics": sorted(per_metrics),
        "rmse_metrics": sorted(rmse_metrics),
        "percentage_metrics": sorted(percentage_metrics),
    }


def plot_bootstrap_violin(
    bootstrap_samples_path: str,
    dir: str = "assets",
    title: str = "Bootstrap Analysis of qdat_bench Average Metrics",
) -> None:
    """
    Create violin plots for bootstrapped average metrics.

    Args:
        bootstrap_samples_path: Path to the JSON file containing bootstrap samples
        dir: Directory to save the plot
        title: Title for the figure
    """
    os.makedirs(dir, exist_ok=True)

    with open(bootstrap_samples_path, "r") as f:
        bootstrap_samples = json.load(f)

    if not bootstrap_samples:
        print("No bootstrap samples found.")
        return

    metric_names = list(bootstrap_samples[0].keys())
    metric_categories = group_avg_metrics_for_violin(metric_names)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    categories = [
        ("per_metrics", "PER Metrics (lower is better)", "Blues"),
        ("rmse_metrics", "RMSE Metrics (lower is better)", "Oranges"),
        ("percentage_metrics", "Percentage Metrics (higher is better)", "Greens"),
    ]

    for ax, (cat_key, cat_title, cmap) in zip(axes, categories):
        metrics = metric_categories[cat_key]

        if not metrics:
            ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
            ax.set_title(cat_title)
            ax.axis("off")
            continue

        data = []
        labels = []
        for metric in metrics:
            values = [sample[metric] for sample in bootstrap_samples]
            data.append(values)
            labels.append(metric.replace("_", "\n"))

        parts = ax.violinplot(
            data, positions=range(len(metrics)), showmeans=True, showmedians=True
        )

        cmap_obj = plt.colormaps.get(cmap)
        for pc in parts["bodies"]:
            pc.set_facecolor(cmap_obj(0.6))
            pc.set_alpha(0.7)

        parts["cmeans"].set_color("red")
        parts["cmedians"].set_color("blue")

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(cat_title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)

    mean_line = Line2D([0], [0], color="red", linewidth=2, label="Bootstrap Mean")
    median_line = Line2D(
        [0], [0], color="blue", linestyle="--", linewidth=2, label="Bootstrap Median"
    )

    fig.legend(
        handles=[mean_line, median_line],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.88),
        fontsize=12,
        frameon=True,
        framealpha=0.9,
    )

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(dir, "bootstrap_violin_plots.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Violin plots saved to {save_path}")
    plt.show()


def plot_age_gender(ages, gender, dir="assets"):
    # Create figure with subplots
    os.makedirs(dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram for ages
    ax1.hist(ages, bins=10, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Age")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Age Distribution")
    ax1.grid(True, alpha=0.3)

    # Histogram for gender
    gender_counts = [gender.count("female"), gender.count("male")]
    gender_labels = ["Female", "Male"]
    colors = ["lightpink", "lightblue"]

    ax2.bar(gender_labels, gender_counts, color=colors, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Gender")
    ax2.set_ylabel("Count")
    ax2.set_title("Gender Distribution")

    # Add count labels on top of bars
    for i, count in enumerate(gender_counts):
        ax2.text(i, count + 0.1, str(count), ha="center", va="bottom")

    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f"{dir}/age_gender_histograms.png", dpi=300, bbox_inches="tight")


COL_TO_GOLDEN_VALS = {
    "qalo_alif_len": {2},
    "qalo_waw_len": {2},
    "laa_alif_len": {2},
    "separate_madd": {2, 4, 5},
    "noon_moshaddadah_len": {NoonMoshaddahLen.COMPLETE},
    "noon_mokhfah_len": {NoonMokhfahLen.COMPLETE},
    "allam_alif_len": {2},
    "madd_aared_len": {2, 4, 6},
    "qalqalah": {Qalqalah.HAS_QALQALAH},
}

COL_TO_STR_LABLE = {
    "noon_moshaddadah_len": NoonMoshaddahLen,
    "noon_mokhfah_len": NoonMokhfahLen,
    "qalqalah": Qalqalah,
}
for col in COL_TO_STR_LABLE:
    COL_TO_STR_LABLE[col] = {
        member.value: member.name for member in COL_TO_STR_LABLE[col]
    }


def is_correct_recitation(item: dict) -> bool:
    cond = True
    for key, golen_vals in COL_TO_GOLDEN_VALS.items():
        cond = cond and item[key] in golen_vals
    return cond


def plot_corrent_recitations(ds: Dataset, dir="assets"):
    # Create figure with subplots
    os.makedirs(dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Histogram
    correct_count = 0

    for item in ds:
        if is_correct_recitation(item):
            correct_count += 1
    incorrect_count = len(ds) - correct_count

    counts_tuple = correct_count, incorrect_count
    labels = ["Correct", "Has Errors"]
    colors = ["lightgreen", "lightpink"]

    ax.bar(labels, counts_tuple, color=colors, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Correctness")
    ax.set_ylabel("Count")
    ax.set_title("Hitogram of Correct/Incorrect Recitations")

    # Add count labels on top of bars
    for i, count in enumerate(counts_tuple):
        ax.text(i, count + 0.1, str(count), ha="center", va="bottom")

    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f"{dir}/correctness_histogram.png", dpi=300, bbox_inches="tight")


def plot_tajweed_columns_histogram(ds: Dataset, dir="assets", num_cols=3):
    os.makedirs(dir, exist_ok=True)

    num_raws = int(np.ceil(len(COL_TO_GOLDEN_VALS) / num_cols))
    total_plots = num_raws * num_cols
    fig, axes = plt.subplots(
        num_raws,
        num_cols,
        figsize=(6 * num_cols, 5 * num_raws),
    )

    axes = axes.reshape((-1,))

    col_to_val_to_counts = {}
    for col in COL_TO_GOLDEN_VALS:
        col_to_val_to_counts[col] = {}
        for val in ds[col]:
            if val not in col_to_val_to_counts[col]:
                col_to_val_to_counts[col][val] = 0
            col_to_val_to_counts[col][val] += 1

    # Histogram
    for idx, col in enumerate(COL_TO_GOLDEN_VALS):
        col_to_val_to_counts[col] = dict(sorted(col_to_val_to_counts[col].items()))
        labels = list(col_to_val_to_counts[col].keys())
        if col in COL_TO_STR_LABLE:
            labels = [COL_TO_STR_LABLE[col][label] for label in labels]
        counts = list(col_to_val_to_counts[col].values())
        colors = [
            "lightgreen" if val in COL_TO_GOLDEN_VALS[col] else "lightpink"
            for val in col_to_val_to_counts[col].keys()
        ]
        axes[idx].bar(
            [str(l) for l in labels], counts, color=colors, edgecolor="black", alpha=0.7
        )
        axes[idx].set_xlabel("col", fontsize=14)
        axes[idx].set_ylabel("Frequency", fontsize=14)
        axes[idx].set_title(f"{col} Distribution", fontsize=16)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].tick_params(axis="x", labelsize=14)

        # Add count labels on top of bars
        for i, count in enumerate(counts):
            axes[idx].text(i, count + 0.1, str(count), ha="center", va="bottom")

    # Create color legend patches for the entire figure
    correct_patch = patches.Patch(color="lightgreen", label="Correct", alpha=0.7)
    wrong_patch = patches.Patch(
        color="lightpink", label="Wrong", alpha=0.7
    )  # Add the color legend to the top right of the entire figure

    # Add main title to the whole figure
    fig.suptitle(
        "Tajweed Rules Correct/Wrong Distribution Analysis for Qdat Bench",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )

    fig.legend(
        handles=[correct_patch, wrong_patch],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.94),
        frameon=True,
        framealpha=0.9,
        fontsize=14,
    )

    plt.tight_layout()

    # Save the combined plot
    plt.savefig(f"{dir}/tajweed_columns_histograms.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Plot statistics for QDAT Bench")
    parser.add_argument(
        "--bootstrap-samples",
        help="Path to bootstrap samples JSON file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save-dir",
        help="Directory to save plots",
        type=str,
        default="assets",
    )
    parser.add_argument(
        "--plot-type",
        help="Type of plot to generate",
        choices=["bootstrap_violin", "dataset_stats", "all"],
        default="all",
    )

    args = parser.parse_args()

    if args.plot_type in ["bootstrap_violin", "all"]:
        if args.bootstrap_samples:
            plot_bootstrap_violin(
                bootstrap_samples_path=args.bootstrap_samples,
                dir=args.save_dir,
            )
        else:
            print("Warning: --bootstrap-samples required for violin plots. Skipping.")

    if args.plot_type in ["dataset_stats", "all"]:
        ds = load_dataset("obadx/qdat_bench")["train"]
        ages = ds["age"]
        gender = ds["gender"]
        plot_age_gender(ages=ages, gender=gender, dir=args.save_dir)
        plot_corrent_recitations(ds, dir=args.save_dir)
        plot_tajweed_columns_histogram(ds, dir=args.save_dir)
