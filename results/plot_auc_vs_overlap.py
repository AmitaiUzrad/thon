#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def capped_auc_yerr(mean: pd.Series, std: pd.Series) -> np.ndarray:
    """Asymmetric yerr so mean ± deviation stays in [0, 1]."""
    s = std.fillna(0.0).to_numpy()
    m = mean.to_numpy()
    return np.array([np.minimum(s, m), np.minimum(s, 1.0 - m)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AUC vs overlap for THGN/TCEN with mean and std across runs."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results_summary.csv",
        help="Path to results summary CSV.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NDC-classes",
        help="Dataset to filter from the CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="auc_vs_overlap_ndc_classes.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == args.dataset].copy()
    if df.empty:
        raise ValueError(f"No rows found for dataset '{args.dataset}' in {csv_path}.")

    grouped = (
        df.groupby(["model", "overlap"], as_index=False)
        .agg(
            old_auc_mean=("old_auc", "mean"),
            old_auc_std=("old_auc", "std"),
            new_auc_mean=("new_auc", "mean"),
            new_auc_std=("new_auc", "std"),
        )
        .sort_values(["model", "overlap"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    color_map = {"thgn": "blue", "tcen": "red"}
    label_map = {"thgn": "THGN", "tcen": "TCEN"}

    for model in ["thgn", "tcen"]:
        mdf = grouped[grouped["model"] == model].sort_values("overlap")
        if mdf.empty:
            continue
        x = mdf["overlap"].to_numpy()

        # Left subplot: Transductive (old_auc)
        axes[0].errorbar(
            x,
            mdf["old_auc_mean"].to_numpy(),
            yerr=capped_auc_yerr(mdf["old_auc_mean"], mdf["old_auc_std"]),
            color=color_map[model],
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=label_map[model],
        )

        # Right subplot: Inductive (new_auc)
        axes[1].errorbar(
            x,
            mdf["new_auc_mean"].to_numpy(),
            yerr=capped_auc_yerr(mdf["new_auc_mean"], mdf["new_auc_std"]),
            color=color_map[model],
            linestyle="--",
            marker="o",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=label_map[model],
        )

    axes[0].set_xlabel("Overlap")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Transductive")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Overlap")
    axes[1].set_title("Inductive")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"AUC vs Overlap ({args.dataset})")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")


if __name__ == "__main__":
    main()
