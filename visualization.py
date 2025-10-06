"""
Plotting utilities for similarity summaries and matrices.

This module provides functions to visualise the summarised similarity
scores produced by ``transform_samples`` as well as the original
similarity matrices themselves.  The goal of these visualisations is
to support interpretation of the statistical results by giving an
intuitive overview of the distributions and inter‑image relationships.

Only matplotlib is used for plotting to conform with the notebook
guidelines: each chart is drawn on its own figure, no seaborn is
utilised, and no specific colour schemes are specified (matplotlib
defaults are used).  The caller must provide an output directory
where PNG files will be written.
"""

from __future__ import annotations

import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from load_validate import load_matrices

MAX_TICK_LABELS = 12


# Pick a fixed color per model/SD version
MODEL_COLORS = {
    "SD1.5": "#1f77b4",   # blue
    "SD2.1": "#2ca02c",   # green
    "SDXL":  "#ff7f0e",   # orange
    "SD3.5": "#9467bd",   # purple
}
def _model_color(m: str) -> str:
    return MODEL_COLORS.get(m, "#7f7f7f")  # fallback grey

# Desired chronological order for SD versions
MODEL_ORDER = ["SD1.5", "SD2.1", "SDXL", "SD3.5"]
def _model_sort_key(m: str) -> int:
    return MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER)


def plot_distributions(summary_path: str, out_dir: str) -> None:
    """Create boxplots of summary scores grouped by model and condition."""
    os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(summary_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(summary_path)
    elif ext == ".parquet":
        df = pd.read_parquet(summary_path)
    else:
        raise ValueError("Unsupported summary file format; use .csv or .parquet")

    required = {"metric", "model", "condition", "summary_score"}
    if not required.issubset(df.columns):
        raise KeyError(f"Summary file missing columns: {required - set(df.columns)}")

    all_stats = []  # collect rows

    for metric in sorted(df["metric"].dropna().unique()):
        df_metric = df[df["metric"] == metric]

        groups, labels, colors = [], [], []
        models_seen_order = []

        for model in sorted(df_metric["model"].dropna().unique(), key=_model_sort_key):
            for cond in sorted(df_metric["condition"].dropna().unique()):
                vals = (
                    df_metric[
                        (df_metric["model"] == model) & (df_metric["condition"] == cond)
                    ]["summary_score"]
                    .dropna()
                    .values
                )
                if len(vals) > 0:
                    groups.append(vals)
                    labels.append(str(cond))          # condition only (no model)
                    colors.append(_model_color(model))
                    if model not in models_seen_order:
                        models_seen_order.append(model)
                    all_stats.append({
                        "metric": metric,
                        "model": model,
                        "condition": cond,
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "n": int(len(vals)),
                    })

        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_path = os.path.join(out_dir, "summary_stats.csv")
            stats_df.to_csv(stats_path, index=False)

        if not groups:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        bp = ax.boxplot(
            groups,
            patch_artist=True,
            medianprops=dict(linewidth=1.5),
            boxprops=dict(linewidth=1.0),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker="o", markersize=4, alpha=0.7),
        )

        # Apply colors per model
        for box, c in zip(bp["boxes"], colors):
            box.set_facecolor(c)
            box.set_alpha(0.6)

        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Summary score")
        ax.set_title(f"Distribution of summary scores for {metric}")

        # Legend for models
        legend_patches = [
            mpatches.Patch(facecolor=_model_color(m), alpha=0.6, label=m)
            for m in models_seen_order
        ]
        if legend_patches:
            ax.legend(handles=legend_patches, title="Model", loc="upper left")

        fig.tight_layout()
        fname = os.path.join(out_dir, f"distribution_{metric}.png")
        fig.savefig(fname)
        plt.close(fig)


def _choose_tick_positions(n: int, max_labels: int) -> np.ndarray:
    """Return up to `max_labels` integer tick positions from [0, n-1]."""
    if n <= max_labels or max_labels < 2:
        return np.arange(n)
    ticks = np.unique(np.round(np.linspace(0, n - 1, num=max_labels)).astype(int))
    ticks[0] = 0
    ticks[-1] = n - 1
    return ticks


def plot_heatmaps(matrix_dir: str, out_dir: str) -> None:
    """Plot and save heatmaps of similarity matrices."""
    os.makedirs(out_dir, exist_ok=True)
    matrices = load_matrices(matrix_dir)
    for entry in matrices:
        mat = entry["matrix"]
        display_mat = np.nan_to_num(mat, nan=0.0)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(display_mat, interpolation="nearest")

        ax.set_title(f"{entry['model']} – {entry['metric']} – {entry['condition']}")
        n = display_mat.shape[0]

        ticks = _choose_tick_positions(n, MAX_TICK_LABELS)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticklabels([str(t) for t in ticks])

        ax.set_xlabel("Image index")
        ax.set_ylabel("Image index")
        fig.colorbar(im, ax=ax)

        def safe(s: str) -> str:
            return (s or "unknown").replace("/", "-").replace(" ", "_")

        fname = f"heatmap_{safe(entry['model'])}_{safe(entry['metric'])}_{safe(entry['condition'])}.png"
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname))
        plt.close(fig)

        hist_dir = os.path.join(out_dir, "histograms")
        os.makedirs(hist_dir, exist_ok=True)

        vals = _flatten_upper_triangle(display_mat)
        if vals.size > 0:
            fig_h, ax_h = plt.subplots(figsize=(7, 4))
            # histogram (low opacity)
            ax_h.hist(vals, bins=40, density=True, alpha=0.35, edgecolor='black', linewidth=0.5)
            # KDE curve
            grid, dens = _kde_curve(vals)
            ax_h.plot(grid, dens, linewidth=2)

            ax_h.set_title(f"Histogram (with KDE) – {entry['metric']} – {entry['model']} – {entry['condition']}")
            ax_h.set_xlabel("Similarity")
            ax_h.set_ylabel("Density")
            fig_h.tight_layout()

            hist_name = f"histogram_{safe(entry['model'])}_{safe(entry['metric'])}_{safe(entry['condition'])}.png"
            fig_h.savefig(os.path.join(hist_dir, hist_name))
            plt.close(fig_h)

def _flatten_upper_triangle(mat: np.ndarray) -> np.ndarray:
    """Return the upper-triangular (k=1) values flattened (exclude diagonal)."""
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Matrix must be square.")
    i_upper = np.triu_indices_from(mat, k=1)
    vals = mat[i_upper]
    return vals[np.isfinite(vals)]

def _kde_curve(x: np.ndarray, gridsize: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Simple Gaussian KDE (Silverman's rule) implemented with NumPy only."""
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    std = np.std(x, ddof=1) if n > 1 else 0.0
    if std == 0.0:
        # All values identical -> spike at the value
        grid = np.linspace(x.min() - 1e-6, x.max() + 1e-6, gridsize)
        y = np.zeros_like(grid)
        y[np.argmin(np.abs(grid - x[0]))] = 1.0
        return grid, y
    h = 1.06 * std * n ** (-1 / 5)  # Silverman's bandwidth
    grid = np.linspace(x.min(), x.max(), gridsize)
    # Gaussian kernels summed over data points
    diff = (grid[:, None] - x[None, :]) / h
    y = np.exp(-0.5 * diff * diff).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))
    return grid, y



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate plots for similarity summaries and matrices"
    )
    parser.add_argument(
        "mode",
        choices=["distributions", "heatmaps"],
        help="Type of plot to generate",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="For distributions: summary CSV/Parquet; for heatmaps: matrix directory",
    )
    parser.add_argument(
        "out_dir", type=str, help="Directory to save the plots"
    )
    args = parser.parse_args()
    if args.mode == "distributions":
        plot_distributions(args.input_path, args.out_dir)
    else:
        plot_heatmaps(args.input_path, args.out_dir)