"""
Entry point for running the similarity analysis pipeline.

This script ties together the matrix loading and validation,
transformation into per‑image summaries, statistical testing, and
visualisations.  Given a directory of similarity matrices, it
produces:

* A CSV file of summarised scores (``summary_scores.csv``).
* A CSV file of statistical test results (``stats_results.csv``).
* A set of distribution plots for each metric (``distribution_<metric>.png``).
* Heatmaps of the original similarity matrices.

The output files are written to a specified result directory.  If the
directory does not exist it will be created.  Use ``python run_all.py
--help`` for usage details.
"""

from __future__ import annotations

import os
import argparse

from transform_samples import transform
from run_stats import run_statistics
from visualization import plot_distributions, plot_heatmaps


def main(matrix_dir: str, out_dir: str) -> None:
    """Run the full analysis pipeline.

    Parameters
    ----------
    matrix_dir : str
        Directory containing similarity matrix files.
    out_dir : str
        Directory where all outputs will be written.  If it does not
        exist, it will be created.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary_scores.csv")
    results_path = os.path.join(out_dir, "stats_results.csv")
    # Step 1: transform matrices into summary samples
    transform(matrix_dir, summary_path)
    # Step 2: run statistical tests
    run_statistics(summary_path, results_path)
    # Step 3: generate distribution plots
    plot_distributions(summary_path, out_dir)
    # Step 4: generate heatmaps for the matrices
    plot_heatmaps(matrix_dir, out_dir)
    print(f"Analysis complete. Results saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the full similarity analysis: summarise, test and visualise"
        )
    )
    parser.add_argument(
        "matrix_dir",
        type=str,
        help="Directory containing similarity matrices to analyse",
    )
    parser.add_argument(
        "out_dir", type=str, help="Directory in which to save outputs"
    )
    args = parser.parse_args()
    main(args.matrix_dir, args.out_dir)