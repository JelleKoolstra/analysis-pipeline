"""
Module for transforming similarity matrices into summarised per‑image
samples.

This script converts each similarity matrix into a tidy table where
each row represents a single image (or prompt) and summarises its
similarities to all other images in the same matrix.  The summary
statistic used is the mean of the off‑diagonal similarities (i.e.,
ignoring self–similarities which have been set to NaN by
``load_validate``).  Using per‑image summaries reduces the dependence
between the numerous pairwise similarity values and provides a
manageable, interpretable sample for non‑parametric statistical
testing.

The function ``transform`` can be imported and called from another
module or used via the command line.  It expects to receive a list of
matrix entries (as returned by ``load_matrices``) and writes the
resulting summary table to disk.  Columns in the output include
``metric``, ``model``, ``condition``, ``image_id`` and
``summary_score``.
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from load_validate import load_matrices


def _row_mean_off_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Compute the mean similarity for each row, ignoring NaNs and diagonal.

    Parameters
    ----------
    matrix : np.ndarray
        A square matrix with NaN on the diagonal.

    Returns
    -------
    np.ndarray
        A one‑dimensional array of length ``n`` where ``n`` is the
        number of rows/columns of the input.  Each element is the mean
        of the non‑NaN entries in that row.
    """
    # Use nanmean to ignore NaNs (the diagonal) automatically.  If a
    # row contains only NaNs (which should not happen for similarity
    # matrices), the result will be NaN.
    return np.nanmean(matrix, axis=1)


def transform(matrix_dir: str, output_path: str) -> pd.DataFrame:
    """Transform similarity matrices into a tidy summary table.

    Parameters
    ----------
    matrix_dir : str
        Directory containing the similarity matrix files.
    output_path : str
        Destination path for the output CSV or Parquet file.  The
        extension of the file name determines the format.  Supported
        formats are ``.csv`` and ``.parquet``.

    Returns
    -------
    pandas.DataFrame
        The constructed summary table.  The DataFrame has columns
        ``metric``, ``model``, ``condition``, ``image_id`` and
        ``summary_score``.

    Notes
    -----
    If the input CSV files include labels in the first row and
    column, these are parsed and the ``image_id`` field will be a
    short string derived from the image identifier.  Otherwise,
    ``image_id`` defaults to the integer index of the row in the
    original matrix.  If the matrices are aligned across models (i.e.,
    the same prompts in the same order), these identifiers (either
    strings or indices) can be used to compare across models.
    """
    matrices = load_matrices(matrix_dir)
    records = []
    for entry in matrices:
        mat = entry["matrix"]
        ids = entry.get("image_ids")
        # Compute the per‑row summary statistic
        summaries = _row_mean_off_diagonal(mat)
        # If labels are provided, use them; otherwise use the row index
        for idx, score in enumerate(summaries):
            image_id = ids[idx] if ids is not None and idx < len(ids) else idx
            records.append(
                {
                    "metric": entry["metric"],
                    "model": entry["model"],
                    "condition": entry["condition"],
                    "image_id": image_id,
                    "summary_score": float(score),
                }
            )
    df = pd.DataFrame.from_records(records)
    # Save to file based on extension
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".csv":
        df.to_csv(output_path, index=False)
    elif ext == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        raise ValueError(
            "Unsupported output format. Use a .csv or .parquet extension."
        )
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert similarity matrices into per‑image summaries"
    )
    parser.add_argument(
        "matrix_dir", type=str, help="Directory containing similarity matrix files"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Destination CSV or Parquet file for the summary table",
    )
    args = parser.parse_args()
    transform(args.matrix_dir, args.output_path)