"""
Module for loading and validating similarity matrices.

This module provides a function ``load_matrices`` which reads all files
in a given directory and attempts to interpret each as a similarity
matrix.  Filenames are assumed to follow a simple convention of
``{model}_{metric}_{condition}.{ext}`` where ``model`` identifies
the version of the generative model (e.g., ``SD1.5`` or ``SDXL``),
``metric`` identifies the similarity embedding (e.g., ``CLIP``,
``DINO`` or ``DiffSim``) and ``condition`` describes the prompt
setting (e.g. ``low``, ``medium`` or ``high`` complexity, or
``neutral``/``styles``/``realworld``).

Matrices are validated to ensure they are square and symmetric.
Diagonal entries are removed (set to NaN) because self–similarities do
not carry meaningful information for downstream analysis.  The caller
is expected to drop these entries or compute summaries using the
off‑diagonal entries only.

Supported file formats are ``.npy`` (NumPy arrays), ``.csv`` and
``.txt`` (CSV files with no header).  Additional formats can easily
be added by extending ``_load_single_matrix``.

The returned data structure is a list of dictionaries containing the
loaded matrix and its associated metadata.  The metadata fields are:
``model``, ``metric`` and ``condition``.  The matrix itself is
represented as a two‑dimensional NumPy array.
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


def _parse_filename(filename: str) -> Dict[str, Optional[str]]:
    """Parse a filename into metadata components.

    This helper supports two naming conventions:

    1. **Similarity prefix format**: filenames beginning with
       ``similarity_`` are assumed to follow the pattern
       ``similarity_<metric>_<model>_<condition...>.<ext>``.  In this case
       ``metric`` is taken from the second component, ``model`` from the
       third component, and all subsequent components are joined with
       underscores to form the ``condition``.  For example,
       ``similarity_clip_1.5_compl_2.csv`` yields
       ``metric='CLIP'``, ``model='SD1.5'``, ``condition='compl_2'``.

    2. **Simple format**: if the filename does not start with
       ``similarity_`` it is assumed to follow the pattern
       ``<model>_<metric>_<condition>.<ext>`` (as documented above).
       Missing parts are padded with ``None``.

    Parameters
    ----------
    filename : str
        The name of the file (with or without extension).

    Returns
    -------
    dict
        A dictionary with keys ``model``, ``metric`` and ``condition``.
    """
    name, _ = os.path.splitext(filename)
    # Handle similarity prefix naming
    if name.lower().startswith("similarity_"):
        parts = name.split("_")
        # Drop the 'similarity' prefix
        parts = parts[1:]
        if len(parts) >= 2:
            raw_metric = parts[0]
            raw_model = parts[1]
            condition_parts = parts[2:]
            condition = "_".join(condition_parts) if condition_parts else None
            # Normalise metric: upper case special handling for DiffSim
            metric_map = {
                "clip": "CLIP",
                "dino": "DINO",
                "diffsim": "DiffSim",
            }
            metric = metric_map.get(raw_metric.lower(), raw_metric.upper())
            # Normalise model: prefix with 'SD' unless already present
            if raw_model.lower() in {"xl", "sdxl"}:
                model = "SDXL"
            else:
                # Ensure it starts with 'SD'
                model = raw_model if raw_model.upper().startswith("SD") else f"SD{raw_model}"
            return {"model": model, "metric": metric, "condition": condition}
        # If parts are insufficient fall through to simple parser
    # Fallback: simple underscore parser
    parts = name.split("_")
    # Fill missing parts with None
    while len(parts) < 3:
        parts.append(None)
    model, metric, condition = parts[:3]
    return {"model": model, "metric": metric, "condition": condition}


def _load_single_matrix(path: str) -> tuple[np.ndarray, Optional[List[str]]]:
    """Load a similarity matrix (and labels) from disk.

    This helper attempts to load the file at ``path`` based on its
    extension.  Supported formats are ``.npy`` for raw NumPy arrays
    and ``.csv``/``.txt`` for delimited numeric data.  CSV and TXT
    files are read using Latin‑1 encoding to gracefully handle any
    extended characters in the prompt text (UTF‑8 decoding errors have
    been observed with these files).  When loading such files, if the
    first row and column contain non‑numeric identifiers (e.g.,
    ``12345 - prompt``), these are extracted as labels and removed
    from the numeric matrix.  The returned ``image_ids`` list will
    contain a string identifier for each image (row/column) after
    stripping text following the first dash and truncating to 5
    characters.  If a file contains no such labels, ``image_ids``
    will be ``None``.

    Parameters
    ----------
    path : str
        Absolute or relative path to the matrix file.

    Returns
    -------
    tuple
        A tuple ``(matrix, image_ids)`` where ``matrix`` is a
        two‑dimensional float array of shape (n, n) and ``image_ids``
        is either ``None`` or a list of ``n`` string identifiers for
        the rows/columns.

    Raises
    ------
    ValueError
        If the file format is not supported or if the resulting
        numeric matrix is not square.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    # Handle NumPy arrays directly
    if ext == ".npy":
        matrix = np.load(path)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix loaded from {path} is not square")
        return matrix.astype(float), None
    # Handle CSV/TXT files with potential labels
    if ext in {".csv", ".txt"}:
        # Read entire file as objects to preserve strings.  Use Latin‑1
        # encoding since some prompt files contain extended characters
        # that are not valid in UTF‑8.  If the file is UTF‑8, Latin‑1
        # decoding will still succeed because it maps bytes directly to
        # Unicode code points.
        df = pd.read_csv(path, header=None, dtype=object, encoding="latin1")
        # Try to interpret the second cell (first data value) as numeric
        # to determine if labels are present.  Using iloc[0,1] assumes
        # there is at least one row and two columns.
        image_ids: Optional[List[str]] = None
        try:
            # Attempt to convert entire DataFrame to numeric
            numeric_df = df.astype(float)
            matrix = numeric_df.values
            # No labels detected
            image_ids = None
        except Exception:
            # Assume first row and column contain labels
            # Extract labels from the first row (excluding the top-left corner)
            # Format: ``<id> - <prompt>``; we take the part before the dash
            # and truncate to 5 characters
            labels = df.iloc[0, 1:].astype(str).tolist()
            processed_labels = []
            for lab in labels:
                # split on the first dash
                parts = str(lab).split("-", 1)
                id_part = parts[0].strip()
                # take first 5 characters to avoid long IDs
                processed_labels.append(id_part[:5])
            image_ids = processed_labels
            # Drop the first row and column to isolate numeric matrix
            numeric_df = df.drop(index=0).drop(columns=0)
            try:
                matrix = numeric_df.astype(float).values
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse numeric values from {path}: {exc}"
                )
        # Ensure square matrix
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix loaded from {path} is not square after processing labels")
        return matrix.astype(float), image_ids
    # Unsupported extension
    raise ValueError(f"Unsupported file extension: {ext}")


def load_matrices(directory: str) -> List[Dict[str, object]]:
    """Load all similarity matrices from a directory tree.

    This function traverses ``directory`` recursively, loading all files
    with supported extensions (``.npy``, ``.csv``, ``.txt``).  It
    supports two filename conventions as described in
    ``_parse_filename``.  Additionally, when matrices are stored
    beneath nested directories (e.g. ``CLIP/similarity_clip_1.5_compl_1.csv``)
    the immediate parent directory is used as a hint for the ``metric``.

    The diagonal of each matrix is replaced with NaN.  Symmetry is
    checked and a warning is printed if a file appears asymmetric.

    Parameters
    ----------
    directory : str
        Root directory containing similarity matrices in arbitrary
        subdirectories.

    Returns
    -------
    list of dict
        A list of dictionaries.  Each dictionary contains the keys
        ``model``, ``metric``, ``condition``, ``matrix`` and
        optionally ``image_ids``.  The ``matrix`` value is a
        two‑dimensional NumPy array and ``image_ids`` is a list of
        strings when labels are present in the source CSV, otherwise
        ``None``.
    """
    matrices: List[Dict[str, object]] = []
    # Walk through directory tree
    for root, dirs, files in os.walk(directory):
        for fname in sorted(files):
            # Skip hidden files
            if fname.startswith('.'):
                continue
            path = os.path.join(root, fname)
            # Determine meta from filename
            meta = _parse_filename(fname)
            # If metric is missing or ambiguous, use parent directory name
            if not meta.get("metric") or meta["metric"] in {None, ""}:
                parent = os.path.basename(root)
                # Normalise metric using map
                m = parent.lower()
                metric_map = {
                    "clip": "CLIP",
                    "dino": "DINO",
                    "diffsim": "DiffSim",
                }
                meta["metric"] = metric_map.get(m, parent)
            # Normalise model (prefix SD for numeric models)
            raw_model = meta["model"]
            if raw_model:
                if raw_model.lower() in {"xl", "sdxl"}:
                    norm_model = "SDXL"
                else:
                    norm_model = (
                        raw_model if raw_model.upper().startswith("SD") else f"SD{raw_model}"
                    )
                meta["model"] = norm_model
            try:
                mat, ids = _load_single_matrix(path)
            except Exception as exc:
                raise RuntimeError(f"Failed to load matrix from {path}: {exc}")
            # Validate symmetry
            if not np.allclose(mat, mat.T, equal_nan=True):
                print(f"Warning: matrix '{fname}' is not symmetric")
            # Replace diagonal with NaN
            np.fill_diagonal(mat, np.nan)
            entry = {
                "model": meta.get("model"),
                "metric": meta.get("metric"),
                "condition": meta.get("condition"),
                "matrix": mat,
                "image_ids": ids,
            }
            matrices.append(entry)
    return matrices


__all__ = ["load_matrices"]