"""
Statistical analysis of summary similarity scores.

This script defines functions for performing non‑parametric statistical
tests on the per‑image summary scores produced by
``transform_samples``.  It supports two classes of comparisons:

1. **Within‑model comparisons of prompt complexity**.  For a given
   model (e.g., ``SDXL``) and metric, the summary scores are grouped
   by the ``condition`` (such as ``low``, ``medium`` and ``high``
   complexity prompts).  A Kruskal–Wallis test is applied across
   these groups to determine if there are any differences among the
   distributions.  If significant, pairwise Mann–Whitney U tests are
   performed between each pair of conditions.  Effect sizes are
   computed and p‑values are corrected using the Benjamini–Hochberg
   False Discovery Rate procedure.

2. **Across‑model comparisons for each condition**.  For a fixed
   condition (e.g., ``low`` complexity) and metric, the summary scores
   are compared between pairs of models (e.g., ``SD1.5`` vs. ``SDXL``)
   using the Mann–Whitney U test.  In the absence of paired design
   (no shared seeds), unpaired tests are used.  Effect sizes are
   computed, and p‑values are FDR corrected.

Results from all tests are assembled into a table with fields
``test_type``, ``metric``, ``model`` or ``model_pair``, ``condition``,
``statistic``, ``effect_size``, ``p_value``, ``p_corrected`` and
``reject_null``.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def _rank_biserial_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute the rank biserial correlation effect size for MWU.

    The rank biserial correlation is defined as ``r = 1 - 2 * U / (n1 * n2)``,
    where ``U`` is the Mann–Whitney U statistic, and ``n1`` and ``n2`` are
    the sample sizes of the two groups.  This effect size ranges from
    -1 to 1, with 0 indicating no difference.

    Parameters
    ----------
    group1 : np.ndarray
        First sample.
    group2 : np.ndarray
        Second sample.

    Returns
    -------
    float
        The rank biserial correlation.
    """
    n1 = len(group1)
    n2 = len(group2)
    if n1 == 0 or n2 == 0:
        return np.nan
    U, _ = mannwhitneyu(group1, group2, alternative="two-sided")
    return 1.0 - 2.0 * U / (n1 * n2)


def _eta_squared_from_kruskal(H: float, group_sizes: List[int]) -> float:
    """Compute eta squared effect size for the Kruskal–Wallis test.

    The formula used is ``eta^2 = (H - k + 1) / (n - k)``, where ``H``
    is the Kruskal–Wallis statistic, ``k`` is the number of groups and
    ``n`` is the total sample size.  This is an estimate of the
    proportion of variance explained by the grouping factor.  The
    statistic may be negative if the sample sizes are very small,
    therefore the result is bounded below by zero.

    Parameters
    ----------
    H : float
        Kruskal–Wallis H statistic.
    group_sizes : list of int
        Sample sizes of each group.

    Returns
    -------
    float
        Eta squared effect size.
    """
    k = len(group_sizes)
    n = sum(group_sizes)
    if n <= k:
        return np.nan
    eta2 = (H - k + 1.0) / (n - k)
    return max(eta2, 0.0)


def run_statistics(summary_path: str, results_path: str) -> pd.DataFrame:
    """Perform statistical tests on summary scores and save results.

    Parameters
    ----------
    summary_path : str
        Path to the CSV or Parquet file containing per‑image summary
        statistics.  The file must have columns ``metric``, ``model``,
        ``condition`` and ``summary_score``.
    results_path : str
        Destination CSV file for the test results.

    Returns
    -------
    pandas.DataFrame
        DataFrame with rows for each test performed.  Columns include
        ``test_type``, ``metric``, ``model`` or ``model_pair``,
        ``condition``, ``statistic``, ``effect_size``, ``p_value``,
        ``p_corrected`` and ``reject_null``.
    """
    # Load summary table
    ext = os.path.splitext(summary_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(summary_path)
    elif ext == ".parquet":
        df = pd.read_parquet(summary_path)
    else:
        raise ValueError(
            "Unsupported summary file format; use .csv or .parquet"
        )
    required_cols = {"metric", "model", "condition", "summary_score"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"Summary file is missing required columns: {missing}")
    # Prepare container for results
    results = []
    for metric in sorted(df["metric"].dropna().unique()):
        df_metric = df[df["metric"] == metric]
        # Within‑model tests (complexity differences only)
        # We restrict these tests to conditions that represent prompt
        # complexity levels, identified by a prefix such as "compl".
        for model in sorted(df_metric["model"].dropna().unique()):
            df_model = df_metric[df_metric["model"] == model]
            # Collect groups only for complexity conditions
            group_map: Dict[str, np.ndarray] = {}
            group_sizes: List[int] = []
            for cond in sorted(df_model["condition"].dropna().unique()):
                # Only consider conditions that look like complexity levels
                if isinstance(cond, str) and cond.lower().startswith("compl"):
                    vals = (
                        df_model[df_model["condition"] == cond]["summary_score"]
                        .dropna()
                        .values
                    )
                    if len(vals) > 0:
                        group_map[cond] = vals
                        group_sizes.append(len(vals))
            # Proceed only if there are at least two complexity groups
            if len(group_map) >= 2:
                groups = list(group_map.values())
                H, p_kw = kruskal(*groups)
                eta2 = _eta_squared_from_kruskal(H, group_sizes)
                results.append(
                    {
                        "test_type": "Kruskal–Wallis",
                        "metric": metric,
                        "model": model,
                        "condition": "vs".join(group_map.keys()),
                        "statistic": H,
                        "effect_size": eta2,
                        "p_value": p_kw,
                    }
                )
                # Pairwise Mann–Whitney U tests among complexity groups
                cond_list = list(group_map.keys())
                for i in range(len(cond_list)):
                    for j in range(i + 1, len(cond_list)):
                        cond1 = cond_list[i]
                        cond2 = cond_list[j]
                        g1 = group_map[cond1]
                        g2 = group_map[cond2]
                        U, p_mwu = mannwhitneyu(g1, g2, alternative="two-sided")
                        r = _rank_biserial_effect_size(g1, g2)
                        results.append(
                            {
                                "test_type": "Mann–Whitney (within)",
                                "metric": metric,
                                "model": model,
                                "condition": f"{cond1}_vs_{cond2}",
                                "statistic": U,
                                "effect_size": r,
                                "p_value": p_mwu,
                            }
                        )
        # Across‑model tests for each condition
        for cond in sorted(df_metric["condition"].dropna().unique()):
            df_cond = df_metric[df_metric["condition"] == cond]
            # Collect groups for each model
            model_groups = []
            for model in sorted(df_cond["model"].dropna().unique()):
                g = (
                    df_cond[df_cond["model"] == model]["summary_score"]
                    .dropna()
                    .values
                )
                if len(g) > 0:
                    model_groups.append((model, g))
            # Pairwise comparisons between models
            for i in range(len(model_groups)):
                for j in range(i + 1, len(model_groups)):
                    model1, g1 = model_groups[i]
                    model2, g2 = model_groups[j]
                    U, p_mwu = mannwhitneyu(g1, g2, alternative="two-sided")
                    r = _rank_biserial_effect_size(g1, g2)
                    results.append(
                        {
                            "test_type": "Mann–Whitney (across)",
                            "metric": metric,
                            "model": f"{model1}_vs_{model2}",
                            "condition": cond,
                            "statistic": U,
                            "effect_size": r,
                            "p_value": p_mwu,
                        }
                    )
    # Convert results to DataFrame
    res_df = pd.DataFrame.from_records(results)
    # Apply Benjamini–Hochberg FDR correction separately per test_type
    # and per metric to avoid overly conservative corrections across
    # unrelated families of tests.
    res_df["p_corrected"] = np.nan
    res_df["reject_null"] = False
    for test_type in res_df["test_type"].unique():
        for metric in res_df["metric"].unique():
            mask = (res_df["test_type"] == test_type) & (
                res_df["metric"] == metric
            )
            pvals = res_df.loc[mask, "p_value"].values
            if len(pvals) == 0:
                continue
            # If only one p‑value, no correction needed
            if len(pvals) == 1:
                res_df.loc[mask, "p_corrected"] = pvals
                res_df.loc[mask, "reject_null"] = pvals < 0.05
            else:
                rej, pcorr, _, _ = multipletests(pvals, method="fdr_bh")
                res_df.loc[mask, "p_corrected"] = pcorr
                res_df.loc[mask, "reject_null"] = rej
    # Save results
    res_df.to_csv(results_path, index=False)
    return res_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform statistical tests on summary similarity scores"
    )
    parser.add_argument(
        "summary_path",
        type=str,
        help="CSV or Parquet file with per‑image summary scores",
    )
    parser.add_argument(
        "results_path", type=str, help="Destination CSV file for results"
    )
    args = parser.parse_args()
    run_statistics(args.summary_path, args.results_path)