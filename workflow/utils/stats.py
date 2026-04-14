"""utils/stats.py — Statistical testing utilities for feature analysis.

Exports
-------
vargha_delaney_A          Effect size for Mann-Whitney U
perform_mann_whitney_tests  Continuous-feature testing
perform_chi2_tests        Categorical-feature testing
apply_fdr_correction      Shared multiple-testing correction helper
run_stat_tests            Convenience wrapper for the combined workflow
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from utils.features import remove_constant_features


def vargha_delaney_A(u_stat: float, n_grp1: int, n_grp2: int) -> float:
    """Return the Vargha-Delaney A effect size from a Mann-Whitney U statistic."""
    if n_grp1 == 0 or n_grp2 == 0:
        raise ValueError("Group sizes must be non-zero to compute Vargha-Delaney A")
    return u_stat / (n_grp1 * n_grp2)


def perform_mann_whitney_tests(
    grp1: pd.Index,
    grp2: pd.Index,
    scalar: pd.DataFrame,
) -> pd.DataFrame:
    """Run Mann-Whitney U tests across continuous features."""
    idx = grp1.union(grp2)
    sc = remove_constant_features(scalar.reindex(idx).fillna(0))
    d1 = sc.loc[grp1.intersection(sc.index)]
    d2 = sc.loc[grp2.intersection(sc.index)]

    rows_cont: list[dict[str, float | str]] = []
    for col in sc.columns:
        u_stat, p_value = mannwhitneyu(
            d2[col].values,
            d1[col].values,
            alternative="two-sided",
        )
        vda = vargha_delaney_A(u_stat, len(d1), len(d2))
        rows_cont.append(
            {
                "feature": col,
                "statistic": u_stat,
                "p_value": p_value,
                "vda": vda,
                "abs_vda": abs(vda - 0.5),
                "test": "Mann-Whitney U",
            }
        )
    return pd.DataFrame(rows_cont).set_index("feature")


def perform_chi2_tests(
    grp1: pd.Index,
    grp2: pd.Index,
    categorical: pd.DataFrame,
) -> pd.DataFrame:
    """Run chi-squared tests across categorical features."""
    idx = grp1.union(grp2)
    ct = remove_constant_features(categorical.reindex(idx).fillna(0))
    c1 = ct.loc[grp1.intersection(ct.index)]
    c2 = ct.loc[grp2.intersection(ct.index)]

    rows_cat: list[dict[str, float | str]] = []
    for col in ct.columns:
        contingency = pd.crosstab(
            pd.concat([c1[col], c2[col]]),
            pd.concat(
                [
                    pd.Series(0, index=c1.index, name="grp"),
                    pd.Series(1, index=c2.index, name="grp"),
                ]
            ),
        )
        if contingency.shape[0] < 2:
            continue

        chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)
        n_total = len(c1) + len(c2)
        cramers_v = (
            np.sqrt(chi2_stat / (n_total * (min(contingency.shape) - 1)))
            if min(contingency.shape) > 1
            else 0.0
        )

        if contingency.shape == (2, 2):
            a = contingency.iloc[0, 0] + 0.5
            b = contingency.iloc[0, 1] + 0.5
            c = contingency.iloc[1, 0] + 0.5
            d = contingency.iloc[1, 1] + 0.5
            odds_ratio = (a * d) / (b * c)
        else:
            odds_ratio = np.nan

        rows_cat.append(
            {
                "feature": col,
                "statistic": chi2_stat,
                "p_value": chi2_p,
                "cramers_v": cramers_v,
                "odds_ratio": odds_ratio,
                "test": "Chi-squared",
            }
        )
    return pd.DataFrame(rows_cat).set_index("feature")


def apply_fdr_correction(
    result_frames: list[pd.DataFrame],
    fdr_method: str = "fdr_bh",
    fdr_alpha: float = 0.01,
    adjusted_column: str = "adj_p",
) -> list[pd.DataFrame]:
    """Apply a joint FDR correction across multiple result tables."""
    non_empty_frames = [frame.copy() for frame in result_frames if not frame.empty]
    if not non_empty_frames:
        return [frame.copy() for frame in result_frames]

    combined_p = pd.concat([frame[["p_value"]] for frame in non_empty_frames])
    adjusted = multipletests(
        combined_p["p_value"],
        method=fdr_method,
        alpha=fdr_alpha,
    )[1]

    offset = 0
    adjusted_frames: list[pd.DataFrame] = []
    for frame in result_frames:
        updated = frame.copy()
        if updated.empty:
            adjusted_frames.append(updated)
            continue
        end = offset + len(updated)
        updated[adjusted_column] = adjusted[offset:end]
        updated["significant"] = updated[adjusted_column] < fdr_alpha
        adjusted_frames.append(updated)
        offset = end

    return adjusted_frames


def run_stat_tests(
    grp1: pd.Index,
    grp2: pd.Index,
    scalar: pd.DataFrame,
    categorical: pd.DataFrame,
    fdr_method: str = "fdr_bh",
    fdr_alpha: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mann-Whitney U (continuous) + chi-squared (categorical) with joint FDR.

    Parameters
    ----------
    grp1, grp2   : transcript indices for the two groups being compared
    scalar       : continuous feature matrix (indexed by transcript)
    categorical  : binary feature matrix (indexed by transcript)
    fdr_method   : multipletests method (default: fdr_bh)
    fdr_alpha    : significance threshold after correction (default: 0.01)

    Returns
    -------
    mannu_df : continuous results — vda, abs_vda, adj_p, significant
    chi2_df  : categorical results — cramers_v, odds_ratio, adj_p, significant

    Note: MannU VDA grp2 > grp1
    """
    mannu_df = perform_mann_whitney_tests(grp1, grp2, scalar)
    chi2_df = perform_chi2_tests(grp1, grp2, categorical)
    mannu_df, chi2_df = apply_fdr_correction(
        [mannu_df, chi2_df],
        fdr_method=fdr_method,
        fdr_alpha=fdr_alpha,
        adjusted_column="adj_p",
    )

    print(f"  MWU:  {mannu_df['significant'].sum()}/{len(mannu_df)} significant")
    print(f"  Chi2: {chi2_df['significant'].sum()}/{len(chi2_df)} significant")
    return mannu_df, chi2_df


compute_pairwise_stats = run_stat_tests  # Alias
