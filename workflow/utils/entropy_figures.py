"""entropy_figures.py — Shared plotting utilities for entropy-stratification figures.

Used by:
  workflow/scripts/plot_entropy_analysis.py  (replaces notebook 014)
  workflow/scripts/plot_entropy_main_figures.py

Statistical testing utilities live in utils/stats.py.

Exports
-------
FEATURE_LABEL_DICT          human-readable feature names
select_top_after_clustering cluster-based de-duplication for continuous features
compute_cat_freq            categorical feature frequencies per group
plot_stat_test_figure       three-panel publication figure (VDA / odds-ratio / freq)
plot_entropy_scatter        H_pred vs I_bald scatter coloured by group
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")


# ── Feature label dictionary (consolidated from notebooks 000 and 014) ────────

FEATURE_LABEL_DICT: dict[str, str] = {
    # ── Sequence-level ────────────────────────────────────────────────────────
    "RNA_size_feelnc": "Transcript length",
    "GC_content_lncDC": "GC content",
    # ── ORF features ─────────────────────────────────────────────────────────
    "ORF_l_cpat": "ORF length (CPAT)",
    "ORF_T0_length_lncDC": "ORF length (type 0)",
    "ORF_T1_length_lncDC": "ORF length (type 1)",
    "ORF_T2_length_lncDC": "ORF length (type 2)",
    "ORF_T0_coverage_lncDC": "ORF coverage (type 0)",
    "ORF_T1_coverage_lncDC": "ORF coverage (type 1)",
    "ORF_coverage_l_cpat": "ORF coverage (CPAT)",
    "orf_coverage_plncpro": "ORF coverage (PlncPro)",
    "ORF_T0_MW_lncDC": "ORF Mol. Weight (type 0)",
    "ORF_T1_MW_lncDC": "ORF Mol. Weight (type 1)",
    "ORF_T2_MW_lncDC": "ORF Mol. Weight (type 2)",
    "ORF.Max.Len_lncfinder": "Max. ORF length",
    "ORF.Max.Cov_lncfinder": "Max. ORF coverage",
    # ── Sequence composition ──────────────────────────────────────────────────
    "RCB_T0_lncDC": "ORF relative codon bias (type 0)",
    "RCB_T1_lncDC": "ORF relative codon bias (type 1)",
    "Fickett_l_cpat": "Fickett score (CPAT)",
    "Fickett_score_lncDC": "Fickett score (lncDC)",
    "Hexamer_l_cpat": "Hexamer score (CPAT)",
    "Hexamer_score_ORF_T0_lncDC": "Hexamer score (ORF type 0)",
    "Hexamer_score_ORF_T1_lncDC": "Hexamer score (ORF type 1)",
    "Hexamer_score_ORF_T2_lncDC": "Hexamer score (ORF type 2)",
    "kmerScore_12mer_feelnc": "12-mer k-mer score",
    "SS_score_k5_lncDC": "Secondary structure score (k=5)",
    # ── LncFinder features ────────────────────────────────────────────────────
    "Signal.Min_lncfinder": "Minimum frame signal",
    "Signal.Q1_lncfinder": "Q1 of frame signal",
    "Signal.Q2_lncfinder": "Q2 of frame signal",
    "Signal.Peak_lncfinder": "Peak frame signal",
    "Signal.Max_lncfinder": "Maximum frame signal",
    "SNR_lncfinder": "Signal-to-noise ratio",
    "MFE_lncfinder": "Minimum free energy",
    "SS.pct.dist_lncfinder": "Log. distance to coding SS",
    "SS.lnc.dist_lncfinder": "Log. distance to lncRNA SS",
    "SS.Dist.Ratio_lncfinder": "SS distance ratio",
    "Seq.pct.Dist_lncfinder": "Log. distance to coding seqs.",
    "Seq.lnc.Dist_lncfinder": "Log. distance to lncRNA seqs.",
    "Seq.Dist.Ratio_lncfinder": "lncRNA/mRNA distance ratio",
    "Dot_pct.dist_lncfinder": "Distance to coding SS (dot format)",
    "Dot_Dist.Ratio_lncfinder": "SS distance ratio (dot format)",
    # ── BLAST features ────────────────────────────────────────────────────────
    "all_Frame_Entropy_plncpro": "BLAST hit frame entropy",
    "all_Bitscore_plncpro": "Sum of BLAST hit bitscores",
    "all_HitScore_plncpro": "Sum of BLAST significance scores",
    # ── TE features (continuous) ──────────────────────────────────────────────
    "te_gaps_max": "Longest gap between TEs",
    "te_sum_hit_length": "Total length of TEs",
    "te_sum_num_fragments": "Total number of TE hit fragments",
    "te_max_hit_reference_coverage": "Max TE hit reference coverage",
    "te_count": "TE count",
    "te_ltr_count": "LTR count",
    "te_max_hit_length": "Max length of a TE",
    "te_count_per_kb": "TE count per kb",
    "te_max_divergence": "Max TE divergence",
    "global_gaps_max": "Max gap between rep. elements",
    "global_rm_total_length": "Total length of rep. elements",
    # ── Non-B DNA features (continuous) ──────────────────────────────────────
    "total_nonb_count": "Total non-B DNA motifs",
    "n_motif_types": "Non-B motif types present",
    "z_hit_count": "Z-DNA hits",
    "gq_hit_count": "G-quadruplex hits",
    "mr_unique_length": "Mirror repeats total length",
    "tri_gaps_max": "Max gap between triplex motifs",
    "ir_gaps_mean_pct": "Mean gap cvg. between IRs",
    "str_mean_length_pct": "Mean coverage of STRs",
    "ir_max_length_pct": "Max coverage of an IR",
    "str_max_length_pct": "Max coverage of a STR",
    # ── TE features (categorical) ─────────────────────────────────────────────
    "te_has_sine": "SINE",
    "te_has_dna": "DNA transposon",
    "te_has_ltr": "LTR",
    "te_has_line": "LINE",
    "te_has_srprna": "srpRNA",
    "pseudo_has_snrna": "snRNA pseudogene",
    "lctr_has_low_complexity": "Low Complexity Region",
    "lctr_has_simple_repeat": "Simple Repeat",
    "lctr_has_satellite": "Satellite",
    # ── Non-B DNA features (categorical) ─────────────────────────────────────
    "gq_present": "G-Quadruplex",
    "z_present": "Z-DNA",
    "tri_present": "Triplex DNA",
    "apr_present": "A-Phased Repeat",
    "ir_present": "Inverted Repeat",
    "dr_present": "Direct Repeat",
    "mr_present": "Mirror Repeat",
    "str_present": "Short Tandem Repeat",
}

# Overwrite categorical feature names for when they are outside of the categorical plot
FEATURE_LABEL_DICT_SHAP = {"te_has_ltr": "LTR presence"}

# ── Feature selection helpers ─────────────────────────────────────────────────


def select_top_after_clustering(
    mannu_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cluster_col: str = "cluster_0.25",
    n: int = 10,
    pinned_feature: str = "RNA_size_feelnc",
) -> pd.DataFrame:
    """Cluster-based de-duplication: keep one representative per cluster (highest |VDA|).

    ``pinned_feature`` is guaranteed to survive by removing all other members of
    its cluster before the per-cluster head selection.
    """
    merged = mannu_df.sort_values("abs_vda", ascending=False).merge(
        cluster_df[[cluster_col]], left_index=True, right_index=True, how="left"
    )
    if pinned_feature in merged.index:
        pinned_cluster = merged.loc[pinned_feature, cluster_col]
        drop_idx = merged[
            (merged[cluster_col] == pinned_cluster) & (merged.index != pinned_feature)
        ].index
        merged = merged.drop(index=drop_idx)
    return (
        merged.groupby(cluster_col, sort=False)
        .head(1)
        .sort_values("abs_vda", ascending=False)
        .head(n)
    )


def compute_cat_freq(
    grp1: pd.Index,
    grp2: pd.Index,
    categorical: pd.DataFrame,
) -> pd.DataFrame:
    """Percentage of transcripts in each group with each categorical feature.

    Returns a DataFrame indexed by feature name with columns ``group1`` and
    ``group2``.  Suitable for saving to TSV and passing to
    ``plot_stat_test_figure`` as ``cat_freq_df``.
    """
    g1 = categorical.loc[grp1.intersection(categorical.index)]
    g2 = categorical.loc[grp2.intersection(categorical.index)]
    return pd.DataFrame({"group1": g1.mean() * 100, "group2": g2.mean() * 100})


# ── Three-panel statistical test figure ──────────────────────────────────────

_NONSIG_COLOR = "#d3d3d3"  # light gray for non-significant features

_PANEL_RC: dict = {
    "figure.dpi": 300,
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
    "axes.titlesize": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "font.size": 6,
    "legend.fontsize": 6,
    "lines.linewidth": 0.5,
    "axes.linewidth": 0.5,
    "patch.linewidth": 0.5,
    "axes.edgecolor": "black",
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
}


def effect_size_direction_annotation(ax, annotation, grp1_label, grp2_label):
    # Use axes-fraction coordinates so labels always align to the axes edges
    # regardless of the data x-limits.
    t = ax.transAxes
    if annotation:
        right_label, left_label = annotation
        ax.text(
            1.0, 1.03, right_label, ha="right", va="bottom", transform=t, fontsize=6
        )
        ax.text(0.0, 1.03, left_label, ha="left", va="bottom", transform=t, fontsize=6)
    else:
        ax.text(
            1.0,
            1.03,
            f"{grp2_label} \u2192",
            ha="right",
            va="bottom",
            transform=t,
            fontsize=6,
        )
        ax.text(
            0.0,
            1.03,
            f"\u2190 {grp1_label}",
            ha="left",
            va="bottom",
            transform=t,
            fontsize=6,
        )
    ax.text(
        0.5, 1.03, "No association", ha="center", va="bottom", transform=t, fontsize=6
    )


def plot_stat_test_figure(
    mannu_df: pd.DataFrame,
    chi2_df: pd.DataFrame,
    grp1_label: str,
    grp2_label: str,
    cat_freq_df: "pd.DataFrame | None" = None,
    cluster_df: "pd.DataFrame | None" = None,
    cluster_col: str = "cluster_0.25",
    title: str = "",
    save_path: "Path | str | None" = None,
    xlim_chi2: tuple = (-1.3, 1.3),
    annotation: "tuple | None" = None,
    n_top: int = 10,
    # backward-compat kwargs (used when cat_freq_df is not provided)
    grp1: "pd.Index | None" = None,
    grp2: "pd.Index | None" = None,
    categorical: "pd.DataFrame | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three-panel publication figure for entropy-stratification statistical tests.

    Panel 1: Continuous features — VDA − 0.5 bar chart (cluster-deduplicated top N)
    Panel 2: Categorical features — log10 odds ratio coloured by Cramér's V
    Panel 3: % transcripts per group with each categorical feature

    Parameters
    ----------
    mannu_df, chi2_df      : outputs of ``utils.stats.run_stat_tests``
    grp1_label, grp2_label : display names
    cat_freq_df            : pre-computed frequencies (index=features,
                             columns ``group1``/``group2``, values = %).
                             When None, computed from ``categorical + grp1 + grp2``.
    cluster_df             : cluster assignment DataFrame (from clustering pipeline)
    cluster_col            : column in cluster_df to use (default: cluster_0.25)
    title                  : optional title above panel 1
    save_path              : path stem (without suffix); saves .pdf + .png if given
    xlim_chi2              : x-axis limits for the odds-ratio panel
    annotation             : (right_label, left_label) for VDA axis direction arrows
    n_top                  : number of top features to show (default: 10)
    grp1, grp2, categorical: legacy — used to compute cat_freq_df when not provided

    Returns
    -------
    top_cont : de-duplicated continuous feature table
    top_cat  : top categorical feature table
    """
    sns.set_theme(style="whitegrid", font="Arial", rc=_PANEL_RC)

    top_cont = select_top_after_clustering(
        mannu_df, cluster_df, cluster_col=cluster_col, n=n_top
    ).copy()
    top_cont["vda_to_plot"] = top_cont["vda"] - 0.5
    top_cont["label"] = top_cont.index.map(lambda x: FEATURE_LABEL_DICT.get(x, x))

    top_cat = chi2_df.sort_values("cramers_v", ascending=False).head(n_top).copy()
    top_cat["label"] = top_cat.index.map(lambda x: FEATURE_LABEL_DICT.get(x, x))
    top_cat["log10_or"] = np.log10(top_cat["odds_ratio"].replace(0, np.nan))

    if cat_freq_df is None:
        cat_freq_df = compute_cat_freq(grp1, grp2, categorical)
    freq_df = cat_freq_df.reindex(top_cat.index)[["group1", "group2"]].rename(
        columns={"group1": grp1_label, "group2": grp2_label}
    )
    freq_df["label"] = freq_df.index.map(lambda x: FEATURE_LABEL_DICT.get(x, x))

    fig = plt.figure(figsize=(8.5 / 2.54, 15 / 2.54), dpi=300)
    gs = gridspec.GridSpec(
        3, 2, figure=fig, width_ratios=[1, 0.05], wspace=0.05, hspace=0.5
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    cax = fig.add_subplot(gs[1, 1])

    # Panel 1 — continuous VDA bars
    sns.barplot(
        x=top_cont["vda_to_plot"],
        y=top_cont.index[::-1],
        ax=ax0,
        color=sns.color_palette("crest")[-2],
    )
    for patch, (_, row) in zip(ax0.patches, top_cont[::-1].iterrows()):
        if not row.get("significant", True):
            patch.set_facecolor(_NONSIG_COLOR)
            patch.set_edgecolor("none")
    ax0.set_yticks(np.arange(len(top_cont))[::-1])
    ax0.set_yticklabels(top_cont["label"][::-1], fontsize=6)
    ax0.set_ylabel("", fontsize=6)
    ax0.set_xlabel("Effect size (VDA) \u2212 0.5", fontsize=6)
    ax0.set_xlim(-0.55, 0.55)
    ax0.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax0.tick_params(axis="x", which="major", bottom=True, length=3, width=0.5)

    if title:
        ax0.set_title(title, fontsize=7, pad=16)

    effect_size_direction_annotation(ax0, annotation, grp1_label, grp2_label)

    # Panel 2 — categorical log10 odds ratio coloured by Cramér's V
    norm_cat = plt.Normalize(0, 0.6)
    cmap_cat = sns.color_palette("crest", as_cmap=True)
    sns.barplot(
        x=top_cat["log10_or"],
        y=top_cat.index[::-1],
        hue=top_cat["cramers_v"],
        palette=cmap_cat,
        hue_norm=norm_cat,
        ax=ax1,
    )
    ax1.set_yticks(np.arange(len(top_cat))[::-1])
    ax1.set_yticklabels(top_cat["label"][::-1], fontsize=6)
    ax1.set_ylabel("", fontsize=6)
    ax1.set_xlabel("log10 odds ratio", fontsize=6)
    ax1.set_xlim(*xlim_chi2)
    ax1.tick_params(axis="x", which="major", bottom=True, length=3, width=0.5)
    ax1.get_legend().remove()
    for patch, (_, row) in zip(ax1.patches, top_cat[::-1].iterrows()):
        if not row.get("significant", True):
            patch.set_facecolor(_NONSIG_COLOR)
            patch.set_edgecolor("none")
    sm = plt.cm.ScalarMappable(cmap=cmap_cat, norm=norm_cat)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label("Cramér's V", fontsize=6)
    cbar.set_ticks([0, 0.2, 0.4, 0.6])
    cbar.ax.tick_params(labelsize=6)

    effect_size_direction_annotation(ax1, annotation, grp1_label, grp2_label)

    # Panel 3 — categorical feature frequencies per group
    freq_plot = freq_df.drop(columns=["label"])
    x_pos = np.arange(len(freq_plot))
    width = 0.35
    ax2.barh(
        x_pos - width / 2,
        freq_plot[grp1_label],
        width,
        label=grp1_label,
        color="#9467bd",
    )
    ax2.barh(
        x_pos + width / 2,
        freq_plot[grp2_label],
        width,
        label=grp2_label,
        color="#d95f02",
    )
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(freq_df["label"], fontsize=6)
    ax2.invert_yaxis()
    ax2.set_xlabel("Transcripts with feature (%)", fontsize=6)
    ax2.set_ylabel("", fontsize=6)
    ax2.grid(False)
    ax2.tick_params(axis="x", which="major", bottom=True, length=3, width=0.5)
    ax2.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=2,
        frameon=False,
        fontsize=6,
    )

    fig.subplots_adjust(left=0.35, right=0.88, top=0.93, bottom=0.07, hspace=0.4)

    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(
            save_path.with_suffix(".pdf"), dpi=300, format="pdf", bbox_inches="tight"
        )
        plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"  Saved \u2192 {save_path.stem}.[pdf/png]")
    plt.close()

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams["pdf.fonttype"] = 42

    return top_cont, top_cat


# ── Entropy scatter ───────────────────────────────────────────────────────────


def plot_entropy_scatter(
    entropy_df: pd.DataFrame,
    group_col: str,
    color_map: dict,
    title: str = "",
    save_path: "Path | str | None" = None,
) -> None:
    """H_pred vs I_bald scatter coloured by ``group_col``."""
    plt.rcParams["mathtext.default"] = "regular"
    plt.rcParams["mathtext.fontset"] = "stix"

    fig_w, fig_h = 6 / 2.54, 5.5 / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    for grp in entropy_df[group_col].unique():
        mask = entropy_df[group_col] == grp
        ax.scatter(
            entropy_df.loc[mask, "H_pred"],
            entropy_df.loc[mask, "I_bald"],
            c=color_map.get(grp, "#95a5a6"),
            s=0.3,
            alpha=0.5,
            linewidths=0,
            label=grp,
            rasterized=True,
        )
    ax.set_xlabel(r"Uncertainty ($\mathit{H}_{\mathit{pred}}$)", fontsize=7)
    ax.set_ylabel(r"Model Disagreement ($\mathit{I}_{\mathit{bald}}$)", fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 0.82)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ax.tick_params(width=0.5, length=2, labelsize=5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.legend(markerscale=6, frameon=False, fontsize=6, scatterpoints=1)
    handles, labels = ax.get_legend_handles_labels()
    labels = [l.replace("_", ", ") for l in labels]
    for handle in handles:
        handle.set_alpha(0.7)
    ax.legend(
        handles, labels, markerscale=6, frameon=False, fontsize=6, scatterpoints=1
    )
    if title:
        ax.set_title(title, fontsize=7)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        plt.savefig(
            save_path.with_suffix(".pdf"), dpi=300, format="pdf", bbox_inches="tight"
        )
        plt.savefig(save_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close()
