configfile: "config/shap_config.yaml"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  shap.smk — Random Forest + SHAP cross-fold pipeline                     ║
# ║                                                                          ║
# ║  Two granularities of rules:                                             ║
# ║    shap_fold      — per-fold RF training + SHAP (fully parallelisable)   ║
# ║    shap_aggregate — aggregate all folds, build plots + cherry analysis   ║
# ║                                                                          ║
# ║  Wildcards                                                               ║
# ║    {expt}  experiment name   (config shap key)                           ║
# ║    {mode}  feature-set mode  (config shap → modes key)                   ║
# ║    {fold}  1-based fold number                                           ║
# ║                                                                          ║
# ║  Usage examples                                                          ║
# ║    snakemake shap_testing  -j 6 --use-conda   # test 100 transcripts     ║
# ║    snakemake shap_full_all -j 6 --use-conda   # full-feature pipeline    ║
# ║    snakemake shap_clustered_all  -j 6 --use-conda   # filtered pipeline  ║
# ║                                                                          ║
# ║  RFECV feature-selection rules live in rfecv.smk (included after this).  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Config helpers ────────────────────────────────────────────────────────────

def _shap_cfg(wc):
    """Return the top-level shap config block for the experiment."""
    return config["shap"][wc.expt]


def _mode_cfg(wc):
    """Return the per-mode sub-config for a given experiment + mode wildcard."""
    return config["shap"][wc.expt]["modes"][wc.mode]


def _n_folds(wc):
    return int(config["shap"][wc.expt]["n_folds"])


def _fold_outputs(wc, fname):
    """Expand fold output paths for the aggregate rule's input."""
    n = _n_folds(wc)
    return expand(
        "results/{expt}/features/shap_{mode}/fold{fold}/" + fname,
        expt=wc.expt, mode=wc.mode, fold=range(1, n + 1),
    )


def _effective_cluster_file(wc):
    """Return the cluster file path: per-mode override takes precedence over experiment-level."""
    mode_cfg = _mode_cfg(wc)
    if mode_cfg.get("feature_mode", "full") != "filtered":
        return ""
    return mode_cfg.get("cluster_file") or _shap_cfg(wc).get("cluster_file", "")


def _effective_cluster_threshold(wc):
    """Return cluster threshold from config, or None to trigger file-based fallback."""
    mode_cfg = _mode_cfg(wc)
    return mode_cfg.get("cluster_threshold") or _shap_cfg(wc).get("cluster_threshold")


def _cluster_file_arg(wc):
    """Return --cluster-file <path> or empty string for 'full' mode."""
    cfile = _effective_cluster_file(wc)
    return f"--cluster-file '{cfile}'" if cfile else ""


def _opt_arg(flag, path):
    """Build optional CLI args for nullable config paths."""
    return f"{flag} '{path}'" if path else ""


def _max_transcripts(wc):
    """Return max_transcripts as string; 'none' if unset / null."""
    val = _mode_cfg(wc).get("max_transcripts_per_fold")
    return str(val) if val else "none"


def _consensus_file_input(wc):
    """Return consensus JSON as list[str] when mode uses RFECV consensus, else []."""
    if _mode_cfg(wc).get("use_rfecv_consensus", False):
        return [f"results/{wc.expt}/features/rfecv/consensus_features.json"]
    return []


def _selected_features_arg(wc):
    """Return --selected-features <path> when mode uses RFECV consensus, else ''."""
    if _mode_cfg(wc).get("use_rfecv_consensus", False):
        return f"--selected-features 'results/{wc.expt}/features/rfecv/consensus_features.json'"
    return ""


def _shap_run_expts():
    """Experiments included in SHAP convenience targets."""
    return list(config.get("shap_run_experiments", config["shap"].keys()))


# ── Per-fold rule ─────────────────────────────────────────────────────────────

rule shap_fold:
    """Train one RF fold + compute SHAP values.  Runs in parallel across folds."""
    input:
        train_pc     = "results/{expt}/datasets/fold{fold}/train_pc.fa",
        train_lnc    = "results/{expt}/datasets/fold{fold}/train_lnc.fa",
        test_all     = "results/{expt}/datasets/fold{fold}/test_all.fa",
        # cluster_file / consensus_file ensure upstream rules complete before
        # shap_fold runs; each resolves to [] when not applicable for this mode
        cluster_file      = lambda wc: ([_effective_cluster_file(wc)]
                                        if _effective_cluster_file(wc) else []),
        consensus_file    = lambda wc: _consensus_file_input(wc),
        optimal_threshold = "results/{expt}/features/clustering/optimal_threshold.txt",
    output:
        rf_model  = "results/{expt}/features/shap_{mode}/fold{fold}/rf_model.joblib",
        X_test    = "results/{expt}/features/shap_{mode}/fold{fold}/X_test.csv",
        y_test    = "results/{expt}/features/shap_{mode}/fold{fold}/y_test.csv",
        y_pred    = "results/{expt}/features/shap_{mode}/fold{fold}/y_pred.csv",
        report    = "results/{expt}/features/shap_{mode}/fold{fold}/classification_report.csv",
        shap_vals = "results/{expt}/features/shap_{mode}/fold{fold}/shap_values.csv",
        base_val  = "results/{expt}/features/shap_{mode}/fold{fold}/base_val.txt",
    params:
        out_dir          = "results/{expt}/features/shap_{mode}/fold{fold}",
        feature_mode     = lambda wc: _mode_cfg(wc).get("feature_mode", "full"),
        cluster_file_arg = lambda wc: _cluster_file_arg(wc),
        cluster_thresh   = lambda wc, input: (
            _effective_cluster_threshold(wc) or open(input.optimal_threshold).read().strip()
        ),
        te_arg           = lambda wc: _opt_arg("--te-features", _shap_cfg(wc).get("te_features")),
        nbd_arg          = lambda wc: _opt_arg("--nbd-features", _shap_cfg(wc).get("nbd_features")),
        max_transcripts  = lambda wc: _max_transcripts(wc),
        background       = lambda wc: _shap_cfg(wc).get("background_sample", 500),
        random_state     = lambda wc: _shap_cfg(wc).get("random_state", 42),
        selected_features_arg = lambda wc: _selected_features_arg(wc),
        force_rerun      = lambda wc: "--force-rerun" if _shap_cfg(wc).get("force_rerun", False) else "",
    log:
        "logs/{expt}/features/shap_{mode}/fold{fold}/train_shap.log",
    threads: 1
    resources:
        mem_mb  = 40000,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/shap_train_fold.py \
            --dataset        {wildcards.expt} \
            --fold           {wildcards.fold} \
            --results-dir    results \
            --output-dir     {params.out_dir} \
            --feature-mode   {params.feature_mode} \
            {params.te_arg} \
            {params.nbd_arg} \
            --max-transcripts {params.max_transcripts} \
            --background-sample {params.background} \
            --random-state   {params.random_state} \
            --cluster-threshold {params.cluster_thresh} \
            {params.cluster_file_arg} {params.selected_features_arg} {params.force_rerun} \
        2>&1 | tee {log}
        """


# ── Aggregation rule ──────────────────────────────────────────────────────────

rule shap_aggregate:
    """Aggregate all folds: importance plots, beeswarm, cherry picks, perf stats."""
    input:
        shap_vals   = lambda wc: _fold_outputs(wc, "shap_values.csv"),
        y_preds     = lambda wc: _fold_outputs(wc, "y_pred.csv"),
        reports     = lambda wc: _fold_outputs(wc, "classification_report.csv"),
        X_tests     = lambda wc: _fold_outputs(wc, "X_test.csv"),
        base_vals   = lambda wc: _fold_outputs(wc, "base_val.txt"),
    output:
        agg         = "results/{expt}/features/shap_{mode}/shap_aggregated.csv",
        per_fold    = "results/{expt}/features/shap_{mode}/shap_per_fold_mean_abs.csv",
        all_shap    = "results/{expt}/features/shap_{mode}/shap_all_transcripts.csv",
        all_preds   = "results/{expt}/features/shap_{mode}/all_predictions.csv",
        perf_summ   = "results/{expt}/features/shap_{mode}/performance_summary.csv",
        imp_plot    = "results/{expt}/features/shap_{mode}/shap_importance_mean_std.png",
        heatmap     = "results/{expt}/features/shap_{mode}/shap_fold_heatmap.png",
        cherry      = "results/{expt}/features/shap_{mode}/cherry_pick_summary.csv",
    params:
        in_dir       = "results/{expt}/features/shap_{mode}",
        out_dir      = "results/{expt}/features/shap_{mode}",
        n_folds      = lambda wc: _n_folds(wc),
        top_n        = lambda wc: _shap_cfg(wc).get("top_n_features", 20),
        cherry_json  = lambda wc: _shap_cfg(wc).get("cherry_picks_file", ""),
    log:
        "logs/{expt}/features/shap_{mode}/aggregate.log",
    threads: 1
    resources:
        mem_mb  = 16000,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/shap_aggregate.py \
            --input-dir    {params.in_dir} \
            --output-dir   {params.out_dir} \
            --n-folds      {params.n_folds} \
            --top-n        {params.top_n} \
            --dataset      {wildcards.expt} \
            --cherry-picks "{params.cherry_json}" \
        2>&1 | tee {log}
        """


# ── Convenience target rules ──────────────────────────────────────────────────

rule shap_testing:
    """Quick test: 100 transcripts per fold, filtered features → shap_testing."""
    input:
        expand(
            "results/{expt}/features/shap_{mode}/shap_aggregated.csv",
            expt=_shap_run_expts(),
            mode=["testing"],
        ),


rule shap_full_all:
    """Full-feature SHAP pipeline across all folds."""
    input:
        expand(
            "results/{expt}/features/shap_{mode}/shap_aggregated.csv",
            expt=_shap_run_expts(),
            mode=["full"],
        ),


rule shap_clustered_all:
    """Correlation-filtered (~100-200 features) SHAP pipeline across all folds."""
    input:
        expand(
            "results/{expt}/features/shap_clustered/shap_aggregated.csv",
            expt=_shap_run_expts(),
        ),


rule shap_all:
    """Run both full and clustered pipelines."""
    input:
        rules.shap_full_all.input,
        rules.shap_clustered_all.input,
