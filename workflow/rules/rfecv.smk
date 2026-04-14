# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  rfecv.smk — RFECV feature-selection pipeline                            ║
# ║                                                                          ║
# ║  Included after shap.smk; shares helpers _shap_cfg / _n_folds.          ║
# ║                                                                          ║
# ║  Rules                                                                   ║
# ║    rfecv_select    — per-fold RFECV + permutation importance             ║
# ║    rfecv_consensus — majority-vote consensus across folds                ║
# ║                                                                          ║
# ║  Convenience targets                                                     ║
# ║    rfecv_testing / rfecv_all       — RFECV-only                         ║
# ║    rfecv_shap_test / rfecv_shap_all — end-to-end RFECV → SHAP          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── RFECV config helpers ───────────────────────────────────────────────────


def _rfecv_shared_cfg(wc):
    """Return the top-level rfecv_select block (shared params) for an experiment."""
    return config["shap"][wc.expt].get("rfecv_select", {})


def _rfecv_cluster_arg(wc):
    """Return --cluster-file argument only when feature_mode=filtered."""
    if _rfecv_shared_cfg(wc).get("feature_mode", "full") != "filtered":
        return ""
    cfile = _rfecv_shared_cfg(wc).get("cluster_file", "")
    return f"--cluster-file '{cfile}'" if cfile else ""


def _rfecv_1se(wc):
    return "--use-1se-rule" if _rfecv_shared_cfg(wc).get("use_1se_rule", False) else ""


def _rfecv_max_transcripts(wc):
    val = _rfecv_shared_cfg(wc).get("max_transcripts")
    return str(val) if val else "none"


def _rfecv_feature_mode_arg(wc):
    """Return --feature-mode argument for the rfecv_select rule."""
    mode = _rfecv_shared_cfg(wc).get("feature_mode", "full")
    return f"--feature-mode {mode}"


def _rfecv_run_expts():
    """Experiments included in RFECV convenience targets."""
    return list(config.get("shap_rfecv_experiments", []))


# ── Per-fold RFECV rule ───────────────────────────────────────────────────

rule rfecv_select:
    """RFECV with permutation-importance RF to identify top discriminating features.

    Runs on one fold (default fold 1) and outputs a cluster_rfecv CSV that is
    directly compatible with shap_train_fold.py --cluster-file / --cluster-threshold.
    """
    input:
        train_pc          = "results/{expt}/datasets/fold{fold}/train_pc.fa",
        train_lnc         = "results/{expt}/datasets/fold{fold}/train_lnc.fa",
        optimal_threshold = "results/{expt}/features/clustering/optimal_threshold.txt",
    output:
        sel_csv   = "results/{expt}/features/rfecv/fold{fold}/rfecv_feature_selection.csv",
        cv_scores = "results/{expt}/features/rfecv/fold{fold}/rfecv_cv_scores.csv",
        perm_imp  = "results/{expt}/features/rfecv/fold{fold}/permutation_importance.csv",
        cv_plot   = "results/{expt}/features/rfecv/fold{fold}/rfecv_cv_curve.png",
        imp_plot  = "results/{expt}/features/rfecv/fold{fold}/rfecv_importance.png",
    params:
        out_dir           = "results/{expt}/features/rfecv/fold{fold}",
        te_arg            = lambda wc: _opt_arg("--te-features", _shap_cfg(wc).get("te_features")),
        nbd_arg           = lambda wc: _opt_arg("--nbd-features", _shap_cfg(wc).get("nbd_features")),
        cluster_file_arg  = lambda wc: _rfecv_cluster_arg(wc),
        cluster_threshold = lambda wc, input: (
            _shap_cfg(wc).get("cluster_threshold") or open(input.optimal_threshold).read().strip()
        ),
        max_transcripts   = lambda wc: _rfecv_max_transcripts(wc),
        feature_mode_arg  = lambda wc: _rfecv_feature_mode_arg(wc),
        cv_folds          = lambda wc: _rfecv_shared_cfg(wc).get("cv_folds", 5),
        cv_repeats        = lambda wc: _rfecv_shared_cfg(wc).get("cv_repeats", 3),
        scoring           = lambda wc: _rfecv_shared_cfg(wc).get("scoring", "balanced_accuracy"),
        rfe_step          = lambda wc: _rfecv_shared_cfg(wc).get("rfe_step", 1),
        perm_n_repeats    = lambda wc: _rfecv_shared_cfg(wc).get("perm_n_repeats", 5),
        n_trees           = lambda wc: _rfecv_shared_cfg(wc).get("n_trees", 200),
        top_n_plot        = lambda wc: _rfecv_shared_cfg(wc).get("top_n_plot", 30),
        random_state      = lambda wc: _shap_cfg(wc).get("random_state", 42),
        use_1se           = lambda wc: _rfecv_1se(wc),
        force_rerun       = lambda wc: "--force-rerun" if _rfecv_shared_cfg(wc).get("force_rerun", False) else "",
    log:
        "logs/{expt}/features/rfecv/fold{fold}/rfecv_select.log",
    threads: 20
    resources:
        mem_mb  = 50000,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/shap_rfecv.py \
            --dataset           {wildcards.expt} \
            --fold              {wildcards.fold} \
            --results-dir       results \
            --output-dir        {params.out_dir} \
            {params.te_arg} \
            {params.nbd_arg} \
            --max-transcripts   {params.max_transcripts} \
            --cv-folds          {params.cv_folds} \
            --cv-repeats        {params.cv_repeats} \
            --scoring           {params.scoring} \
            --rfe-step          {params.rfe_step} \
            --perm-n-repeats    {params.perm_n_repeats} \
            --n-trees           {params.n_trees} \
            --top-n-plot        {params.top_n_plot} \
            --cluster-threshold {params.cluster_threshold} \
            --random-state      {params.random_state} \
            {params.feature_mode_arg} \
            {params.cluster_file_arg} \
            {params.use_1se} \
            {params.force_rerun} \
            --n-jobs {threads} \
        2>&1 | tee {log}
        """


# ── RFECV consensus rule ───────────────────────────────────────────────────

rule rfecv_consensus:
    """Majority-vote consensus across per-fold RFECV selections."""
    input:
        sel_csvs = lambda wc: expand(
            "results/{expt}/features/rfecv/fold{fold}/rfecv_feature_selection.csv",
            expt=wc.expt, fold=range(1, _n_folds(wc) + 1),
        ),
    output:
        consensus_json = "results/{expt}/features/rfecv/consensus_features.json",
        consensus_csv  = "results/{expt}/features/rfecv/consensus_summary.csv",
        consensus_plot = "results/{expt}/features/rfecv/consensus_votes.png",
    params:
        input_dir  = "results/{expt}/features/rfecv",
        output_dir = "results/{expt}/features/rfecv",
        n_folds    = lambda wc: _n_folds(wc),
        min_folds  = lambda wc: _rfecv_shared_cfg(wc).get("consensus_min_folds", 3),
    log:
        "logs/{expt}/features/rfecv/consensus.log",
    threads: 1
    resources:
        mem_mb  = 4000,
    conda:
        "lnc-datasets"
    shell:
        """
        python -u workflow/scripts/shap_rfecv_consensus.py \\
            --input-dir     {params.input_dir} \\
            --output-dir    {params.output_dir} \\
            --n-folds       {params.n_folds} \\
            --min-folds     {params.min_folds} \\
        2>&1 | tee {log}
        """


# ── RFECV-only convenience targets ────────────────────────────────────────

rule rfecv_testing:
    """RFECV on all folds (subsampled per config) + consensus aggregation."""
    input:
        expand(
            "results/{expt}/features/rfecv/consensus_features.json",
            expt=_rfecv_run_expts(),
        ),


rule rfecv_all:
    """Full RFECV feature selection on all folds + consensus aggregation."""
    input:
        expand(
            "results/{expt}/features/rfecv/consensus_features.json",
            expt=_rfecv_run_expts(),
        ),


# ── RFECV + SHAP end-to-end convenience targets ───────────────────────────

rule rfecv_shap_test:
    """End-to-end test: RFECV (1000 transcripts, fold 1) → SHAP (all folds) → aggregate."""
    input:
        expand(
            "results/{expt}/features/shap_rfecv_test/shap_aggregated.csv",
            expt=_rfecv_run_expts(),
        ),


rule rfecv_shap_all:
    """Full run: RFECV on all training transcripts → consensus → SHAP all transcripts."""
    input:
        expand(
            "results/{expt}/features/shap_rfecv_full/shap_aggregated.csv",
            expt=_rfecv_run_expts(),
        ),
