import os

DEFAULT_N_FOLDS = 1
_config = None
_expand = None


def configure(workflow_config, expand_function):
    global _config, _expand
    _config = workflow_config
    _expand = expand_function


def check_use_of_common_transcripts(wildcards):
    expt = wildcards.expt
    expt_config = _config["experiments"][expt]
    if "preprocessing" in expt_config:
        prep = expt_config.get("preprocessing", {})
        common = prep.get("common_with")
        if common is not None:
            ref = expt_config["reference"]
            try:
                ref_ver = int(_config["databases"][ref]["version"])
            except (ValueError, TypeError, KeyError):
                raise ValueError(
                    f"Database version for '{ref}' must be convertible to int, "
                    f"got: {_config['databases'][ref].get('version', 'MISSING')}"
                )
            try:
                old_ver = int(_config["databases"][common]["version"])
            except (ValueError, TypeError, KeyError):
                raise ValueError(
                    f"Database version for '{common}' must be convertible to int, "
                    f"got: {_config['databases'][common].get('version', 'MISSING')}"
                )
            return (
                "results/gencode_comparison/"
                f"v{old_ver}_vs_v{ref_ver}/{ref}.common_same_class_transcripts.fa"
            )
    return None


def check_use_of_redundancy_reduction(wildcards):
    expt = wildcards.expt
    expt_config = _config["experiments"][expt]
    if "preprocessing" in expt_config:
        redundancy = expt_config.get("preprocessing", {}).get("redundancy", "default")
        if redundancy == "cdhit":
            return f"results/{expt}/datasets/cdhit/{expt}_cdhit90.fa"
        if redundancy == "1tpg":
            return f"results/{expt}/datasets/1tpg/{expt}_1tpg.fa"
    return None


def remove_redundancy_with_cdhit_input(wildcards):
    common = check_use_of_common_transcripts(wildcards)
    if common is not None:
        return common
    return f"results/{wildcards.expt}/datasets/{wildcards.expt}.pc_and_lnc.fa"


def prepare_cv_splits_input(wildcards):
    expt = wildcards.expt
    pc_fasta = _config["experiments"][expt]["pc_fasta"]
    lnc_fasta = _config["experiments"][expt]["lnc_fasta"]
    redundancy = check_use_of_redundancy_reduction(wildcards)
    if redundancy is not None:
        fasta_file = redundancy
    else:
        common = check_use_of_common_transcripts(wildcards)
        fasta_file = (
            common if common is not None else _config["experiments"][expt]["fasta"]
        )
    return {"fasta": fasta_file, "pc_file": pc_fasta, "lnc_file": lnc_fasta}


def get_cv_trained_models(wildcards):
    tool_list = [
        ("cpat", "{fold}.logit.RData"),
        ("lncfinder", "{fold}_ss.RData"),
        ("lncfinder", "{fold}_no-ss.RData"),
        ("plncpro", "{fold}.model"),
        ("lncDC", ""),
        ("lncDC_ss", ""),
        ("mRNN", "trained/best_models/"),
        ("lncrnabert", "kmer/models/"),
        ("rnasamba", "{fold}_full.hdf5"),
    ]
    expt = wildcards.expt
    fold = wildcards.fold
    basedir = f"results/{expt}/training/{fold}/{{tool}}"
    return [
        os.path.join(
            basedir.format(tool=tool), path_template.format(expt=expt, fold=fold)
        )
        for tool, path_template in tool_list
    ]


def get_cv_trained_models_for_tool(wildcards):
    expt = wildcards.expt
    tool_name = wildcards.tool
    n_folds = _config["experiments"][expt].get("n_folds", DEFAULT_N_FOLDS)
    tool_patterns = {
        "cpat": "{fold}.logit.RData",
        "lncfinder": ["{fold}_ss.RData", "{fold}_no-ss.RData"],
        "plncpro": "{fold}.model",
        "lncDC": "",
        "lncDC_ss": "",
        "mRNN": "trained/best_models/",
        "lncrnabert": "kmer/models/",
        "rnasamba": "{fold}_full.hdf5",
    }
    if tool_name not in tool_patterns:
        raise ValueError(f"Tool '{tool_name}' not recognized.")
    patterns = tool_patterns[tool_name]
    if not isinstance(patterns, list):
        patterns = [patterns]
    return [
        f"results/{expt}/training/fold{fold}/{tool_name}/{pattern.format(fold=f'fold{fold}')}"
        for fold in range(1, n_folds + 1)
        for pattern in patterns
    ]


def get_cv_test_results(wildcards):
    tool_list = [
        ("FEELnc", "{fold}_RF.txt"),
        ("cpat", "{fold}.cpat.l.ORF_prob.best.tsv"),
        ("cpat", "{fold}.cpat.p.ORF_prob.best.tsv"),
        ("lncfinder", "{fold}_ss.lncfinder"),
        ("lncfinder", "{fold}_no-ss.lncfinder"),
        ("plncpro", "{fold}.plncpro"),
        ("lncDC", "{fold}.lncDC.no_ss.csv"),
        ("lncDC", "{fold}.lncDC.ss.csv"),
        ("mRNN", "{fold}.mRNN.multi.tsv"),
        ("lncrnabert", "kmer/classification.csv"),
        ("rnasamba", "{fold}_full.tsv"),
    ]
    expt = wildcards.expt
    fold = wildcards.fold
    basedir = f"results/{expt}/testing/{fold}/{{tool}}"
    return [
        os.path.join(
            basedir.format(tool=tool), path_template.format(expt=expt, fold=fold)
        )
        for tool, path_template in tool_list
    ]


def get_cv_tested_models_for_tool(wildcards):
    expt = wildcards.expt
    tool_name = wildcards.tool
    n_folds = _config["experiments"][expt].get("n_folds", DEFAULT_N_FOLDS)
    tool_patterns = {
        "FEELnc": "{fold}_RF.txt",
        "cpat": [
            "{fold}.cpat.l.ORF_prob.best.tsv",
            "{fold}.cpat.p.ORF_prob.best.tsv",
            "results/{expt}/training/{fold}/cpat/cv/optimal_cutoff.txt",
        ],
        "lncfinder": ["{fold}_ss.lncfinder", "{fold}_no-ss.lncfinder"],
        "plncpro": "{fold}.plncpro",
        "lncDC": ["{fold}.lncDC.no_ss.csv", "{fold}.lncDC.ss.csv"],
        "mRNN": "{fold}.mRNN.multi.tsv",
        "lncrnabert": "kmer/classification.csv",
        "rnasamba": "{fold}_full.tsv",
    }
    if tool_name not in tool_patterns:
        raise ValueError(f"Tool '{tool_name}' not recognized.")
    patterns = tool_patterns[tool_name]
    if not isinstance(patterns, list):
        patterns = [patterns]
    return [
        f"results/{expt}/testing/fold{fold}/{tool_name}/{pattern.format(fold=f'fold{fold}', expt=expt)}"
        for fold in range(1, n_folds + 1)
        for pattern in patterns
    ]


def get_cv_training_inputs(wildcards):
    expt = wildcards.expt
    n_folds = _config["experiments"][expt].get("n_folds", DEFAULT_N_FOLDS)
    return _expand(
        "results/{expt}/training/{fold}/training.done",
        expt=expt,
        fold=[f"fold{f}" for f in range(1, n_folds + 1)],
    )


def get_all_cv_testing_inputs(wildcards):
    for expt in _config["to_train"]:
        n_folds = _config["experiments"][expt].get("n_folds", DEFAULT_N_FOLDS)
        for fold in range(1, n_folds + 1):
            yield f"results/{expt}/testing/fold{fold}/testing.done"
