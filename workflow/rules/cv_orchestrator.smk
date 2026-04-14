import os
from pathlib import Path

configfile: "config/config.yaml"

# Include tool-specific rules needed for CV
# NOTE: These rules need to be adapted to work with CV folder structure
# or you need to create CV-specific config entries

include: "train.smk"
include: "lncfinder.smk"
include: "feelnc.smk"
include: "cpat.smk"
include: "mrnn_cv.smk"
include: "plncpro.smk"
include: "process.smk"
include: "lncDC.smk"
#include: "lncrnabert.smk" -> No need here, already included in train.smk

# Uncomment these as you adapt each tool for CV:
# include: "workflow/rules/lncfinder.smk"
# include: "workflow/rules/plncpro.smk"
# include: "workflow/rules/lncrnabert.smk"
# include: "workflow/rules/rnasamba.smk"
# etc.

DEFAULT_N_FOLDS = 1

rule all_cv:
    input:
        "results/cv_orchestrator/final_report.txt"


#############################
# DATASET PREPARATION RULES #
#############################
def check_use_of_common_transcripts(wildcards):
    """
    Input function that returns the appropriate reference fasta file
    based on the common transcripts setting specified in the config.
    Params:
        wildcards: Snakemake wildcards object
    Returns:
        str or None: Path to the common transcripts fasta file, or None if not applicable.
    """
    expt = wildcards.expt
    expt_config = config["experiments"][expt]
    if "preprocessing" in expt_config:
        prep = expt_config.get("preprocessing", {})
        common = prep.get("common_with", None)
        if common is not None:
            ref = expt_config["reference"]
            # Validate and convert version numbers to integers
            try:
                ref_ver = int(config["databases"][ref]["version"])
            except (ValueError, TypeError, KeyError):
                raise ValueError(
                    f"Database version for '{ref}' must be convertible to int, "
                    f"got: {config['databases'][ref].get('version', 'MISSING')}"
                )
            try:
                old_ver = int(config["databases"][common]["version"])
            except (ValueError, TypeError, KeyError):
                raise ValueError(
                    f"Database version for '{common}' must be convertible to int, "
                    f"got: {config['databases'][common].get('version', 'MISSING')}"
                )
            return f"results/gencode_comparison/v{old_ver}_vs_v{ref_ver}/{ref}.common_same_class_transcripts.fa"

    return None


def check_use_of_redundancy_reduction(wildcards):
    """
    Input function that returns the appropriate reference fasta file
    based on the redundancy reduction setting specified in the config.
    Params:
        wildcards: Snakemake wildcards object
    Returns:
        str or None: Path to the redundancy-reduced fasta file, or None if not applicable
    """
    expt = wildcards.expt
    expt_config = config["experiments"][expt]
    if "preprocessing" in expt_config:
        prep = expt_config.get("preprocessing", {})
        redundancy = prep.get("redundancy", "default")
        if redundancy == "cdhit":
            return f"results/{expt}/datasets/cdhit/{expt}_cdhit90.fa"
        elif redundancy == "1tpg":
            return f"results/{expt}/datasets/1tpg/{expt}_1tpg.fa"
    else:
        # No preprocessing specified
        return None



rule combine_pc_and_lnc_cv:
    input:
        pc=lambda wc: config["experiments"][wc.expt]["pc_fasta"],
        lnc=lambda wc: config["experiments"][wc.expt]["lnc_fasta"]
    output:
        "results/{expt}/datasets/{expt}.pc_and_lnc.fa",
    log:
        "logs/{expt}/datasets/combine_pc_and_lnc_{expt}.log"
    shell:
        """
        cat {input.pc} {input.lnc} > {output} 2> {log}
        echo "Combined PC and lncRNA sequences into {output}" >> {log}
        """


def remove_redundancy_with_cdhit_input(wildcards):
    """
    Input function for remove_redundancy_with_cdhit rule.
    Basically reuses check_use_of_common_transcripts but adding a default case.
    Params:
        wildcards: Snakemake wildcards object
    Returns:
        str: Path to the input fasta file for redundancy removal.
    """
    common = check_use_of_common_transcripts(wildcards)
    if common is not None:
        return common
    else:
        expt = wildcards.expt
        return f"results/{expt}/datasets/{expt}.pc_and_lnc.fa"


rule remove_redundancy_with_cdhit:
    input:
        remove_redundancy_with_cdhit_input
    output:
        # TODO: Make percentage configurable
        fasta="results/{expt}/datasets/cdhit/{expt}_cdhit90.fa",
        clusters="results/{expt}/datasets/cdhit/{expt}_cdhit90.fa.clstr"
    log:
        "logs/{expt}/datasets/remove_redundancy_with_cdhit_{expt}.log",
    benchmark:
        "benchmarks/{expt}/datasets/remove_redundancy_with_cdhit_{expt}.tsv"
    conda:
        "../envs/cdhit_env.yaml"
    threads: 60
    resources:
        mem_mb=10240  # 10 GB
    params:
        homology=0.9,
        word_size=8,
        memory=0,  # use all available memory (limited by snakemake/slurm)
        extra="",
    shell:
        """
        set -x
        cd-hit-est \
        -i {input} \
        -o {output.fasta} \
        -c 0.9 \
        -n 8 \
        -d 0 \
        -T {threads} \
        -M {params.memory} \
        {params.extra} > {log} 2>&1
        """


def prepare_cv_splits_input(wildcards):
    """Input function to prepare_cv_splits rule."""
    expt = wildcards.expt
    pc_fasta = config["experiments"][expt]["pc_fasta"]
    lnc_fasta = config["experiments"][expt]["lnc_fasta"]

    # fasta_file depends on preprocessing steps
    redundancy = check_use_of_redundancy_reduction(wildcards)
    if redundancy is not None:
        fasta_file = redundancy
    else:
        common = check_use_of_common_transcripts(wildcards)
        if common is not None:
            fasta_file = common
        else:
            fasta_file = config["datasets"][expt]["fasta"]

    return {
        "fasta": fasta_file,
        "pc_file": pc_fasta,
        "lnc_file": lnc_fasta
    }


# One common split function for all CV folds
rule prepare_cv_splits:
    input:
        unpack(prepare_cv_splits_input)
    output:
        "results/{expt}/datasets/cv_split.done",
        #multiext("results/{expt}/datasets/fold1/", "test_all.fa", "test_pc.fa", "test_lnc.fa", "train_pc.fa", "train_lnc.fa")
        # Outputs for fold1, so that Snakemake can infer this rule is the one producing the splits
    log:
        "logs/{expt}/datasets/prepare_cv_splits.log",
    conda:
        'lnc-datasets'
    params:
        n_splits=lambda wc: config["experiments"][wc.expt].get("n_folds", DEFAULT_N_FOLDS),
        seed=42,
        output_dir=lambda wc: f"results/{wc.expt}/datasets/",
    script:
        "../scripts/cv_split.py"


rule aggregate_cv_splits:
    """
    This is a dirty workaround to avoid having to write a checkpoint for the rule
    prepare_cv_splits and having to rewrite all downstream rules to use aggregator functions.
    It collects the done flag and declares the expected output files for each fold.
    Note: If by mistake any rule requests a fold not created by prepare_cv_splits,
    that rule will fail with FileNotFoundError as expected. It may be good to raise the error here?
    """
    input:
        "results/{expt}/datasets/cv_split.done"
    output:
        multiext("results/{expt}/datasets/{fold}/", "test_all.fa", "test_pc.fa", "test_lnc.fa", "train_pc.fa", "train_lnc.fa")


#######################
# FOLD TRAINING RULES #
#######################
def get_cv_trained_models(wildcards):
    """Helper function to get paths to trained models for a given CV fold."""
    tool_list = [
            ("cpat", "logit.RData",),
            ("lncfinder", "{fold}_ss.RData",),
            ("lncfinder", "{fold}_no-ss.RData",),
            ("plncpro", "{fold}.model",),
            ("lncDC", "",),
            ("lncDC_ss", "",),
            ("mRNN", "trained/best_models/",),
            ("lncrnabert", "models/",),
            ("rnasamba", "{fold}_full.hdf5",)
    ]

    expt = wildcards.expt
    fold = wildcards.fold
    basedir = f"results/{expt}/training/{fold}/{{tool}}"
    models = []
    for tool, path_template in tool_list:
        model_path = os.path.join(basedir.format(tool=tool), path_template.format(expt=expt, fold=fold))
        models.append(model_path)
    return models


# Request trained models for each fold
rule cv_model_training:
    input:
        get_cv_trained_models
    output:
        touch("results/{expt}/training/{fold}/training.done")
    log:
        "logs/{expt}/training/{fold}/cv_model_training.log"
    shell:
        """
        echo "Gathering trained models for {wildcards.expt} {wildcards.fold}..." >> {log}
        touch {output}
        """


def get_cv_trained_models_for_tool(wildcards):
    """Helper function to get paths to trained models for a given CV fold and tool."""
    expt = wildcards.expt
    tool_name = wildcards.tool
    n_folds = config["experiments"][expt].get("n_folds", DEFAULT_N_FOLDS)

    tool_patterns = {
        "cpat": "{fold}.logit.RData",
        "lncfinder": ["{fold}_ss.RData", "{fold}_no-ss.RData"],
        "plncpro": "{fold}.model",
        "lncDC": "",
        "lncDC_ss": "",
        "mRNN": "trained/best_models/",
        "lncrnabert": "kmer/models/",
        "rnasamba": "{fold}_full.hdf5"
    }

    if tool_name not in tool_patterns:
        raise ValueError(f"Tool '{tool_name}' not recognized.")

    patterns = tool_patterns[tool_name]
    if not isinstance(patterns, list):
        patterns = [patterns]

    models = []
    for fold in range(1, n_folds + 1):
        for pattern in patterns:
            model_path = f"results/{expt}/training/fold{fold}/{tool_name}/{pattern.format(fold=f'fold{fold}')}"
            models.append(model_path)

    return models


rule cv_train_all_folds_for_tool:
    input:
        get_cv_trained_models_for_tool
    output:
        touch("results/{expt}/training/{tool}.done")
    wildcard_constraints:
        tool="(cpat|lncfinder|plncpro|lncDC|lncDC_ss|mRNN|lncrnabert|rnasamba)"
    log:
        "logs/{expt}/training/cv_train_all_folds_for_{tool}.log"
    shell:
        """
        echo "Gathering trained models for {wildcards.expt} all folds for tool {wildcards.tool}..." >> {log}
        touch {output}
        """


######################
# FOLD TESTING RULES #
######################
def get_cv_test_results(wildcards):
    """Helper function to get paths to trained models for a given CV fold."""
    expt = wildcards.expt
    tool_name = wildcards.tool
    n_folds = config["experiments"][expt].get("n_folds", DEFAULT_N_FOLDS)

    tool_patterns = {
        "FEELnc": "{fold}_RF.txt",
        "CPAT": ["{fold}.cpat.l.ORF_prob.best.tsv", "{fold}.cpat.p.ORF_prob.best.tsv"],
        "lncfinder": ["{fold}_ss.lncfinder", "{fold}_no-ss.lncfinder"],
        "plncpro": "{fold}.plncpro",
        "lncDC": ["{fold}.lncDC.no_ss.csv", "{fold}.lncDC.ss.csv"],
        "mRNN": "{fold}.mRNN.multi.tsv",
        "lncrnabert": "kmer/classification.csv",
        "rnasamba": "{fold}_full.txt"
    }

    if tool_name not in tool_patterns:
        raise ValueError(f"Tool '{tool_name}' not recognized.")

    patterns = tool_patterns[tool_name]
    if not isinstance(patterns, list):
        patterns = [patterns]

    basedir = f"results/{expt}/testing/{fold}/{{tool}}"
    models = []
    for tool, path_template in tool_list:
        model_path = os.path.join(basedir.format(tool=tool), path_template.format(fold=fold))
        models.append(model_path)
    return models


# Request test results for each fold
rule cv_model_testing:
    input:
        get_cv_test_results
    output:
        touch("results/{expt}/testing/{fold}/testing.done")
    log:
        "logs/{expt}/testing/{fold}/cv_model_testing.log"
    shell:
        """
        echo "Gathering testing results for {wildcards.expt} fold {wildcards.fold}..." >> {log}
        touch {output}
        """

def get_cv_tested_models_for_tool(wildcards):
    """Helper function to get paths to trained models for a given CV fold and tool."""
    expt = wildcards.expt
    tool_name = wildcards.tool
    n_folds = config["experiments"][expt].get("n_folds", DEFAULT_N_FOLDS)

    tool_patterns = {
        "FEELnc": "{fold}_RF.txt",
        "cpat": ["{fold}.cpat.l.ORF_prob.best.tsv", "{fold}.cpat.p.ORF_prob.best.tsv", "results/{expt}/training/{fold}/cpat/cv/optimal_cutoff.txt"],
        "lncfinder": ["{fold}_ss.lncfinder", "{fold}_no-ss.lncfinder"],
        "plncpro": "{fold}.plncpro",
        "lncDC": ["{fold}.lncDC.no_ss.csv", "{fold}.lncDC.ss.csv"],
        "mRNN": "{fold}.mRNN.multi.tsv",
        "lncrnabert": "kmer/classification.csv",
        "rnasamba": "{fold}_full.tsv"
    }

    if tool_name not in tool_patterns:
        raise ValueError(f"Tool '{tool_name}' not recognized.")

    patterns = tool_patterns[tool_name]
    if not isinstance(patterns, list):
        patterns = [patterns]

    models = []
    for fold in range(1, n_folds + 1):
        for pattern in patterns:
            model_path = f"results/{expt}/testing/fold{fold}/{tool_name}/{pattern.format(fold=f'fold{fold}')}"
            models.append(model_path)

    return models


rule cv_test_all_folds_for_tool:
    input:
        get_cv_tested_models_for_tool
    output:
        touch("results/{expt}/testing/{tool}.done")
    wildcard_constraints:
        tool="(cpat|lncfinder|plncpro|lncDC|lncDC_ss|mRNN|lncrnabert|rnasamba|FEELnc)"
    log:
        "logs/{expt}/testing/cv_test_all_folds_for_{tool}.log"
    shell:
        """
        echo "Gathering trained models for {wildcards.expt} all folds for tool {wildcards.tool}..." >> {log}
        touch {output}
        """


# Helper function for cv_training_orchestrator input
def get_cv_training_inputs(wildcards):
    """Expand training fold outputs for the orchestrator rule."""
    expt = wildcards.expt
    n_folds = config["experiments"][expt].get("n_folds", DEFAULT_N_FOLDS)
    return expand(
        "results/{expt}/training/{fold}/training.done",
        expt=expt,
        fold=[f"fold{f}" for f in range(1, n_folds + 1)]
    )


# Main orchestrator rule
rule cv_training_orchestrator:
    input:
        get_cv_training_inputs
    output:
        "results/{expt}/training/cv_training.done"
    log:
        "logs/{expt}/training/cv_training_orchestrator.log"
    shell:
        """
        echo "CV training complete for {wildcards.expt}" > {log}
        touch {output}
        """
