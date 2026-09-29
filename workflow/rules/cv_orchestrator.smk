configfile: "config/config.yaml"

from workflow.scripts.cv_helpers import (
    DEFAULT_N_FOLDS,
    check_use_of_common_transcripts,
    check_use_of_redundancy_reduction,
    configure,
    get_all_cv_testing_inputs,
    get_cv_test_results,
    get_cv_tested_models_for_tool,
    get_cv_trained_models,
    get_cv_trained_models_for_tool,
    get_cv_training_inputs,
    prepare_cv_splits_input,
    remove_redundancy_with_cdhit_input,
)

configure(config, expand)

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

rule all_cv:
    input:
        expand(
            "results/{expt}/training/cv_training.done",
            expt=config["to_train"]
        )


#############################
# DATASET PREPARATION RULES #
#############################
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
        '../envs/lnc-datasets_env.yaml'
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


rule cv_test_all_folds:
    """
    Target rule to gather all testing results for a given experiment across all folds.
    """
    input:
        get_all_cv_testing_inputs
