configfile: "config/new_config.yaml"


N_FOLD = 5

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
            ref_ver = config["databases"][ref]["version"]
            old_ver = config["databases"][common]["version"]
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


# One common split function for all CV folds
rule prepare_cv_splits:
    input:
        unpack(prepare_cv_splits_input)
    output:
        "results/{expt}/datasets/folds/cv_split.done"
    log:
        "logs/{expt}/datasets/prepare_cv_splits.log",
    conda:
        'lnc-datasets'
    params:
        n_splits=N_FOLD,
        seed=42,
        output_dir=lambda wc: f"results/{wc.expt}/datasets/folds/",
    script:
        "../scripts/cv_split.py"


def get_cv_trained_models(wildcards):
    """Helper function to get paths to trained models for a given CV fold."""
    tool_list = [
            ("cpat", "{dset}_fold{fold}.logit.RData",),
            ("lncfinder", "{dset}_fold{fold}_ss.RData",),
            ("lncfinder", "{dset}_fold{fold}_no-ss.RData",),
            ("plncpro", "{dset}_fold{fold}.model",),
            ("lncDC", "",),
            ("lncDC_ss", "",),
            ("mrnn", "trained/best_models/",),
            ("lncrnabert", "models/",),
            ("rnasamba", "{dset}_fold{fold}.hdf5",)
    ]

    dset = wildcards.dset
    fold = wildcards.fold
    basedir = f"results/{dset}/training/fold{fold}/{{tool}}"
    models = []
    for tool, path_template in tool_list:
        model_path = os.path.join(basedir.format(tool=tool), path_template.format(dset=dset, fold=fold))
        models.append(model_path)
    return models


# Request trained models for each fold
rule cv_model_training:
    input:
        get_cv_trained_models
    output:
        touch("results/{expt}/training/fold{fold}/training.done")
    log:
        "logs/{expt}/training/fold{fold}/cv_model_training.log"
    shell:
        """
        echo "Gathering trained models for {wildcards.expt} fold {wildcards.fold}..." >> {log}
        touch {output}
        """
