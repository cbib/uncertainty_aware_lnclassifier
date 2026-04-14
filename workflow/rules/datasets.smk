configfile: "config/config.yaml"
'''
ruleorder: split_pc_and_lnc_after_cdhit > one_transcript_per_gene_split
'''
rule all_datasets:
    input:
        #expand("resources/training/{db}/{db}_cdhit90.fa", db=config["training"])
        #expand("results/{db}/training/{db}.train_pc.fa", db="gencode.v47.nochange"),
        expand("results/{db}/training/{db}.inference.fa", db=config["training"]),

'''
rule create_training_toy:
    input:
        "resources/{db}.{type}_transcripts.fa.gz",
    output:
        "resources/training/{db}.{type}_transcripts.fa",
    params:
        n=1000,
        extra="-s 42",  # -s Seed for reproducibility
    conda:
        "base"
    shell:
        "python workflow/scripts/create_toy.py {input} {output} {params.n} {params.extra}"


rule one_transcript_per_gene_split:
    # Rule creating a training/testing split following the method of FEELnc
    input:
        db="resources/{db}.transcripts.fa",
        pc="resources/{db}.pc_transcripts.fa",
        lnc="resources/{db}.lncRNA_transcripts.fa",
        gtf="resources/annotation/{db}.chr_patch_hapl_scaff.annotation.gtf",
    output:
        multiext(
            "results/{db}/training/1tpg/{db}",
            ".train_pc.fa",
            ".train_lnc.fa",
            ".test_pc.fa",
            ".test_lnc.fa",
            ".redun_pc.fa",
            ".redun_lnc.fa",
        ),
        one_per_gene="results/{db}/training/1tpg/{db}.one_transcript_per_gene.tsv",
    params:
        seed=42,  # Random seed for subsets
        split=20,  # Percentage of train data
        balanced=False,  # Whether the classes need to be balanced
        out_dir=lambda wc, output: subpath(output.one_per_gene, parent=True),
        prefix=lambda wc: f"{wc.db}",
    conda:
        "lnc-datasets"
    script:
        "../scripts/one_transcript_per_gene_split.py"
'''

rule train_plus_validation_dataset:
    # Some tools not only require a training set, but also a validation set
    # To allow for comparison, we extract the validation set as a subset of the training set
    # The split percentage defined in the config file denote:
    #   - The percentage of the total data used for testing
    #   - The percentage of the total data that should be used for validation
    # Therefore, for tools requiring validation, the actual training split is reduced to training - validation
    input:
        pc="results/{expt}/datasets/{fold}/train_pc.fa",
        lnc="results/{expt}/datasets/{fold}/train_lnc.fa",
    output:
        pc_train="results/{expt}/training/{fold}/{tool}/datasets/{tool}_train_pc.fa",
        lnc_train="results/{expt}/training/{fold}/{tool}/datasets/{tool}_train_lnc.fa",
        pc_valid="results/{expt}/training/{fold}/{tool}/datasets/{tool}_valid_pc.fa",
        lnc_valid="results/{expt}/training/{fold}/{tool}/datasets/{tool}_valid_lnc.fa",
    log:
        "logs/{expt}/datasets/{fold}/{tool}/{fold}.{tool}_train_plus_validation_split.log",
    conda:
        "lnc-datasets"
    params:
        seed=None,
        train_split=lambda wc: config["training"][wc.expt]["train_split"],  # This is global for all tools benchmarked
        valid_split=lambda wc: int(
            config["training"][wc.expt][f"{wc.tool}_validation_split"]
        ),
    script:
        "../scripts/train_plus_validation_dataset.py"


def get_reference(wildcards):
    """Helper function to get reference database for a given training database"""
    dset = wildcards.dset
    if "datasets" not in config.keys():
        raise KeyError("No 'datasets' key found in config file.")
    if dset not in config["datasets"].keys():
        raise KeyError(f"Dataset '{dset}' not found in config['datasets']")
    if "reference" not in config["datasets"][dset].keys():
        raise KeyError(
            f"No 'reference' key found for dataset '{dset}' in config['datasets']"
        )
    ref_db = config["datasets"][dset]["reference"]
    return ref_db

'''
rule combine_pc_and_lnc:
    input:
        pc=lambda wc: f"resources/{get_reference(wc)}.pc_transcripts.fa",
        lnc=lambda wc: f"resources/{get_reference(wc)}.lncRNA_transcripts.fa",
    output:
        "results/{expt}/training/cdhit/{expt}.pc_and_lnc.fa",
    conda:
        "lnc-datasets"
    log:
        "logs/training/combine_pc_and_lnc_{expt}.log",
    shell:
        """
        cat {input.pc} {input.lnc} > {output} 2> {log}
        """


use rule combine_pc_and_lnc as combine_testing_pc_and_lnc with:
    input:
        pc="results/{db}/training/{db}.test_pc.fa",
        lnc="results/{db}/training/{db}.test_lnc.fa",
    output:
        "results/{db}/training/{db}.test_pc_and_lnc.fa",
    log:
        "logs/training/combine_testing_pc_and_lnc_{db}.log",
'''

rule common_transcripts_between_versions:
    input:
        # NOTE: Only compatible with Gencode for now
        # These variables could be defined in the config file if needed
        old_db_fasta="resources/gencode.{old_db}.transcripts.fa",
        old_db_pc="resources/gencode.{old_db}.pc_transcripts.fa",
        old_db_lnc="resources/gencode.{old_db}.lncRNA_transcripts.fa",
        new_db_fasta="resources/gencode.{new_db}.transcripts.fa",
        new_db_pc="resources/gencode.{new_db}.pc_transcripts.fa",
        new_db_lnc="resources/gencode.{new_db}.lncRNA_transcripts.fa",
    output:
        multiext(
            "results/gencode_comparison/{old_db}_vs_{new_db}/gencode.{new_db}",
            ".common_same_class_transcripts.fa",
            ".common_class_change_transcripts.fa",
            ".added_with_class_transcripts.fa",
            ".no_class_transcripts.fa",
            ".comparison.tsv"
        ),
    conda:
        "lnc-datasets"
    log:
        "logs/gencode_comparison/common_transcripts_{old_db}_{new_db}.log",
    params:
        output_dir=lambda wc, output: subpath(output[0], parent=True),
        old_version=lambda wc: wc.old_db,
        new_version=lambda wc: wc.new_db,
    script:
        "../scripts/common_transcripts_between_versions.py"


def get_version_number(db: str) -> int:
    """
    Helper function to extract version number from database name

    Args:
        db (str): Database name (e.g., 'gencode.v47')
    Returns:
        int: Version number (e.g., 47)
    """
    # Assuming database names are in the format 'gencode.vXX' where XX is the version number
    if "gencode.v" in db:
        return int(db.split("gencode.v")[1])
    else:
        raise ValueError(f"Cannot extract version number from database name: {db}")

'''
def cdhit_input(wildcards):
    dset = wildcards.dset
    ref = get_reference(wildcards)
    if "common" in dset:
        # Only use common transcripts between current and previous version
        print("Using only common transcripts between versions for CD-HIT redundancy removal")
        ver = get_version_number(ref)
        old_ver = ver - 1
        print(f"Reference database: {ref} (version v{ver}), compared with v{old_ver}")
        return f"results/gencode_comparison/v{old_ver}_vs_v{ver}/{ref}.common_same_class_transcripts.fa"
    else:
        # Use combined pc and lnc transcripts from the full dataset
        print("Using full dataset for CD-HIT redundancy removal")
        return f"results/{expt}/training/cdhit/{dset}.pc_and_lnc.fa"


rule remove_redundancy_with_cdhit:
    input:
        cdhit_input
    output:
        # TODO: Make percentage configurable
        fasta="results/{expt}/datasets/cdhit/{dset}_cdhit90.fa",
        clusters="results/{expt}/datasets/cdhit/{dset}_cdhit90.fa.clstr"
    log:
        "logs/{dset}/training/remove_redundancy_with_cdhit_{dset}.log",
    benchmark:
        "benchmarks/{dset}/training/remove_redundancy_with_cdhit_{dset}.tsv"
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


def inputs_split_after_cdhit(wildcards):
    dset = wildcards.dset
    ref_db = config["datasets"][dset]["reference"]

    inputs = {
        "cluster": f"results/{expt}/training/cdhit/{dset}_cdhit90.fa.clstr",
        "fasta": f"resources/{ref_db}.transcripts.fa",
        "pc_ids": f"resources/{ref_db}.pc_transcripts.fa",
        "lnc_ids": f"resources/{ref_db}.lncRNA_transcripts.fa",
    }
    return inputs


rule split_pc_and_lnc_after_cdhit:
    input:
        unpack(inputs_split_after_cdhit),
    output:
        multiext(
            "results/{expt}/training/cdhit/{dset}",
            ".train_pc.fa",
            ".train_lnc.fa",
            ".test_pc.fa",
            ".test_lnc.fa",
            ".redun_pc.fa",
            ".redun_lnc.fa",
        ),
    conda:
        "lnc-datasets"
    params:
        output_prefix=lambda wc, output: os.path.join(
            subpath(output[0], parent=True), wc.dset
        ),
        train_frac=lambda wc: int(config["training"][wc.dset]["train_split"]) / 100,
        balance_train=lambda wc: config["training"][wc.dset]["balanced"],
        seed=42,
    log:
        "logs/{dset}/training/split_pc_and_lnc_after_cdhit_{dset}.log",
    benchmark:
        "benchmarks/{dset}/training/split_pc_and_lnc_after_cdhit_{dset}.tsv",
    script:
        "../scripts/split_pc_and_lnc_after_cdhit.py"


def get_training_config(dset, config_key):
    """Helper function to get training config parameters"""
    if dset not in config["datasets"]:
        raise KeyError(f"Dataset '{dset}' not found in config['datasets']")
    if "training_db" not in config["datasets"][dset]:
        raise KeyError(f"'training_db' key not found for dataset '{dset}' in config['datasets']")
    training_db = config["datasets"][dset]["training_db"]
    training_config = config["training"][training_db]
    return training_config[config_key]


def training_dataset(wildcards):
    dset = wildcards.dset
    conf = get_training_config(dset, "redundancy_reduction")
    if conf in ["cdhit", "1tpg"]:
        print(f"Using '{conf}' redundancy reduction for dataset '{dset}'")
        return multiext(
            f"results/{expt}/training/{conf}_dataset/{dset}",
            ".train_pc.fa",
            ".train_lnc.fa"
        )
    else:
        raise ValueError(
            f"Unknown redundancy reduction method '{conf}' for dataset '{dset}'"
        )


rule select_training_dataset:
    input:
        training_dataset,
    output:
        # TODO: Decide 1) copy or symlink 2) training/files.fa or training/working_dataset/files.fa
        multiext(
            "results/{expt}/training/{dset}",
            ".train_pc.fa",
            ".train_lnc.fa"
        ),
    conda:
        "base"
    log:
        "logs/{dset}/training/select_working_dataset_{dset}.log",
    run:
        # TODO: Write as a shell command
        import logging
        import sys

        if log:
            logging.basicConfig(
                filename=log[0],
                level=logging.DEBUG,
                format="%(asctime)s [%(levelname)s] %(message)s",
            )
            sys.stderr = open(log[0], "a")
        else:
            logging.basicConfig(level=logging.DEBUG)

        for c, v in enumerate(input):
            i = v
            o = output[c]
            logging.info(f"Creating symlink from {i} to {o}")
            shell(f"ln -sr {i} {o}")


def inference_dataset_components(wildcards):
    """
    Input function to get all components needed for creating the inference dataset
    By default, this includes only the test splits (test_pc.fa and test_lnc.fa).
    If the dataset configuration requires, additional components are added.
    """
    # Get wildcards and configuration
    dset = wildcards.dset
    ref = get_reference(wildcards)

    print(f"Creating inference dataset components for dataset: {dset}")

    # Dataset-specific files
    if "cdhit" in dset:
        print("Using cdhit dataset files for inference dataset creation")
        # Redundancy reduced files
        dataset_files = multiext(
            f"results/{expt}/training/cdhit/{dset}",
            ".test_pc.fa",
            ".test_lnc.fa",
            ".redun_pc.fa",
            ".redun_lnc.fa",
        )
    else:
        # Simple test split
        print("Using simple test split dataset files for inference dataset creation")
        dataset_files = multiext(
            f"results/{expt}/training/{dset}", ".test_pc.fa", ".test_lnc.fa"
        )

    # Version comparison files (independent of the dataset treatment)
    if "common" in dset:
        # This analysis uses common transcripts between versions
        print(
            "Adding transcripts from database version comparison for inference dataset creation"
        )

        ver = get_version_number(ref)
        old_ver = ver - 1  # Assuming we are comparing with the previous version

        print(f"Reference database: {ref} (version v{ver}), compared with v{old_ver}")

        version_files = multiext(
            f"results/gencode_comparison/v{old_ver}_vs_v{ver}/gencode.v{ver}",
            ".common_class_change_transcripts.fa",
            ".added_with_class_transcripts.fa",
            ".no_class_transcripts.fa"
        )

        return dataset_files + version_files

    else:
        return dataset_files


# For our benchmark, we only selected common transcripts between
# versions v46 and v47 of Gencode for training. However, we want
# to run inference on every transcript in v47.
# We need to:
# 1) Identify all files containing additional transcripts
# 2) Combine them into a new inference dataset
# 3) Create a table mapping each transcript in the inference dataset to:
#    - Its classification (PC/lncRNA/other)
#    - Whether it was present in the testing dataset split -> test_pc.fa/test_lnc.fa have such transcripts
#    - Whether it was removed by CD-HIT -> redun_pc.fa/redun_lnc.fa have such transcripts
#    - Whether it is a leftover transcript (left aside to balance training)-> These do not exist, since they are automatically included in the test_pc.fa/test_lnc.fa files
#    - Whether it a new v47 transcript -> gencode.v47.new_with_class_transcripts.fa
#    - Whether it changed classification between v46 and v47 -> gencode.v47.common_class_change_transcripts.fa
#    - Whether it is in v47, but does not have a coding class assigned (e.g. nonsense_mediated_decay, retained_intron, etc.) -> gencode.v47.no_class_transcripts.fa
rule create_inference_dataset:
    """
    NOTE: Currently not used
    Create inference dataset by combining the desired components.
    By default, this only includes the test splits (test_pc.fa and test_lnc.fa).
    Additional components can be added to the input function 'inference_dataset_components'
    depending on the dataset configuration.

    Additionally, create a table mapping each transcript in the inference dataset to
    its classification (coding vs lncRNA) and the original file source.
    """
    input:
        # Dynamic input files based on config
        pc_fasta=lambda wc: f"resources/{get_reference(wc)}.pc_transcripts.fa",  # Annotation for pc classification
        lnc_fasta=lambda wc: f"resources/{get_reference(wc)}.lncRNA_transcripts.fa",  # Annotation for lnc classification
        inference_files=inference_dataset_components,
    output:
        fasta="results/{expt}/training/{dset}.inference.fa",
        info_table="results/{expt}/training/{dset}.inference_info.tsv",
    conda:
        "lnc-datasets"
    log:
        "logs/{dset}/training/create_inference_dataset_{dset}.log",
    script:
        "../scripts/create_inference_dataset.py"


rule feelnc_dataset:
    output:
        pc='resources/training/FEELnc_datasets/FEELnc_v24_5000_pc_train.fasta',
        lnc='resources/training/FEELnc_datasets/FEELnc_v24_5000_lnc_train.fasta'
    conda:
        "lnc-datasets"
    log:
        "logs/feelnc_dataset.log"
    script:
        "workflow/scripts/create_feelnc_v24_dataset.py"

'''
