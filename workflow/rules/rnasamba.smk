import sys
import os

module gpu_helpers:
    snakefile: "gpu_helpers.smk"

use rule * from gpu_helpers

rule all_rnasamba:
    input:
        expand("results/{fold}/rnasamba/{fold}_{mode}.tsv", dset=["gencode.v47"], mode=["full"], fold="fold1"),


def get_rnasamba_model(wildcards):
    """Helper function to get the path to the RNAsamba model for a given dataset and mode."""
    wc = wildcards
    rnasamba_config = config["experiments"][wc.expt]["models"].get("rnasamba", "default")
    if rnasamba_config == "custom":
        return f"results/{wc.expt}/training/{wc.fold}/rnasamba/{wc.fold}_{wc.mode}.hdf5"
    else:
        return f"software/RNAsamba-0.2.5/data/{wc.mode}_length_weights.hdf5"


# TODO: Generalize to other rules
def get_test_set(wildcards):
    """Helper function to get the test set for a given dataset."""
    wc = wildcards
    expt_config = config["experiments"][wc.expt]
    if "custom_test_set" in expt_config:
        return expt_config["custom_test_set"]
    else:
        return "results/{expt}/datasets/{fold}/test_all.fa".format(expt=wc.expt, fold=wc.fold)

rule rnasamba_cv:
    input:
        flag="software/rnasamba.installed",
        weights_file=get_rnasamba_model,
        fasta=get_test_set
    output:
        "results/{expt}/testing/{fold}/rnasamba/{fold}_{mode}.tsv",
    wildcard_constraints:
        dset="[^/]+",
        mode="full"
    conda:
        #"workflow/envs/rnasamba_env.yaml"
        "../envs/rnasamba-gpu_env.yaml"
    resources:
        mem_mb=80000,
        time="2d"
    log:
        "logs/{expt}/testing/{fold}/rnasamba/{fold}_{mode}.log",
    benchmark:
        "benchmarks/{expt}/testing/{fold}/rnasamba/{fold}_{mode}.txt",
    shell:
        """
        rnasamba classify {output} {input.fasta} {input.weights_file} -v 1  2>&1 | tee {log}
        """


rule rnasamba:
    input:
        flag="software/rnasamba.installed",
        weights_file=get_rnasamba_model,
        fasta=get_test_set
    output:
        "results/{expt}/testing/rnasamba/{fold}_{mode}.tsv",
    wildcard_constraints:
        dset="[^/]+",
        mode="full"
    conda:
        #"workflow/envs/rnasamba_env.yaml"
        "../envs/rnasamba-gpu_env.yaml"
    resources:
        mem_mb = 25000,
        slurm_partition="gpu",
        # Option 1: Use config-based requirements
        gres="nvidia_h100_nvl:1",
        # Option 2: Direct specification (commented out)
        # gres=lambda wildcards: get_dynamic_gpu(min_vram_gb=12, max_vram_gb=30),
        cpus_per_gpu=1
    log:
        "logs/{expt}/testing/{fold}/rnasamba/{fold}_{mode}.log",
    benchmark:
        "benchmarks/{expt}/testing/{fold}/rnasamba/{fold}_{mode}.txt",
    shell:
        """
        rnasamba classify {output} {input.fasta} {input.weights_file} -v 1  2>&1 | tee {log}
        """


rule train_rnasamba_gpu:
    input:
        flag="software/rnasamba.installed",
        pc="results/{expt}/datasets/{fold}/train_pc.fa",
        lnc="results/{expt}/datasets/{fold}/train_lnc.fa"
    output:
        "results/{expt}/training/{fold}/rnasamba/gpu/{fold}_{mode}.hdf5",
    params:
        verbosity=2,
    conda:
        "../envs/rnasamba-gpu_env.yaml"
    log:
        "logs/{expt}/training/{fold}/rnasamba/training.{fold}_{mode}.log",
    benchmark:
        "benchmarks/{expt}/training/{fold}/rnasamba/training.{fold}_{mode}.txt",
    resources:
        mem_mb = 15000,
        slurm_partition="gpu",
        # Option 1: Use config-based requirements
        gres=gpu_helpers.get_gpu_for_rule_name("rnasamba_train"),
        # Option 2: Direct specification (commented out)
        # gres=lambda wildcards: get_dynamic_gpu(min_vram_gb=12, max_vram_gb=30),
        cpus_per_gpu=1
    priority: 80
    shell:
        """
        {{
            echo "Starting Rnasamba training"
            echo "GPU Info:"
            echo "Available physical GPUs:"
            nvidia-smi --query-gpu=index,platform.module_id,name,driver_version,memory.total,compute_cap,mig.mode.current --format=csv
            echo "Available MIG devices:"
            nvidia-smi -L | grep -i mig || echo "No MIG devices found"
            echo "Using visible GPU Device with index $CUDA_VISIBLE_DEVICES"
            echo "---"

            rnasamba train \
            {output} \
            {input.pc} \
            {input.lnc} \
            -v {params.verbosity}
        }} 2>&1 | tee {log}
        """

use rule train_rnasamba_gpu as rnasamba_train with:
    output:
        "results/{expt}/training/{fold}/rnasamba/{fold}_{mode}.hdf5"
