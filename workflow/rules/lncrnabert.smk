if "__rule_all_datasets" not in globals():

    include: "./datasets.smk"


module gpu_helpers:
    snakefile: "gpu_helpers.smk"

use rule * from gpu_helpers

rule all_lncrnabert:
    input:
        #expand("results/{db}/training/lncrnabert/results", db=config["training"]),
        expand("results/{db}/lncrnabert/classification.csv", db=config["training"]),


def get_lncrnabert_pretrained_model(wildcards):
    """Helper function to get the path to the lncRNA-BERT pretrained model for a given dataset."""
    wc = wildcards
    model_dict = {
        "kmer": "luukromeijn/lncRNA-BERT-kmer-k3-pretrained",
        "cse": "luukromeijn/lncRNA-BERT-CSE-k9-pretrained",
        "scratch": "",
    }
    return model_dict.get(wc.encoding, "luukromeijn/lncRNA-BERT-kmer-k3-pretrained")


wildcard_constraints:
    encoding="kmer|cse"

#########################
# TRAINING RULES
#########################

rule lncrnabert_train:
    input:
        pc_train="results/{expt}/training/{fold}/lncrnabert/datasets/lncrnabert_train_pc.fa",
        lnc_train="results/{expt}/training/{fold}/lncrnabert/datasets/lncrnabert_train_lnc.fa",
        pc_valid="results/{expt}/training/{fold}/lncrnabert/datasets/lncrnabert_valid_pc.fa",
        lnc_valid="results/{expt}/training/{fold}/lncrnabert/datasets/lncrnabert_valid_lnc.fa",
    output:
        results=directory("results/{expt}/training/{fold}/lncrnabert/{encoding}/results"),
        model=directory("results/{expt}/training/{fold}/lncrnabert/{encoding}/models"),
    log:
        "logs/{expt}/training/{fold}/lncrnabert/train_{fold}_{encoding}.log",
    benchmark:
        "benchmarks/{expt}/training/{fold}/lncrnabert/train_{fold}_{encoding}.txt",
    resources:
        mem_mb=25000,
        slurm_partition="gpu",
        gres=gpu_helpers.get_gpu_for_rule_name("lncrnabert_train"),
        cpus_per_gpu=1,
    conda:
        "lncrnabert"
    params:
        pretrained_model=get_lncrnabert_pretrained_model,
        input_dir=lambda wc, input: subpath(input[0], parent=True),
        gpu=0,
        encoding_method=lambda wc: wc.encoding,
        extra=lambda wc: "--k 3" if wc.encoding == "kmer" else "",  # TODO: Make configurable
        shell_message=lambda wc: f"Model: {get_lncrnabert_pretrained_model(wc)}"
    shell:
        """
        {{
            mkdir -p $(dirname {log})
            cd software/lncRNA-Py
            pwd
            export CUDA_VISIBLE_DEVICES={params.gpu}
            echo "Starting lncRNA-BERT training"
            echo "GPU Info:"
            echo "Available physical GPUs:"
            nvidia-smi --query-gpu=index,platform.module_id,name,driver_version,memory.total,compute_cap,mig.mode.current --format=csv
            echo "Available MIG devices:"
            nvidia-smi -L | grep -i mig || echo "No MIG devices found"
            echo "Using visible GPU Device with index $CUDA_VISIBLE_DEVICES"
            echo "Model: {params.pretrained_model}"
            echo "Input FASTA:"
            echo "- {input.pc_train}"
            echo "- {input.lnc_train}"
            echo "- {input.pc_valid}"
            echo "- {input.lnc_valid}"
            echo "Output directories:"
            echo "- Results: {output.results}"
            echo "- Model: {output.model}"
            echo "Encoding method: {params.encoding_method}"
            echo "Extra params: {params.extra}"
            echo "---"

            python -m lncrnapy.scripts.train \
            ../../{input.pc_train} \
            ../../{input.lnc_train} \
            ../../{input.pc_valid} \
            ../../{input.lnc_valid} \
            --pretrained_model {params.pretrained_model} \
            --results_dir ../../{output.results} \
            --model_dir ../../{output.model} \
            --encoding_method {params.encoding_method} \
            {params.extra}
        }} 2>&1 | tee {log}
        """


rule run_lncrnabert_train_scratch:
    input:
        expand(
            "results/{expt}/training/lncrnabert/scratch/{encoding}/models",
            expt="gencode.v47.common.cdhit",
            encoding=["cse", "kmer"],
        ),


use rule lncrnabert_train as lncrnabert_train_scratch with:
    output:
        results=directory("results/{expt}/training/{fold}/lncrnabert/scratch/{encoding}/results"),
        model=directory("results/{expt}/training/{fold}/lncrnabert/scratch/{encoding}/models"),
    log:
        "logs/{expt}/training/{fold}/lncrnabert/train_scratch_{encoding}_{fold}.log",
    benchmark:
        "benchmarks/{expt}/training/{fold}/lncrnabert/train_scratch_{encoding}_{fold}.txt"
    params:
        shell_message= "Model: finetuning from scratch"


#########################
# INFERENCE RULES
#########################

def lncrnabert_model_is_file(wildcards):
    model_path = config["datasets"][wildcards.db]["models"].get("lncrnabert_model", "")
    return model_path != "" and not model_path.startswith("luukromeijn/")


def get_lncrnabert_encoding(wildcards):
    model_path = get_lncrnabert_model(wildcards, return_default=True)
    if "kmer" in model_path:
        return "kmer"
    else:
        return "cse"


def get_lncrnabert_model(wildcards, return_default=False):
    wc = wildcards
    lncrnabert_config = config["experiments"][wc.expt]["models"].get("lncrnabert", "default")
    if lncrnabert_config == "custom":
        model_dir = f"results/{wc.expt}/training/{wc.fold}/lncrnabert/{wc.encoding}/models"
        model_list = [f for f in os.listdir(model_dir) if f.startswith("CLS")]
        if model_list == []:
            raise ValueError(f"No model files found in {model_dir}")
        elif len(model_list) > 1:
            raise ValueError(f"Multiple model files found in {model_dir}: {model_list}")
        else:
            return os.path.join(model_dir, model_list[0])
    else:
        return "luukromeijn/lncRNA-BERT-kmer-k3-finetuned" if return_default else ""


rule lncrnabert_test:
    input:
        model=get_lncrnabert_model,
        fasta="results/{expt}/datasets/{fold}/test_all.fa",
    output:
        "results/{expt}/testing/{fold}/lncrnabert/{encoding}/classification.csv",
    log:
        "logs/{expt}/testing/{fold}/lncrnabert/test_{encoding}.log",
    benchmark:
        "benchmarks/{expt}/testing/{fold}/lncrnabert/test_{encoding}.txt"
    resources:
        mem_mb=10000,
        slurm_partition="gpu",
        gres="gpu:nvidia_h100_nvl_1g.12gb:1",
        cpus_per_gpu=2,
    threads: 2
    conda:
        "lncrnabert"
    params:
        model=lambda wc: get_lncrnabert_model(wc, return_default=True),
        out_dir=lambda wc, output: subpath(output[0], parent=True),
        gpu=0,
        encoding_method=lambda wc: get_lncrnabert_encoding(wc),
        extra="",
    shell:
        """
        mkdir -p {params.out_dir}

        {{
            echo "Starting lncRNA-BERT classification"
            echo "GPU Info:"
            echo "Available physical GPUs:"
            nvidia-smi --query-gpu=index,platform.module_id,name,driver_version,memory.total,compute_cap,mig.mode.current --format=csv
            echo "Available MIG devices:"
            nvidia-smi -L | grep -i mig || echo "No MIG devices found"
            echo "Using visible GPU Device with index $CUDA_VISIBLE_DEVICES"
            echo "Model: {params.model}"
            echo "Input FASTA: {input.fasta}"
            echo "Output directory: {params.out_dir}"
            echo "Encoding method: {params.encoding_method}"
            echo "---"

            export CUDA_VISIBLE_DEVICES={params.gpu}
            cd software/lncRNA-Py/
            python -m lncrnapy.scripts.classify \
            ../../{input.fasta} \
            --results_dir ../../{params.out_dir} \
            --model_file ../../{params.model} \
            --encoding_method {params.encoding_method}
        }} 2>&1 | tee {log}
        """


use rule lncrnabert_test as lncrnabert_test_kmermodel with:
    input:
        model="results/{expt}/training/{fold}/lncrnabert_kmer/models/CLS_kmer_finetuned_k3_dm768_N12_dff3072_h12_bs8_lr1e-05_wd0_cl768_d0/",
        fasta="resources/gencode.v47.transcripts.fa",
    output:
        "results/{expt}/training/{fold}/lncrnabert/kmer/classification.csv",
    log:
        "logs/{expt}/training/{fold}/lncrnabert_test_kmer_{expt}.log",
    params:
        encoding_method="kmer",
        gpu=0
    benchmark:
        "benchmarks/{expt}/training/{fold}/{expt}.inference.lncrnabert.kmer"


rule lncrnabert_test_pretrained_model:
    input:
        fasta="resources/gencode.v47.transcripts.fa",
    output:
        "results/{expt}/testing/{fold}/lncrnabert/pretrained/classification.csv",
    log:
        "logs/{expt}/testing/{fold}/lncrnabert_test_{expt}_pretrained.log",
    benchmark:
        "benchmarks/{expt}/testing/{fold}/{expt}.inference.lncrnabert.pretrained"
    resources:
        mem_mb=25000,
        slurm_partition="gpu",
        gres="gpu:nvidia_h100_nvl_1g.24gb:1",
        cpus_per_gpu=1,
    conda:
        "lncrnabert"
    params:
        out_dir=lambda wc, output: subpath(output[0], parent=True),
        gpu=gpu_helpers.get_gpu_for_rule_name("lncrnabert_test_pretrained_model"),
        encoding_method=lambda wc: wc.encoding,
        extra="",
        model_file=get_lncrnabert_pretrained_model,
    shell:
        """
        mkdir -p {params.out_dir}

        {{
            echo "Starting lncRNA-BERT classification"
            echo "GPU Info:"
            echo "Available physical GPUs:"
            nvidia-smi --query-gpu=index,platform.module_id,name,driver_version,memory.total,compute_cap,mig.mode.current --format=csv
            echo "Available MIG devices:"
            nvidia-smi -L | grep -i mig || echo "No MIG devices found"
            echo "Using visible GPU Device with index $CUDA_VISIBLE_DEVICES"
            echo "Model: {params.model_file}"
            echo "Input FASTA: {input.fasta}"
            echo "Output directory: {params.out_dir}"
            echo "Encoding method: {params.encoding_method}"
            echo "---"

            export CUDA_VISIBLE_DEVICES={params.gpu}
            cd software/lncRNA-Py/
            python -m lncrnapy.scripts.classify \
            ../../{input.fasta} \
            --results_dir ../../{params.out_dir} \
            --model_file {params.model_file} \
            --encoding_method {params.encoding_method}
        }} 2>&1 | tee {log}
        """


use rule lncrnabert_test_pretrained_model as lncrnabert_test_finetuned with:
    output:
        "results/{expt}/testing/{fold}/lncrnabert/finetuned/classification.csv",
    params:
        model_file="luukromeijn/lncRNA-BERT-CSE-k9-finetuned"
    log:
        "logs/{expt}/testing/{fold}/lncrnabert_test_{expt}_finetuned.log",
    benchmark:
        "benchmarks/{expt}/testing/{fold}/{expt}.inference.lncrnabert.finetuned"


use rule lncrnabert_test_pretrained_model as lncrnabert_test_published with:
    output:
        "results/{expt}/testing/{fold}/lncrnabert/published/classification.csv",
    params:
        model_file="luukromeijn/lncRNA-BERT-kmer-k3-finetuned",
        encoding_method="kmer"
    log:
        "logs/{expt}/testing/{fold}/lncrnabert_test_{expt}_published.log",
    benchmark:
        "benchmarks/{expt}/testing/{fold}/{expt}.inference.lncrnabert.published"
