import os

def lncDC_custom_params(wildcards):
    use_custom_training = config["experiments"][wildcards.expt]["models"].get("lncDC", "default")
    if use_custom_training  == "custom":
        custom_dir = f"results/{wildcards.expt}/training/{wildcards.fold}/lncDC/"
        mode_suffix = "_ss" if wildcards.mode == "ss" else ""
        feature_suffix = "_SSF" if wildcards.mode == "ss" else ""
        return {
            "use_custom_training": True,
            "hexamer": os.path.join(custom_dir, mode_suffix, "train_hexamer_table.csv"),
            "model": os.path.join(custom_dir, mode_suffix, f"train_xgb_model_SIF_PF{feature_suffix}.pkl"),
            "imputer": os.path.join(custom_dir, mode_suffix, f"train_imputer_SIF_PF{feature_suffix}.pkl"),
            "scaler": os.path.join(custom_dir, mode_suffix, f"train_scaler_SIF_PF{feature_suffix}.pkl"),
            "ss_kmer_prefix": os.path.join(custom_dir, "_ss", "train_ss_table") if wildcards.mode == "ss" else None
        }
    else:
        return {
            "use_custom_training": False,
            "hexamer": None,
            "model": None,
            "imputer": None,
            "scaler": None,
            "ss_kmer_prefix": None
        }


rule lncDC_cv:
    input:
        branch(
            condition=lambda wc: wc.mode == "ss",
            then="results/{expt}/datasets/{fold}/test_all.fa.ss",
            otherwise="results/{expt}/datasets/{fold}/test_all.fa"
        )
    output:
        "results/{expt}/testing/{fold}/lncDC/{fold}.lncDC.{mode}.csv"
    log:
        "logs/{expt}/testing/{fold}/lncDC/{fold}.lncDC.{mode}.log"
    benchmark:
        "benchmarks/{expt}/testing/{fold}/lncDC/{fold}.lncDC.{mode}.txt"
    conda:
        "../envs/lncdc_env.yaml"
    params:
        use_custom_training=lambda wc: lncDC_custom_params(wc)["use_custom_training"],
        hexamer=lambda wc: lncDC_custom_params(wc)["hexamer"],
        model=lambda wc: lncDC_custom_params(wc)["model"],
        imputer=lambda wc: lncDC_custom_params(wc)["imputer"],
        scaler=lambda wc: lncDC_custom_params(wc)["scaler"],
        ss_kmer_prefix=lambda wc: lncDC_custom_params(wc)["ss_kmer_prefix"],
    threads: 10
    resources:
        mem_mb=8000,
        runtime="1h"
    shell:
       """
        set -e  # Exit on any error
        set -x  # Print commands as they execute

        #exec 2>&1  # Redirect stderr to stdout
        exec >> {log}  # Redirect all output to log

        echo "=== lncDC Inference ==="
        echo "Date: $(date)"
        echo "Python version: $(python --version)"
        echo "Working directory: $PWD"
        echo "Fold: {wildcards.fold}"
        echo "Input fasta: {input}"
        echo "Output file: {output}"
        echo "Mode: {wildcards.mode}"
        echo "Use custom training: {params.use_custom_training}"
        echo "Threads: {threads}"

        if [ "{params.use_custom_training}" = "True" ]; then
            echo "Using custom-trained model"
            echo "Custom hexamer: {params.hexamer}"
            echo "Custom model: {params.model}"
            echo "Custom imputer: {params.imputer}"
            echo "Custom scaler: {params.scaler}"
            CUSTOM_ARGS="-x {params.hexamer} -m {params.model} -p {params.imputer} -s {params.scaler}"
            if [ "{wildcards.mode}" = "ss" ]; then
            echo "Custom SS kmer prefix: {params.ss_kmer_prefix}"
            CUSTOM_ARGS="$CUSTOM_ARGS -k {params.ss_kmer_prefix}"
            fi
        else
            echo "Using default pre-trained model"
            CUSTOM_ARGS=""
        fi

        # Build optional -r --ss_file flag based on mode
        SS_FLAG=""
        if [ "{wildcards.mode}" = "ss" ]; then
            SS_FLAG="-r --ss_file"
        fi

        echo "Running lncDC with args: $SS_FLAG $CUSTOM_ARGS"

        python -u software/LncDC-1.3.6/bin/lncDC.py \
        -i {input} \
        -o {output} \
        $SS_FLAG \
        $CUSTOM_ARGS

        echo "lncDC completed successfully"

        # No need to specify the -t argument. The default (-1) takes all available threads.
        # These are limited by the thread parameter in the rule.
        """
