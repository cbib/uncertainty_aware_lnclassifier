def get_mrnn_input_weights(wildcards):
    wc = wildcards
    mrnn_config = config["experiments"][wc.expt]["models"].get("mrnn", "default")
    if mrnn_config == "custom":
        model_dir  = f"results/{wc.expt}/training/{wc.fold}/mRNN/trained/best_models"
        return [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith("mRNN.model")]
    else:
        return [f"software/mRNN/weights/{w}.pkl" for w in ["w10u5", "w14u3", "w16u5", "w18u1b", "w23u2"]]


rule mRNN_cv:
    input:
        fasta="results/{expt}/datasets/{fold}/test_all.fa",
        weights=get_mrnn_input_weights
    output:
        "results/{expt}/testing/{fold}/mRNN/{fold}.mRNN.multi.tsv"
    log:
        "logs/{expt}/testing/{fold}/mRNN.log"
    benchmark:
        "benchmarks/{expt}/testing/{fold}/mRNN.txt"
    conda:
        "../envs/mrnn_env.yaml"
    resources:
        mem_mb=10000
    params:
        output_dir=lambda wc, output: subpath(output[0], parent=True),
        weights=lambda wc, input: ",".join(input.weights)
    shell:
        """
        set -e  # Exit on any error
        set -x  # Print commands as they execute

        exec 2>&1  # Redirect stderr to stdout
        exec >> {log}  # Redirect all output to log

        echo "=== mRNN Ensemble Inference ==="
        echo "Date: $(date)"
        echo "Python version: $(python --version)"
        echo "Working directory: $PWD"
        echo "Input fasta: {input.fasta}"
        echo "Output file: {output}"
        echo "Weights: {params.weights}"

        export THEANO_FLAGS="device=cpu,floatX=float32,base_compiledir=/tmp/theano_$RANDOM"
        mkdir -p {params.output_dir}
        python -u software/mRNN/mRNN_ensemble.py \
        -w {params.weights} \
        -o {output} \
        {input.fasta}

        echo "Weights used: {params.weights}" > results/{wildcards.expt}/testing/{wildcards.fold}/mRNN/weights_used.txt
        echo "Weights used: {params.weights}" >> {log}
        """
