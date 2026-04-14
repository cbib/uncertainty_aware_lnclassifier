#########################
# CROSS-VALIDATION RULES
#########################

rule feelnc_codpot_cv:
    input:
        fasta="results/{expt}/datasets/folds/{fold}/all_test.fa",
        mrna="results/{expt}/datasets/folds/{fold}/pc_train.fa",
        lncrna="results/{expt}/datasets/folds/{fold}/lnc_train.fa"
    output:
        "results/{expt}/testing/{fold}/FEELnc/{expt}_fold{fold}_RF.txt"
    threads: 20
    params:
        outdir=lambda wc, output: os.path.dirname(output[0]),
        outname=lambda wc: f"{wc.expt}_fold{wc.fold}",
        extra=""
    conda:
        "workflow/envs/feelnc_env.yaml"
    benchmark:
        "benchmarks/{expt}/testing/{fold}/feelnc_codpot.txt"
    log:
        "logs/{expt}/testing/{fold}/feelnc_codpot.log"
    shell:
        """
        FEELnc_codpot.pl \
            -i {input.fasta} \
            -a {input.mrna} \
            -l {input.lncrna} \
            -p {threads} \
            --kmer=1,2,3,6,9,12 \
            --learnorftype=4 \
            --testorftype=4 \
            --outdir={params.outdir} \
            --outname={params.outname} \
            {params.extra} > {log} 2>&1
        """


rule cpat_cv:
    input:
        "results/{expt}/datasets/folds/{fold}/all_test.fa"
    output:
        orfs="results/{expt}/testing/{fold}/CPAT/{expt}_fold{fold}.cpat.{mode}.ORF_seqs.fa",
        best_orfs="results/{expt}/testing/{fold}/CPAT/{expt}_fold{fold}.cpat.{mode}.ORF_prob.best.tsv"
    conda:
        "workflow/envs/cpat_env.yaml"
    params:
        x="software/cpat-3.0.5/prebuilt_models/Human_Hexamer.tsv",  # TODO: Use trained model
        d="software/cpat-3.0.5/prebuilt_models/Human_logitModel.RData",  # TODO: Use trained model
        best_orf=lambda wc: wc.mode,
        min_orf=9,
        out_prefix=lambda wc, output: output.orfs.rsplit(".", 2)[0]
    benchmark:
        "benchmarks/{expt}/testing/{fold}/cpat.{mode}.txt"
    log:
        "logs/{expt}/testing/{fold}/cpat_{mode}.log"
    shell:
        """
        cpat \
        -x {params.x} \
        -d {params.d} \
        -g {input} \
        -o {params.out_prefix} \
        --best-orf={params.best_orf} \
        --min-orf={params.min_orf} > {log} 2>&1
        """


rule lncDC_cv:
    input:
        "results/{expt}/datasets/folds/{fold}/all_test.fa"
    output:
        "results/{expt}/testing/{fold}/lncDC/{expt}_fold{fold}.lncDC.{mode}.csv"
    conda:
        "workflow/envs/lncdc_env.yaml"
    params:
        use_custom_training="False",  # TODO: Support custom training
    threads: 20
    resources:
        mem_mb_per_cpu=4000,
        runtime="7d"
    benchmark:
        "benchmarks/{expt}/testing/{fold}/lncDC.{mode}.txt"
    log:
        "logs/{expt}/testing/{fold}/lncDC_{mode}.log"
    shell:
        """
        set -e
        set -x

        exec 2>&1
        exec >> {log}

        echo "=== lncDC CV Inference ==="
        echo "Date: $(date)"
        echo "Input fasta: {input}"
        echo "Output file: {output}"
        echo "Mode: {wildcards.mode}"
        echo "Fold: {wildcards.fold}"
        echo "Threads: {threads}"

        CUSTOM_ARGS=""
        if [ "{wildcards.mode}" = "ss" ]; then
            CUSTOM_ARGS="-ss"
        fi

        lncRNA_detection.py -f {input} -o {output} -t {threads} $CUSTOM_ARGS
        """


rule mRNN_cv:
    input:
        fasta="results/{expt}/datasets/folds/{fold}/all_test.fa",
        model="results/{expt}/training/{fold}/mRNN/trained/best_models/"
    output:
        "results/{expt}/testing/{fold}/mRNN/{expt}_fold{fold}.mRNN.multi.tsv"
    log:
        "logs/{expt}/testing/{fold}/mRNN.log"
    benchmark:
        "benchmarks/{expt}/testing/{fold}/mRNN.txt"
    shell:
        """
        echo "mRNN CV inference for {wildcards.expt} fold {wildcards.fold}" > {log} 2>&1
        # TODO: Implement mRNN inference
        touch {output}
        """


#########################
# NON-CV RULES
#########################


rule feelnc_codpot:
    input:
        fasta=lambda wc: config["datasets"][wc.dset]["test"],
        mrna=lambda wc: config["datasets"][wc.dset]["train_pc"],
        lncrna=lambda wc: config["datasets"][wc.dset]["train_lnc"]
    output:
        "results/{expt}/testing/{dset}/FEELnc/{dset}_RF.txt"
        # TODO: add more outputs
    threads: 20
    params:
        outdir=lambda wc, output: os.path.dirname(output[0]),
        extra=""
    conda:
        "workflow/envs/feelnc_env.yaml"
    benchmark:
        "benchmarks/{expt}/testing/{dset}/feelnc_codpot.txt"
    log:
        "logs/{expt}/testing/{dset}/feelnc_codpot.log"
    shell:
        """
        FEELnc_codpot.pl \
            -i {input.fasta} \
            -a {input.mrna} \
            -l {input.lncrna} \
            -p {threads} \
            --kmer=1,2,3,6,9,12 \
            --learnorftype=4 \
            --testorftype=4 \
            --outdir={params.outdir} \
            --outname={wildcards.dset} \
            {params.extra} > {log} 2>&1
        """

# After feelnc_codpot rule, the pipeline runs for about 2h with 40 cores (20 per dataset)

rule cpat:
    input:
        lambda wc: config["datasets"][wc.dset]["test"]
    output:
        orfs="results/{expt}/testing/{dset}/CPAT/{dset}.cpat.{mode}.ORF_seqs.fa",
        best_orfs="results/{expt}/testing/{dset}/CPAT/{dset}.cpat.{mode}.ORF_prob.best.tsv"
    conda:
        "workflow/envs/cpat_env.yaml"
    params:
        x=lambda wc:config["datasets"][wc.dset]["models"]["cpat_hexamer"], # Hexamer table of the species
        d=lambda wc:config["datasets"][wc.dset]["models"]["cpat_logit"], # Logistic regression model of the species
        best_orf=lambda wc: wc.mode,  # Defines how the best ORF is selected: 'p' for coding potential, 'l' for length
        min_orf=9,
        out_prefix=lambda wc, output: output.orfs.rsplit(".", 2)[0]
    benchmark:
        "benchmarks/{expt}/testing/{dset}/cpat.{mode}.txt"
    log:
        "logs/{expt}/testing/{dset}/cpat_{mode}.log"
    shell:
        """
        cpat \
        -x {params.x} \
        -d {params.d} \
        -g {input} \
        -o {params.out_prefix} \
        --best-orf={params.best_orf} \
        --min-orf={params.min_orf} > {log} 2>&1
        """

def lncDC_custom_params(wildcards):
    use_custom_training = config["datasets"][wildcards.dset]["models"]["lncDC_use_custom"]
    if use_custom_training:
        custom_dir = config["datasets"][wildcards.dset]["models"]["lncDC_custom_dir"]
        mode_suffix = "_ss" if wildcards.mode == "ss" else ""
        return {
            "use_custom_training": True,
            "hexamer": f"{custom_dir}{mode_suffix}/train_hexamer_table.csv",
            "model": f"{custom_dir}{mode_suffix}/train_xgb_model_SIF_PF{'_SSF' if wildcards.mode == 'ss' else ''}.pkl",
            "imputer": f"{custom_dir}{mode_suffix}/train_imputer_SIF_PF{'_SSF' if wildcards.mode == 'ss' else ''}.pkl",
            "scaler": f"{custom_dir}{mode_suffix}/train_scaler_SIF_PF{'_SSF' if wildcards.mode == 'ss' else ''}.pkl",
            "ss_kmer_prefix": f"{custom_dir}_ss/train_ss_table" if wildcards.mode == "ss" else None
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

rule lncDC:
    input:
        lambda wc: config["datasets"][wc.dset]["test"]
    output:
        "results/{expt}/testing/{dset}/lncDC/{dset}.lncDC.{mode}.csv"
    conda:
        "workflow/envs/lncdc_env.yaml"
    params:
        use_custom_training=lambda wc: lncDC_custom_params(wc)["use_custom_training"],
        hexamer=lambda wc: lncDC_custom_params(wc)["hexamer"],
        model=lambda wc: lncDC_custom_params(wc)["model"],
        imputer=lambda wc: lncDC_custom_params(wc)["imputer"],
        scaler=lambda wc: lncDC_custom_params(wc)["scaler"],
        ss_kmer_prefix=lambda wc: lncDC_custom_params(wc)["ss_kmer_prefix"],
    threads: 20
    resources:
        mem_mb_per_cpu=4000,
        runtime="7d"
    benchmark:
        "benchmarks/{expt}/testing/{dset}/lncDC.{mode}.txt"
    log:
        "logs/{expt}/testing/{dset}/lncDC_{mode}.log"
    shell:
        """
        set -e  # Exit on any error
        set -x  # Print commands as they execute

        exec 2>&1  # Redirect stderr to stdout
        exec >> {log}  # Redirect all output to log

        echo "=== lncDC Inference ==="
        echo "Date: $(date)"
        echo "Python version: $(python --version)"
        echo "Working directory: $PWD"
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

        # Build optional -r flag based on mode
        SS_FLAG=""
        if [ "{wildcards.mode}" = "ss" ]; then
            SS_FLAG="-r"
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


def get_mrnn_input_weights(wildcards):
    # Check config
    custom_training = config["datasets"][wildcards.dset]["models"]["mrnn_use_custom"]
    if custom_training:
        return [os.path.join(f"results/{wildcards.dset}/training/mRNN/trained/best_models", f) for f in os.listdir(f"results/{wildcards.dset}/training/mRNN/trained/best_models") if f.startswith("mRNN.model")]
    else:
        return [f"software/mRNN/weights/{w}.pkl" for w in ["w10u5", "w14u3", "w16u5", "w18u1b", "w23u2"]]


rule mrnn:
    # We use the ensemble model, i.e. the combination of different models into one.
    input:
        fasta=lambda wc: config["datasets"][wc.dset]["test"],
        weights=lambda wc: get_mrnn_input_weights(wc)
    output:
        "results/{expt}/testing/{dset}/mRNN/{dset}.mRNN.multi.tsv"
    conda:
        "workflow/envs/mrnn_env.yaml"
    params:
        output_dir=lambda wc, output: subpath(output[0], parent=True),
        weights=lambda wc, input: ",".join(input.weights)
    resources:
        mem_mb=10000
    benchmark:
        "benchmarks/{expt}/testing/{dset}/mrnn_inference.txt"
    log:
        "logs/{expt}/testing/{dset}/mrnn_inference.log"
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

        echo "Weights used: {params.weights}" > results/{wildcards.dset}/mRNN/weights_used.txt
        echo "Weights used: {params.weights}" >> {log}
        """
