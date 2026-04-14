import os

include: "rnasamba.smk"
include: "mrnn_train.smk"
include: "lncrnabert.smk"

"""
rule only_mrnn:
    input:
        expand("results/{expt}/training/{fold}/mRNN/trained/best_models", expt=config['to_train'])


rule only_feelnc:
    input:
        expand("results/{expt}/training/{fold}/FEELnc/{dset}.test_pc_and_lnc.fa_RF.txt", expt=config['to_train'])


rule only_lncfinder:
    input:
        expand("results/{expt}/training/{fold}/lncfinder/{dset}_{ss}.RData",dset="gencode.v49.cdhit", ss="no-ss")


rule only_plncpro:
    input:
        expand("results/{expt}/training/{fold}/plncpro/{dset}.model", expt=config['to_train'])


rule only_cpat:
    input:
        expand("results/{expt}/training/{fold}/cpat/{dset}{ext}", expt=config['to_train'], ext=[".feature.xls", ".logit.RData", ".make_logitModel.r"])


rule test_cpat:
    input:
        expand("results/{expt}/training/{fold}/cpat/{dset}.logit.RData", dset=config['to_train'])


rule all_but_mrnn:
    input:
        expand("results/{expt}/training/{fold}/cpat/{dset}{ext}", expt=config['to_train'], ext=[".feature.xls", ".logit.RData", ".make_logitModel.r"]),
        expand("results/{expt}/training/{fold}/lncDC", expt=config['to_train']),
        expand("results/{expt}/training/{fold}/lncfinder/{dset}.RData", expt=config['to_train']),
        expand("results/{expt}/training/{fold}/FEELnc/{dset}.test_pc_and_lnc.fa_RF.txt", expt=config['to_train']),
        expand("results/{expt}/training/{fold}/plncpro/{dset}.model", expt=config['to_train']),


rule all:
    input:
        #expand("results/{expt}/training/{fold}/train_pc.cds.fa",dset="gencode.v47.common.cdhit"),
        #expand("results/{expt}/training/{fold}/cpat/{dset}{ext}",dset="gencode.v47.common.cdhit", ext=[".feature.xls", ".logit.RData", ".make_logitModel.r"]),
        #expand("results/{expt}/training/{fold}/lncDC",dset="gencode.v47.common.cdhit"),
        #expand("results/{expt}/training/{fold}/plncpro/{dset}.model",dset="gencode.v47.common.cdhit"),
        #expand("results/{expt}/training/{fold}/lncfinder/{dset}.RData",dset="gencode.v47.common.cdhit"),
        #expand("results/{expt}/training/{fold}/FEELnc/{dset}.test_pc_and_lnc.fa_RF.txt",dset="gencode.v47.common.cdhit"),
        #expand("results/{expt}/training/{fold}/mRNN/trained/best_models", expt=config['training']),
        expand("results/{expt}/training/{fold}/train_pc.cds.fa", expt=config['training']),
        expand("results/{expt}/training/{fold}/cpat/{dset}{ext}", expt=config['training'], ext=[".feature.xls", ".logit.RData", ".make_logitModel.r"]),
        expand("results/{expt}/training/{fold}/lncDC", expt=config['training']),
        expand("results/{expt}/training/{fold}/lncfinder/{dset}.RData", expt=config['training']),
        expand("results/{expt}/training/{fold}/FEELnc/{dset}.test_pc_and_lnc.fa_RF.txt", expt=config['training']),
        expand("results/{expt}/training/{fold}/mRNN/trained/best_models", expt=config['training']),
        #expand("results/{expt}/training/{fold}/plncpro/{dset}.model",dset="gencode.v47.cdhit"),

"""

rule get_cds_from_dataset:
    """
    Extract CDS sequences of a list of transcripts, using as reference the database fasta file.
    """
    input:
        transcripts="results/{expt}/datasets/{fold}/train_pc.fa",
        cds=lambda wc: config["experiments"][wc.expt]["reference_cds"]
    output:
        "results/{expt}/datasets/{fold}/train_pc.cds.fa"
    log:
        "logs/{expt}/datasets/get_cds_from_dataset_{fold}.log"
    conda:
        "lnc-datasets"
    script:
        "../scripts/get_cds_from_ids.py"


rule cpat_make_hexamer_tab:
    input:
        cds= "results/{expt}/datasets/{fold}/train_pc.cds.fa",
        nc="results/{expt}/datasets/{fold}/train_lnc.fa"
    output:
        "results/{expt}/training/{fold}/cpat/{fold}_Hexamer.tsv"
    log:
        "logs/{expt}/training/{fold}/cpat/make_hexamer_tab_{fold}.log"
    benchmark:
        "benchmarks/{expt}/training/{fold}/cpat/make_hexamer_tab_{fold}.txt"
    params:
        out_dir=lambda wc, output: subpath(output[0], parent=True)
    conda:
        "../envs/cpat_env.yaml"
    shell:
        """
        mkdir -p {params.out_dir}
        make_hexamer_tab \
        -c {input.cds} \
        -n {input.nc} > {output} 2> {log}
        """

def get_models_config(wildcards):
    """Helper function to get the models configuration for a given experiment and fold."""
    return config["experiments"][wildcards.expt]["models"]

def cpat_hexamer(wildcards):
    wc = wildcards
    """Helper function to get the path to the CPAT hexamer file for a given experiment and fold."""
    cpat_config = get_models_config(wc)["cpat"]
    if cpat_config == "custom":
        return f"results/{wc.expt}/training/{wc.fold}/cpat/{wc.fold}_Hexamer.tsv"
    else:
        return "software/cpat-3.0.5/prebuilt_models/Human_Hexamer.tsv"

rule cpat_train:
    input:
        pc="results/{expt}/datasets/{fold}/train_pc.fa",
        nc="results/{expt}/datasets/{fold}/train_lnc.fa",
        hexamer_file=cpat_hexamer
    output:
        multiext("results/{expt}/training/{fold}/cpat/{fold}", ".feature.xls", ".logit.RData", ".make_logitModel.r")
    log:
        "logs/{expt}/training/{fold}/cpat/cpat_train_{fold}.log"
    benchmark:
        "benchmarks/{expt}/training/{fold}/cpat/cpat_train_{fold}.txt"
    params:
        # NOTE: CPAT adds .<suffix> to the output prefix. Set path and file prefix
        out_prefix=lambda wc, output: os.path.join(subpath(output[0], parent=True), wc.fold),
    conda:
        "../envs/cpat_env.yaml"
    shell:
        """
        mkdir -p {params.out_prefix}
        make_logitModel \
        -x {input.hexamer_file} \
        -c {input.pc} \
        -n {input.nc} \
        -o {params.out_prefix} \
        > {log} 2>&1
        """


rule lncDC_train:
    input:
        pc="results/{expt}/datasets/{fold}/train_pc.fa",
        cds="results/{expt}/datasets/{fold}/train_pc.cds.fa",
        lnc="results/{expt}/datasets/{fold}/train_lnc.fa"
    output:
        directory("results/{expt}/training/{fold}/lncDC")
    log:
        "logs/{expt}/training/{fold}/lncDC/train_{fold}.log"
    benchmark:
        "benchmarks/{expt}/training/{fold}/lncDC/train_{fold}.txt"
    conda:
        "../envs/lncdc_env.yaml"
    threads: 20
    resources:
        mem_mb=10000
    params:
        out_prefix="train",
        extra="" # add -r if training with ss features
    shell:
        """
        python software/LncDC-1.3.6/bin/lncDC-train.py \
        -m {input.pc} \
        -c {input.cds} \
        -l {input.lnc} \
        -o {output}/{params.out_prefix} \
        -t {threads} \
        {params.extra} \
        > {log} 2>&1
        """

rule filter_cds_with_ss:
    """
    Helper rule to generate files containing only the CDS of associated transcripts
    that have secondary structure information. This avoids downstream issues with
    mismatched sequences (e.g., lncDC does not seem to keep the transcript IDs)
    """
    input:
        cds="results/{expt}/datasets/{fold}/train_pc.cds.fa",
        ss="results/{expt}/datasets/{fold}/train_pc.fa.ss"
    output:
        "results/{expt}/datasets/{fold}/train_pc.cds.fa.with_ss"
    log:
        "logs/{expt}/datasets/{fold}/filter_cds_with_ss_{fold}.log"
    conda:
        "lnc-datasets"
    script:
        "../scripts/filter_cds_with_ss.py"


use rule lncDC_train as lncDC_train_with_precomputed_ss with:
    """
    This rule trains LncDC including secondary structure (SS) features.
    Instead of relying on SS being calculated within lncDC,
    we externalize SS calculation to the dataset preparation step.
    """
    input:
        pc="results/{expt}/datasets/{fold}/train_pc.fa.ss",
        cds=rules.filter_cds_with_ss.output,
        lnc="results/{expt}/datasets/{fold}/train_lnc.fa.ss"
    output:
        directory("results/{expt}/training/{fold}/lncDC_ss")
    log:
        "logs/{expt}/training/{fold}/lncDC/train_{fold}_ss.log"
    params:
        extra="-r -ss-file"


rule diamond_makedb_train:
    input:
        fname = "resources/blast_dbs/uniprot_sprot.fasta",
    output:
        fname = "resources/blast_dbs/uniprot_sprot.db.dmnd"
    log:
        "logs/diamond_makedb/uniprot_sprot.log"
    params:
        extra=""
    threads: 8
    wrapper:
        "v7.4.0/bio/diamond/makedb"


# Plncpro train produces temporary files in the basedir of the output directory
# e.g. resuts/{dset}/training insteahd of resuts/{dset}/training/plncpro
# Also, it is supposed to move the output files from the cwd to the output directory,
# but it creates the final model file in the wrong place, so it crashes.
# https://github.com/urmi-21/PLncPRO/blob/9270d15b768b294f099b0bdc70fa059a1c646dfc/plncpro/build.py#L404
# https://github.com/urmi-21/PLncPRO/blob/9270d15b768b294f099b0bdc70fa059a1c646dfc/plncpro/build.py#L363-L365
# https://github.com/urmi-21/PLncPRO/blob/9270d15b768b294f099b0bdc70fa059a1c646dfc/plncpro/bin/rf/buildmodel.py#L48-L50
rule plncpro_train:
    input:
        pc="results/{expt}/datasets/{fold}/train_pc.fa",
        lnc="results/{expt}/datasets/{fold}/train_lnc.fa",
        blast_db="resources/blast_dbs/uniprot"
    output:
        "results/{expt}/training/{fold}/plncpro/{fold}.model"
    log:
        "logs/{expt}/training/plncpro/train_{fold}.log"
    benchmark:
        "benchmarks/{expt}/training/{fold}/plncpro/train_{fold}.txt"
    conda:
        "../envs/plncpro_env.yaml"
    params:
        blast_db="resources/blast_dbs/uniprot/uniprotdb",
        output_dir=lambda wc, output: subpath(output[0], parent=True),
        model_name=lambda wc: f"{wc.fold}.model",
        extra="-v"
    threads: 20
    resources:
        mem_mb_per_cpu=12000
    shell:
        """
        mkdir -p {params.output_dir}
        plncpro build \
        -p {input.pc} \
        -n {input.lnc} \
        -o {params.output_dir} \
        -m {params.model_name} \
        -d {params.blast_db} \
        -t {threads} \
        {params.extra} > {log} 2>&1
        """


def get_lncfinder_train_inputs(wildcards):
    # Default, compulsory inputs
    inputs = {
        "pc": "results/{expt}/datasets/{fold}/train_pc.fa",
        "cds": "results/{expt}/datasets/{fold}/train_pc.cds.fa",
        "lnc": "results/{expt}/datasets/{fold}/train_lnc.fa"
    }
    wc = wildcards
    # Add SS files if mode is ss
    if wc.ss == "ss":
        inputs["pc"] = inputs["pc"] + ".ss"
        inputs["lnc"] = inputs["lnc"] + ".ss"
    return inputs


rule lncfinder_train:
    input:
        unpack(get_lncfinder_train_inputs)
    output:
        model="results/{expt}/training/{fold}/lncfinder/{fold}_{ss}.RData"
    wildcard_constraints:
        ss="(ss|no-ss)"
    log:
        "logs/{expt}/training/{fold}/lncfinder/lncfinder_train_{fold}_{ss}.log"
    benchmark:
        "benchmarks/{expt}/training/{fold}/lncfinder/lncfinder_train_{fold}_{ss}.txt"
    conda:
        "../envs/lncfinder_env.yaml"
    threads: 20
    params:
        ss=lambda wc: "TRUE" if wc.ss == "ss" else "FALSE",
    resources:
        mem_mb_per_cpu=3000,
        runtime="7d"
    script:
        "../scripts/train_lncfinder.R"

rule feelnc_train:
    input:
        #fasta=lambda wc:f"resources/{config["datasets"][wc.db]["reference"]}.transcripts.fa",
        #fasta="results/{expt}/training/{fold}/{dset}.{test_info}.fa",  # If needed, use wildcard constraints
        fasta=lambda wc:config["datasets"][wc.dset]["test"],
        mrna="results/{expt}/datasets/{fold}/train_pc.fa",
        lncrna="results/{expt}/datasets/{fold}/train_lnc.fa"
    output:
        # TODO: separate output into training and inference results
        "results/{expt}/training/{fold}/FEELnc/{fold}.{test_info}.fa_RF.txt",
    log:
        "logs/{expt}/training/feelnc_train_{fold}.{test_info}.log"
    benchmark:
        "benchmarks/{expt}/training/feelnc_train_{fold}.{test_info}.txt"
    threads: 20
    resources:
        mem_mb_per_cpu=12000
    params:
        output_dir=lambda wc, output: os.path.dirname(output[0]),
        # Since FEELnc creates output names by default as {input_name}{suffix}
        # we must set it manually to match the output file defined above.
        # TODO: make this more intuitive. Might have to rethink the {test_info} wildcard usage.
        output_name=lambda wc: f"{wc.fold}.{wc.test_info}.fa",
        extra=""
    conda:
        "../envs/feelnc_env.yaml"
    shell:
        """
        FEELnc_codpot.pl \
            -i {input.fasta} \
            -a {input.mrna} \
            -l {input.lncrna} \
            -o {params.output_name} \
            -p {threads} \
            --kmer=1,2,3,6,9,12 \
            --learnorftype=0 \
            --testorftype=0 \
            --outdir={params.output_dir} \
            {params.extra} \
            > {log} 2>&1 \
        """
