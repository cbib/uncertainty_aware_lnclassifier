def get_tools_to_process(wildcards):
    selected_tools = config["experiments"][wildcards.expt]["tools"]
    tools = {
        "feelnc": "results/{expt}/testing/{fold}/FEELnc/{fold}_RF.txt",
        "cpat_p": "results/{expt}/testing/{fold}/cpat/{fold}.cpat.p.ORF_prob.best.tsv",
        "cpat_l": "results/{expt}/testing/{fold}/cpat/{fold}.cpat.l.ORF_prob.best.tsv",
        'cpat_cutoff': "results/{expt}/training/{fold}/cpat/cv/optimal_cutoff.txt",
        "lncDC_no_ss": "results/{expt}/testing/{fold}/lncDC/{fold}.lncDC.no_ss.csv",
        "lncDC_ss": "results/{expt}/testing/{fold}/lncDC/{fold}.lncDC.ss.csv",
        "mrnn": "results/{expt}/testing/{fold}/mRNN/{fold}.mRNN.multi.tsv",
        "lncrnabert": "results/{expt}/testing/{fold}/lncrnabert/kmer/classification.csv",
        "rnasamba_full": "results/{expt}/testing/{fold}/rnasamba/{fold}_full.tsv",
        "plncpro": "results/{expt}/testing/{fold}/plncpro/{fold}.with_feats.plncpro",
        "lncfinder_no_ss": "results/{expt}/testing/{fold}/lncfinder/{fold}_no-ss.lncfinder",
        "lncfinder_ss": "results/{expt}/testing/{fold}/lncfinder/{fold}_ss.lncfinder",
    }
    # Expand wildcards in paths
    tools = {k: v.format(expt=wildcards.expt, fold=wildcards.fold) for k, v in tools.items()}
    # Return only selected tools
    return {k: v for k, v in tools.items() if k in selected_tools}

def get_process_inputs(wildcards):
    ref = config["experiments"][wildcards.expt]["reference"]
    inputs = {
        "pc_transcripts": f"resources/{ref}.pc_transcripts.fa.gz",
        "lncRNA_transcripts": f"resources/{ref}.lncRNA_transcripts.fa.gz",
    }

    inputs = {**inputs, **get_tools_to_process(wildcards)}
    return inputs


rule merge_plncpro_feature_table:
    input:
        predictions="results/{expt}/testing/{fold}/plncpro/{fold}.plncpro",
        features="results/{expt}/testing/{fold}/plncpro/test_all.fa_all_features",  # TODO: Might want to rename rename this in the plncpro rule
    output:
        "results/{expt}/testing/{fold}/plncpro/{fold}.with_feats.plncpro",
    log:
        "logs/{expt}/testing/{fold}/merge_plncpro_feature_table.log"
    conda:
        "lnc-datasets"
    run:
        import pandas as pd
        df = pd.read_csv(
            input.predictions,
            sep="\t",
            header=None,
            names=["transcript_id", "prediction", "prob_coding", "prob_noncoding"],
        ).set_index("transcript_id")
        features = pd.read_csv(input.features, sep="\t", header=0).set_index('seqid')
        df = df.join(features, how="left")
        df.to_csv(output[0], sep="\t")

rule process:
    input:
        unpack(get_process_inputs)
    output:
        full="results/{expt}/testing/{fold}/tables/{fold}_full_table.tsv",
        class_table="results/{expt}/testing/{fold}/tables/{fold}_class_table.tsv",
        simple_class="results/{expt}/testing/{fold}/tables/{fold}_simple_class_table.tsv",
        no_class="results/{expt}/testing/{fold}/tables/{fold}_no_class_table.tsv",
        binary_class="results/{expt}/testing/{fold}/tables/{fold}_binary_class_table.tsv",
    log:
        "logs/{expt}/testing/{fold}/process.log"
    conda:
        "lnc-datasets"
    params:
        prefix=lambda wc: f"results/{wc.expt}/testing/{wc.fold}/tables/{wc.fold}",
    script:
        "../scripts/process.py"

def get_n_folds(wildcards):
    return config["experiments"][wildcards.expt]["n_folds"]


rule process_all_folds:
    input:
        lambda wc: expand(
            "results/{{expt}}/testing/{fold}/tables/{fold}_full_table.tsv",
            fold=[f"fold{i+1}" for i in range(get_n_folds(wc))],
        )
    output:
        touch("results/{expt}/testing/processing.done"),


rule merge_all_fold_tables:
    """Aggregate all fold-level tables into dataset-level merged tables."""
    input:
        full=lambda wc: expand(
            "results/{{expt}}/testing/{fold}/tables/{fold}_full_table.tsv",
            fold=[f"fold{i+1}" for i in range(get_n_folds(wc))],
        ),
        class_table=lambda wc: expand(
            "results/{{expt}}/testing/{fold}/tables/{fold}_class_table.tsv",
            fold=[f"fold{i+1}" for i in range(get_n_folds(wc))],
        ),
        simple_class=lambda wc: expand(
            "results/{{expt}}/testing/{fold}/tables/{fold}_simple_class_table.tsv",
            fold=[f"fold{i+1}" for i in range(get_n_folds(wc))],
        ),
        no_class=lambda wc: expand(
            "results/{{expt}}/testing/{fold}/tables/{fold}_no_class_table.tsv",
            fold=[f"fold{i+1}" for i in range(get_n_folds(wc))],
        ),
        binary_class=lambda wc: expand(
            "results/{{expt}}/testing/{fold}/tables/{fold}_binary_class_table.tsv",
            fold=[f"fold{i+1}" for i in range(get_n_folds(wc))],
        ),
    output:
        full="results/{expt}/tables/{expt}_full_table.tsv",
        class_table="results/{expt}/tables/{expt}_class_table.tsv",
        simple_class="results/{expt}/tables/{expt}_simple_class_table.tsv",
        no_class="results/{expt}/tables/{expt}_no_class_table.tsv",
        binary_class="results/{expt}/tables/{expt}_binary_class_table.tsv",
    log:
        "logs/{expt}/testing/merge_all_fold_tables.log"
    conda:
        "lnc-datasets"
    script:
        "../scripts/merge_folds.py"

# Alias for backwards compatibility
rule merge_fold_stats:
    input:
        expand(rules.merge_all_fold_tables.output.binary_class, expt="gencode.v47.common.cdhit.cv")
