def get_cpat_hexamer(wildcards):
    wc = wildcards
    cpat_config = config["experiments"][wc.expt]["models"].get("cpat", "default")
    if cpat_config == "custom":
        return f"results/{wc.expt}/training/{wc.fold}/cpat/{wc.fold}_Hexamer.tsv"
    else:
        species_name = species(wc.dset).capitalize() # NOTE: Only works with Human and Mouse for now
        return f"software/cpat-3.0.5/prebuilt_models/{species_name}_Hexamer.tsv"


def get_cpat_logit(wildcards):
    wc = wildcards
    cpat_config = config["experiments"][wc.expt]["models"].get("cpat", "default")
    if cpat_config == "custom":
        return f"results/{wc.expt}/training/{wc.fold}/cpat/{wc.fold}.logit.RData"
    else:
        species_name = species(wc.dset).capitalize() # NOTE: Only works with Human and Mouse for now
        return f"software/cpat-3.0.5/prebuilt_models/{species_name}_logitModel.RData"


rule cpat_cv:
    input:
        "results/{expt}/datasets/{fold}/test_all.fa"
    output:
        orfs="results/{expt}/testing/{fold}/cpat/{fold}.cpat.{mode}.ORF_seqs.fa",
        best_orfs="results/{expt}/testing/{fold}/cpat/{fold}.cpat.{mode}.ORF_prob.best.tsv"
    conda:
        "../envs/cpat_env.yaml"
    params:
        x=get_cpat_hexamer,
        d=get_cpat_logit,
        best_orf=lambda wc: wc.mode,
        min_orf=9,
        out_prefix=lambda wc, output: output.orfs.rsplit(".", 2)[0]
    benchmark:
        "benchmarks/{expt}/testing/{fold}/cpat/cpat.{mode}.txt"
    log:
        "logs/{expt}/testing/{fold}/cpat/cpat_{mode}.log"
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


rule run_internal_cv_cpat:
    input:
        expand("results/{expt}/training/{fold}/cpat/cv/test1.xls",
        expt=["gencode.v47.common.cdhit.cv"],
        fold=[f"fold{i}" for i in range(1, 6)]
        )


rule cpat_internal_10fold_cv:
    input:
        training_data="results/{expt}/training/{fold}/cpat/{fold}.feature.xls",
    output:
       "results/{expt}/training/{fold}/cpat/cv/test1.xls",
       "results/{expt}/training/{fold}/cpat/cv/optimal_cutoff.txt"
    params:
        output_prefix=subpath(output[0], parent=True),
        cutoff_precision=0.001  # CPAT uses 3 decimals
    conda:
        "../envs/cpat_env.yaml"
    log:
        "logs/{expt}/training/{fold}/cpat_internal_10fold_cv.log"
    benchmark:
        "benchmarks/{expt}/training/{fold}/cpat_internal_10fold_cv.txt"
    script:
        "../scripts/10Fold_CrossValidation.smk.R"
