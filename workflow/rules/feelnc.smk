rule feelnc_codpot_cv:
    input:
        fasta="results/{expt}/datasets/{fold}/test_all.fa",
        mrna="results/{expt}/datasets/{fold}/train_pc.fa",
        lncrna="results/{expt}/datasets/{fold}/train_lnc.fa"
    output:
        "results/{expt}/testing/{fold}/FEELnc/{fold}_RF.txt"
    threads: 20
    params:
        outdir=lambda wc, output: os.path.dirname(output[0]),
        outname=lambda wc: f"{wc.fold}",
        extra=""
    conda:
        "../envs/feelnc_env.yaml"
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
