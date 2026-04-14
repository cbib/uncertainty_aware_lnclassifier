rule all:
    input:
        expand("{db}_ORFs.fasta", db=["gencode.v47.toy"]),
        expand("{db}.cpc2", db=["gencode.v47.toy"]),

rule create_toy:
    input:
        fasta="resources/gencode.v47.transcripts.fa.gz"
    output:
        fasta="gencode.v47.toy.fa"
    params:
        n=1000
        extra="-s 42" # Seed for reproducibility
    conda:
        "base"
    script:
        "workflow/scripts/create_toy.py {input.fasta} {output.fasta} {params.n} {params.extra}"

rule run_cpc2_and_extract_orfs:
    input:
        fasta="{db}_ORFs.fa"
    output:
        txt="{db}.cpc2"
    params:
        orf_flag="--ORF"
    conda:
        "envs/cpc2_env.yaml"
    shell:
        """
        python CPC2.py \
            -i {input.fasta} \
            -o {output.txt} \
            {params.orf_flag}
        """

rule run_kmer_in_short:
    input:
        fasta_file="{db}_ORFs.fasta"
    output:
        kis_file="{db}_KIS.txt"
    params:
        kmer_size="1",
        nb_cores="20",
        dont_reverse="--dont-reverse",
        # per_seq="--perSeq" -> Commented to get only one file with all sequences
        extra=""
    conda:
        "KIS_env.yaml"
    shell:
        """
        KmerInShort \
            -file {input.fasta_file} \
            -kmer-size {params.kmer_size} \
            -nb-cores {params.nb_cores} \
            -out {output.kis_file} \
            {params.dont_reverse} \
            {params.per_seq} \
            {params.extra}
        """

rule feelnc_codpot:
    input:
        fasta="{db}.transcripts.fasta", # Full database
        mrna="{db}.pc_transcripts.fa"
        lncrna="{db}.lncRNA_transcripts.fa"
    output:
        txt="{db}_feelnc.txt"
    params:
        extra=""
    conda:
        "envs/feelnc_env.yaml"
    shell:
        """
        FEELnc_codpot.pl \
            -i {input.fasta} \
            -a {input.mrna} \
            -l {input.lncrna} \
            --kmer=1,2,3,6,9,12 \
            --learnorftype=0 \
            --testorftype=0 \
            --outdir=results/{wildcards.db}/FEELnc
        """
