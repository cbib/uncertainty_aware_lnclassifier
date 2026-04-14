import os

def get_version(db):
    # Extract the version number from the database name
    match = re.search(r'M?\d+', db)
    if match:
        return match.group(0)
    else:
        raise ValueError(f"Could not extract version from database name: {dset}")


rule all_plncpro:
    input:
        #"software/plncpro.installed"
        # GRCh38, GRCm3expand("results/{dset}/lncDC/{dset}.lncDC.ss.csv", dset=config["run_dbs"]),8
        #expand("resources/blast_dbs/{reference}.primary_assembly.fa", reference=["GRCh38", "GRCm39"])
        #expand("resources/blast_dbs/uniprot_sprot.dmnd", reference=["GRCh38", "GRCm39"])
        #"results/plncpro/model/plncpro_model.rds",
        expand("results/{dset}/plncpro/{dset}.plncpro", dset=["gencode.v47.test", "gencode.v47.trained_test"])

ruleorder: blast_dbs_symlink > diamond_makedb

rule blast_dbs_symlink:
    output:
        "resources/blast_dbs/uniprot_sprot.fasta",
        "resources/blast_dbs/uniprot_sprot.dmnd",
    log:
        "logs/blast_dbs_symlink.log"
    conda:
        "base"
    params:
        fasta_dir="/mnt/cbib/bank/uniprot/current_release/uniprot_sprot.fasta",
        diamond_dir="/mnt/cbib/bank/uniprot/current_release/UNIPROT.dmnd"
    shell:
        """
        mkdir -p resources/blast_dbs
        {{
            if [ ! -f resources/blast_dbs/uniprot_sprot.fasta ]; then
            ln -s {params.fasta_dir} resources/blast_dbs/uniprot_sprot.fasta
            fi
            if [ ! -f resources/blast_dbs/uniprot_sprot.dmnd ]; then
            ln -s {params.diamond_dir} resources/blast_dbs/uniprot_sprot.dmnd
            fi
            touch {output}
        }} > {log} 2>&1
        """


rule diamond_makedb:
    input:
        fname = "resources/blast_dbs/uniprot_sprot.fasta",
    output:
        fname = "resources/blast_dbs/uniprot_sprot.dmnd"
    log:
        "logs/diamond_makedb/uniprot_sprot.log"
    params:
        extra=""
    threads: 8
    wrapper:
        "v7.4.0/bio/diamond/makedb"


rule diamond_blastx:
    input:
        fname_fastq="results/{expt}/datasets/{fold}/test_all.fa",
        fname_db="resources/blast_dbs/uniprot_sprot.dmnd",
    output:
        fname="results/{expt}/testing/{fold}/plncpro/test_all.fa_blastres",
    log:
        "logs/{expt}/testing/{fold}/plncpro/diamond_blastx_{fold}.log",
    benchmark:
        "benchmarks/{expt}/testing/{fold}/plncpro/diamond_blastx_{fold}.txt",
    params:
        extra="--outfmt 6 qseqid sseqid pident evalue nident qcovhsp score bitscore qframe qstrand",
    threads: 8,
    resources:
        runtime=120,
    wrapper:
        "v7.4.0/bio/diamond/blastx"


rule install_plncpro:
    output:
        "software/plncpro.installed",
    conda:
        "../envs/plncpro_env.yaml"
    log:
        "logs/plncpro/install.log",
    shell:
        """
        cd software
        git clone https://github.com/urmi-21/PLncPRO.git > {log} 2>&1
        cd PLncPRO
        bash tests/local_test.sh > {log} 2>&1
        echo "Manually download the training data from https://drive.google.com/file/d/108S-9Bt4CLCHTaCn6-HKTqQZDo0nssZe/" | tee -a {log}
        touch ../plncpro.installed
        """

# This rule builds the default Human model using the provided training transcripts.
# However, it does not work for gencode.v47.test (classification of test dataset with default models)
# Because the current rule plncpro_predict expects model to be in results/db/
rule plncpro_build:
    input:
        mrna="software/PLncPRO/plncpro_data/hg24/train/hg24_pct_train_5000.fa",
        lncRNA="software/PLncPRO/plncpro_data/hg24/train/hg24_lnct_train_5000.fa",
        db="resources/blast_dbs/uniprot_sprot.dmnd",  # They do not mention the release version of uniprot.
    output:
        model="software/PLncPRO/plncpro_data/hg24/train/model/plncpro_model.rds",
    conda:
        "../envs/plncpro_env.yaml"
    log:
        "logs/plncpro/build.log",
    threads: 1,
    resources:
        mem_mb_per_cpu=10000,
    params:
        # Get file and folder from output
        outdir=lambda wc, output: os.path.split(output.model)[0],
        outfile=lambda wc, output: os.path.split(output.model)[1],
    log:
        "logs/inference/plncpro/build.log",
    shell:
        """
        plncpro build -p {input.mrna} \
        -n {input.lncRNA} \
        -o {params.outdir} \
        -m {params.outfile} \
        -d {input.db} \
        -t {threads} > {log} 2>&1
        """


def get_plncpro_model(wildcards):
    """
    Input function to return the appropriate plncpro model,
    either custom trained or default human model.
    TODO: Generalize for mouse too
    """
    use_custom_training = config["experiments"][wildcards.expt]["models"].get("plncpro", "default")
    if use_custom_training == "custom":
        model = f"results/{wildcards.expt}/training/{wildcards.fold}/plncpro/{wildcards.fold}.model"
        print(f"Using custom trained PLncPRO model: {model}")
        return model
    else:
        print("Using default PLncPRO human model")
        return "software/PLncPRO/plncpro_data/hg24/train/model/plncpro_model.rds"


rule plncpro_predict:
    input:
        fasta="results/{expt}/datasets/{fold}/test_all.fa",
        blastres="results/{expt}/testing/{fold}/plncpro/test_all.fa_blastres",
        model=get_plncpro_model
    output:
        "results/{expt}/testing/{fold}/plncpro/{fold}.plncpro"
    log:
        "logs/{expt}/testing/{fold}/plncpro/{fold}.log",
    benchmark:
        "benchmarks/{expt}/testing/{fold}/plncpro/plncpro_predict.txt"
    conda:
        "../envs/plncpro_env.yaml"
    threads: 8,
    resources:
        mem_mb_per_cpu=5000,
    params:
        output_dir=lambda wc, output: subpath(output[0], parent=True),
    shell:
        """
        plncpro predict -i \
        {input.fasta} \
        -o {params.output_dir} \
        -p {wildcards.fold}.plncpro \
        -t {threads} \
        --blastres {input.blastres} \
        -m {input.model} > {log} 2>&1
        """
