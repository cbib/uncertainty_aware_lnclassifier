from pathlib import Path
import re

def get_version(db):
    # Extract the version number from the database name
    match = re.search(r'M?\d+', db)
    if match:
        return match.group(0)
    else:
        raise ValueError(f"Could not extract version from database name: {db}")

def species(db):
    # Determine species based on the database name
    return "Mouse" if get_version(db).startswith("M") else "Human"

rule all_lncadeep:
    input:
        #"software/lncadeep.installed",
        expand("results/{db}/lncadeep/{model}_LncADeep_lncRNA_results", db=["gencode.v47", "gencode.v49"], model=["full", "partial"])

rule install_lncadeep:
    output:
        "software/lncadeep.installed"
    conda:
        "../envs/lncadeep_env.yaml"
    log:
        "logs/setup/lncadeep_install.log"
    shell:
        """
        cd software
        git clone https://github.com/cyang235/LncADeep.git temp
        mv temp/* LncADeep/
        rm -rf temp
        cd LncADeep

        # Download pfam markov model files according to LncADeep documentation
        wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam29.0/Pfam-A.hmm.gz
        gzip -d Pfam-A.hmm.gz
        mv Pfam-A.hmm ./LncADeep_lncRNA/src/

        # Run custom LncADeep configuration (modified from original configure script)
        ## Make all scripts in src/ executable
        workdir=$(pwd)
        LncADeep_src_dir=$workdir/LncADeep_anno/src
        chmod +x -R $LncADeep_src_dir

        ## Add LncADeep folder to PATH within the conda environment
        mkdir -p $CONDA_PREFIX/etc/conda/activate.d
        echo 'export OLD_PATH=$PATH' >> $CONDA_PREFIX/etc/conda/activate.d/lncadeep_path.sh
        echo 'export PATH=$PATH:$LncADeep_src_dir:$workdir' >> $CONDA_PREFIX/etc/conda/activate.d/lncadeep_path.sh
        mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
        echo 'export PATH=$OLD_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/lncadeep_path.sh
        echo 'unset OLD_PATH' >> $CONDA_PREFIX/etc/conda/deactivate.d/lncadeep_path.sh

        echo "LncADeep installation completed."
        cd ../..
        touch {output}
        """


rule lncadeep:
    input:
        fasta=ancient("resources/{db}.transcripts.fa"),
    output:
        out=directory("results/{db}/lncadeep/{model}_LncADeep_lncRNA_results")
    wildcard_constraints:
        model="full|partial"
    conda:
        "../envs/lncadeep_env.yaml"
    log:
        "logs/{db}/lncadeep/{db}_{model}.log"
    benchmark:
        "benchmarks/{db}/lncadeep/{db}_{model}.tsv"
    threads: 40
    resources:
        mem_mb=20000,
        time="3d"
    params:
        # Use a simple, slash-free prefix to avoid nested paths inside LncADeep
        prefix=lambda wc: wc.model,
        species=lambda wc: species(wc.db).lower(),
        hmm_threads=40,
        class_threads=40
    shell:
        """
        mkdir -p $(dirname "{log}")
        {{
            echo "Running LncADeep on {input.fasta}"
            echo "Using model: {wildcards.model}"
            echo "Species: {params.species}"
            echo "Output prefix: {params.prefix}"

            # Compute absolute paths before changing working directory
            SCRIPT=$(python -c "import os; print(os.path.abspath('software/LncADeep/LncADeep.py'))")
            FASTA=$(python -c "import os; print(os.path.abspath('{input.fasta}'))")

            # Work inside the parent of the declared output directory
            OUT_PARENT=$(dirname "{output.out}")
            mkdir -p "$OUT_PARENT"
            cd "$OUT_PARENT"

            python "$SCRIPT" -MODE lncRNA \
            -f "$FASTA" \
            -o {params.prefix} \
            -s {params.species} \
            -m {wildcards.model} \
            -th {params.class_threads} \
            -HMM {params.hmm_threads}
        }} >> {log} 2>&1

        echo "Done!" >> {log}
        """
