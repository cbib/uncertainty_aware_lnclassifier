configfile: "../../config/config.yaml"

rule old_tools_all:
    input:
        #expand("results/{expt}/testing/{dset}/CPC2/{dset}.cpc2.txt", dset=config["inference_datasets"]),

# CPC2 requires libsvm installed inside the CPC2 folder
# This is not good for automation and reproducibility
# libsvm can be installed via conda, but CPC2 expects it in a specific path, throwing an error
# TODO: Fix libsvm path issue in CPC2 installation
rule install_cpc2:
    output:
        "software/CPC2.installed"
    conda:
        "workflow/envs/cpc2_env.yaml"
    log:
        "logs/setup/cpc2_install.log"
    shell:
        """
        # CPC2 was installed via conda, now fix libsvm installation
        {{
            echo "Checking for libsvm installation"
            if ! command -v svm-predict &> /dev/null; then
                echo "Installing libsvm via conda"
                conda install -c conda-forge libsvm -y
            else
                echo "libsvm is already installed"
            fi

            echo "Linking libsvm binaries to CPC2 expected location"
            mkdir -p $CONDA_PREFIX/libs/libsvm/libsvm-3.18
            for l in svm-predict svm-train svm-scale; do
                echo "Linking $l to CPC2 libsvm folder"
                ln -s $(which $l) $CONDA_PREFIX/libs/libsvm/libsvm-3.18/$l
            done
            echo "CPC2 installation completed."
            touch {output}
        }} > {log} 2>&1
        """



rule run_cpc2:
    input:
        fasta=lambda wc: config["datasets"][wc.dset]["test"],
        install="software/CPC2.installed"
    output:
        txt="results/{expt}/testing/{dset}/CPC2/{dset}.cpc2.txt"
    params:
        orf_flag="--ORF",
        out_prefix=lambda wildcards, output: output.txt.replace(".txt", "")
    conda:
        "workflow/envs/cpc2_env.yaml"
    benchmark:
        "benchmarks/{expt}/testing/{dset}/run_cpc2.txt"
    log:
        "logs/{expt}/testing/{dset}/run_cpc2.txt"
    shell:
        """
        {{
            echo "Running CPC2 on {input.fasta}"
            echo "Output will be saved to {output.txt}"
            echo "Using ORF flag: {params.orf_flag}"

            CPC2.py \
            -i {input.fasta} \
            -o {params.out_prefix} \
            {params.orf_flag}
        }} > {log} 2>&1
        """

rule extract_orfs:
    input:
        fasta=lambda wc: config["datasets"][wc.dset]["test"],
        cpc2_txt="results/{expt}/testing/{dset}/CPC2/{dset}.cpc2.txt"
    output:
        orfs="results/{expt}/testing/{dset}/CPC2/{dset}.ORFs.fa"
        # TODO: At some point we may need to output ORFs in a table format as well
    conda:
        "/isilon/home/dgarcia/software/miniconda3"  # User Base Conda environment (not snakemake base)
    benchmark:
        "benchmarks/{expt}/testing/{dset}/extract_orfs.txt"
    script:
        "workflow/scripts/extract_orfs.py"
