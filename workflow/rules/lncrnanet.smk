rule all_lncrnanet:
    input:
        #"software/lncrnanet.installed",
        expand("results/{dset}/lncrnanet/{dset}.lncrnanet", dset=["gencode.v47"])


rule install_lncrnanet:
    output:
        "software/lncrnanet.installed"
    conda:
        "../envs/lncrnanet_env.yaml"
    shell:
        """
        set -x
        cd software
        git clone https://github.com/nofundamental/lncRNAnet.git temp
        rm -rf LncRNAnet/
        mv temp/ LncRNAnet/

        chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/keras_config.sh

        cd ..
        touch {output}
        """


rule install_lncrnanet_old:
    output:
        "software/lncrnanet.installed_old"
    conda:
        "../envs/lncrnanet_env.yaml"
    shell:
        """
        set -x
        cd software
        git clone https://github.com/nofundamental/lncRNAnet.git temp
        rm -rf LncRNAnet/
        mv temp/ LncRNAnet/

        # Create activate.d script
        mkdir -p $CONDA_PREFIX/etc/conda/activate.d
        ## We use cat with a "heredoc" to pass multiple lines to the script
        ## Nice explanation here: https://stackoverflow.com/questions/2500436/how-does-cat-eof-work-in-bash
        ## Note how we removed indentation so that the script does not have leading spaces
        ## Also note the activate script also uses a heredoc to create the keras.json file, hence the complicated escaping

        cat > $CONDA_PREFIX/etc/conda/activate.d/keras_config.sh <<'ACTIVATE_SCRIPT'
#!/bin/bash
# Backup existing keras.json if it exists
if [ -f "$HOME/.keras/keras.json" ]; then
    cp "$HOME/.keras/keras.json" "$HOME/.keras/keras.json.bak"
fi

# Create new keras.json configuration
mkdir -p "$HOME/.keras"
cat > "$HOME/.keras/keras.json" <<'KCONFIG'
{{
    "epsilon": 1e-07,
    "image_data_format": "channels_first",
    "backend": "theano",
    "floatx": "float32"
}}
KCONFIG
ACTIVATE_SCRIPT
        chmod +x $CONDA_PREFIX/etc/conda/activate.d/keras_config.sh

        # Create deactivate.d script
        mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
        cat > $CONDA_PREFIX/etc/conda/deactivate.d/keras_config.sh <<'DEACTIVATE_SCRIPT'
#!/bin/bash
# Restore the original keras.json if backup exists
if [ -f "$HOME/.keras/keras.json.bak" ]; then
    mv "$HOME/.keras/keras.json.bak" "$HOME/.keras/keras.json"
else
    rm -f "$HOME/.keras/keras.json"
fi
DEACTIVATE_SCRIPT
        chmod +x $CONDA_PREFIX/etc/conda/deactivate.d/keras_config.sh

        cd ..
        touch {output}
        """


rule test_lncrnanet:
    # We test the tool with the provided data to ensure correct installation
    input:
        "software/lncrnanet.installed"
    output:
        "software/lncrnanet.test_ok"
    conda:
        "../envs/lncrnanet_env.yaml"
    shell:
        """
        set -x
        cd software/LncRNAnet
        python ./code/lncRNAnet.py ./data/test/HT_100.fasta ./data/test/HT_100.out
        cd ../..
        touch {output}
        """


rule filter_fasta_for_lncrnanet:
    input:
        "resources/{dset}.transcripts.fa"
    output:
        "results/{dset}/lncrnanet/transcripts.fa.for_lncrnanet"
    conda:
        "lnc-datasets"
    script:
        "../scripts/filter_fasta_for_lncrnanet.py"


rule lncrnanet:
    # 1. lncrnanet does not deal with sequences carrying non-ACGT characters, so we filter them out
    # 2. lncrnanet divides sequences in buckets according to their length,
    # and processes each bucket separately. However, the buckets are hardcoded in the script,
    # so sequences longer than 100,000 nt are not processed at all.
    # We could modify the script to add more buckets, but for now we just filter out
    # sequences longer than 100,000 nt.
    input:
        fasta=ancient("results/{dset}/lncrnanet/transcripts.fa.for_lncrnanet"),
        test_ok="software/lncrnanet.test_ok"
    output:
        out="results/{dset}/lncrnanet/{dset}.lncrnanet"
    conda:
        "../envs/lncrnanet_env.yaml"
    log:
        "logs/{dset}/lncrnanet/{dset}.log"
    benchmark:
        "benchmarks/{dset}/lncrnanet/{dset}.txt"
    threads: 2
    resources:
        mem_mb=120000,
        time="4d"
    shell:
        """
        {{
            echo "Running LncRNAnet on {input.fasta}"
            export THEANO_FLAGS="device=cpu,optimizer=None,exception_verbosity=high"
            echo "Using THEANO_FLAGS: $THEANO_FLAGS"
            echo "Output will be saved to {output.out}"
            mkdir -p $(dirname "{output.out}")

            cd software/LncRNAnet
            python ./code/lncRNAnet.py ../../{input.fasta} ../../{output.out}
        }} 2>&1 | tee {log}
        """
