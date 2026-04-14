rule all:
    input:
        "software/lncdc.installed",
        "software/cpat.installed"


rule prepare_install:
    output:
        dir("software"),
    shell:
        """
        mkdir -p software
        """


rule install_lncdc:
    output:
        "software/lncdc.installed"
    conda:
        "../../workflow/envs/lncdc_env.yaml"
    log:
        "logs/setup/lncdc_install.log"
    shell:
        """
        pwd
        # Prepare
        cd software

        # Install
        echo "Installing LncDC in conda environment..."
        wget https://github.com/lim74/LncDC/archive/refs/tags/v1.3.6.zip
        unzip -o v1.3.6.zip
        cd LncDC-1.3.6
        python setup.py install

        # Check installation (provided by LncDC)
        python test_requirements.py

        #TODO: Decide if we symlink the LncDC.py script to $CONDA_PREFIX/bin

        # Complete installation
        echo "LncDC installation completed."
        cd ../..
        pwd
        touch {output}
        """


rule install_cpat_manual:
    output:
        "software/cpat.installed"
    conda:
        "../../workflow/envs/cpat_env.yaml"
    log:
        "logs/setup/cpat_install.log"
    shell:
        """
        pwd
        # CPAT was installed with pip, checking if it is available
        cpat --version || {{ echo "CPAT is not installed. Please install it manually."; exit 1; }}

        # We still need to download the source code to have test files

        # Prepare
        cd software

        # CPAT is actively maintained, so you might want to check the latest version
        wget https://github.com/liguowang/cpat/archive/refs/tags/v3.0.5.zip
        unzip -o v3.0.5.zip

        # Complete installation
        echo "CPAT installation completed."
        cd ..
        pwd
        touch {output}
        """


rule install_mrnn:
    output:
        "software/mrnn.installed"
    conda:
        "../../workflow/envs/mrnn_env.yaml"
    log:
        "logs/setup/mrnn_install.log"
    shell:
        """
        pwd
        # Prepare
        cd software
        #git clone https://github.com/IndicoDataSolutions/Passage.git
        #cd Passage
        #git checkout 4b8be6dc4d17ccc78a21abc82afb2d0f72c04b91
        #pip install .
        #cd ..

        # Install
        echo "Installing mrnn in conda environment..."
        git clone https://github.com/hendrixlab/mRNN.git

        echo "Getting mrrn datasets..."
        wget https://files.osf.io/v1/resources/4htpy/providers/osfstorage/?zip= -O mrnn_data.zip
        unzip -o mrnn_data.zip -d mRNN/

        # Complete installation
        echo "mrnn installation completed."
        cd ..
        pwd
        touch {output}
        """
