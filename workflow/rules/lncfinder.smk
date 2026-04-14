def get_version(db):
    # Extract the version number from the database name
    match = re.search(r"M?\d+", db)
    if match:
        return match.group(0)
    else:
        if db == "toy":
            return "47"  # Default to human for toy dataset
        else:
            raise ValueError(f"Could not extract version from database name: {db}")


def species(expt):
    ref = config["experiments"][expt]["reference"]
    # Determine species based on the database name
    return "Mouse" if get_version(ref).startswith("M") else "Human"


rule all_lncfinder:
    input:
        expand(
            "results/{expt}/lncfinder/{fold}_{ss}.lncfinder",
            expt=["gencode.v47.common.cdhit.cv"],
            ss=["ss", "no-ss"],
            fold=[f"fold{n}" for n in range(1, 6)],
        ),


rule install_lncfinder:
    output:
        "software/lncfinder.installed",
    conda:
        "../envs/lncfinder_env.yaml"
    log:
        "logs/lncfinder/install.log",
    shell:
        """
        Rscript -e 'install.packages(c("LncFinder", "logger"), repos="https://cloud.r-project.org/")' > {log} 2>&1
        touch software/lncfinder.installed
        """


rule lncfinder_calculate_ss:
    """
    Calculate secondary structure using LncFinder.
    Use the full fasta as input, allowing downstream tools to subset as needed.
    """
    input:
        "software/lncfinder.installed",
        fasta=lambda wc: config["experiments"][wc.expt]["fasta"],
    output:
        out="results/{expt}/datasets/ss/lncfinder/{expt}.fa.ss",
    conda:
        "../envs/lncfinder_env.yaml"
    log:
        "logs/{expt}/datasets/lncfinder/calculate_ss.log",
    benchmark:
        "benchmarks/{expt}/datasets/lncfinder/calculate_ss.txt"
    resources:
        mem_mb=60000,
        runtime="5d",
    threads: 20
    script:
        "../scripts/lncfinder_calculate_ss.R"


rule lncfinder_subset_ss:
    """
    Subset secondary structures for each fold.
    """
    input:
        ss=rules.lncfinder_calculate_ss.output.out,
        fasta="results/{expt}/datasets/{fold}/{dset}.fa",
    output:
        out="results/{expt}/datasets/{fold}/{dset}.fa.ss",
    conda:
        "lnc-datasets"
    log:
        "logs/{expt}/datasets/{fold}/lncfinder/{dset}_subset_ss.log",
    benchmark:
        "benchmarks/{expt}/datasets/{fold}/lncfinder/{dset}_subset_ss.txt"
    threads: 1
    script:
        "../scripts/lncfinder_subset_ss.py"


def get_lncfinder_model(wildcards):
    wc = wildcards
    expt_config = config["experiments"][wc.expt]
    if expt_config["models"]["lncfinder"] == "custom":
        # Model is custom, return path to trained model based on SS usage
        ss_suffix = "ss" if wc.ss == "ss" else "no-ss"
        return f"results/{wc.expt}/training/{wc.fold}/lncfinder/{wc.fold}_{ss_suffix}.RData"
    else:
        # Model is the species name in lowercase
        return species(expt_config["reference"]).lower()


def get_run_lncfinder_inputs(wildcards):
    # Default, compulsory inputs
    inputs = {
        "installed": "software/lncfinder.installed",
        "fasta": "results/{expt}/datasets/{fold}/test_all.fa",
    }
    wc = wildcards
    # SS file if mode is ss
    if wc.ss == "ss":
        inputs["fasta"] = inputs["fasta"] + ".ss"

    # Trace existence of custom model
    model = get_lncfinder_model(wc)
    if model.endswith(".RData"):
        inputs["model"] = model
    return inputs


rule run_lncfinder:
    input:
        unpack(get_run_lncfinder_inputs),
    output:
        out="results/{expt}/testing/{fold}/lncfinder/{fold}_{ss}.lncfinder",
    wildcard_constraints:
        ss="(ss|no-ss)",
    conda:
        "../envs/lncfinder_env.yaml"
    log:
        "logs/{expt}/testing/{fold}/lncfinder/{fold}_{ss}.log",
    benchmark:
        "benchmarks/{expt}/testing/{fold}/lncfinder/{fold}_{ss}.txt"
    resources:
        mem_mb=6000,
        runtime="1h",
    threads: 10
    params:
        ss=lambda wc: "TRUE" if wc.ss == "ss" else "FALSE",
        species=lambda wc: species(wc.expt).lower(),
        use_custom=lambda wc: (
            "TRUE"
            if config["experiments"][wc.expt]["models"]["lncfinder"] == "custom"
            else "FALSE"
        ),
        model=get_lncfinder_model,
    script:
        "../scripts/run_lncfinder.R"
