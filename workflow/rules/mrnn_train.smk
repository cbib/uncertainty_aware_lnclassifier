include: "./datasets.smk"
import os

"""
mRNN Training Pipeline with Two-Stage Training
===============================================

Stage 1 (Pretrain):
    - Train N models from scratch with different seeds
    - Test all model checkpoints
    - Select top K best models

Stage 2 (Train):
    - Continue training from K best pretrained models
    - Run M training runs per pretrained model
    - Test all trained model checkpoints
    - Select final best model

The pipeline uses checkpoints to handle dynamic outputs and enables
parallel execution at both stages.
"""

###########################################
# TRAINING STAGE CONFIGURATION
###########################################
TRAIN_STAGES = {
    "pretrain": {
        "n_seeds": 30,
        "seed_start": 42,
        "epochs": 4,
        "n_select": 6,
    },
    "train": {
        "n_seeds": 10,  # Will use preselected models
        "seed_start": 0,  # Not used, comes from checkpoint
        "epochs": 10,
        "n_select": 5,
    }
}


def get_pretrain_seeds():
    """Generate list of all pretrain seed values"""
    start = TRAIN_STAGES["pretrain"]["seed_start"]
    n = TRAIN_STAGES["pretrain"]["n_seeds"]
    return list(range(start, start + n))


def get_train_runs():
    """Generate list of training run numbers"""
    return list(range(TRAIN_STAGES["train"]["n_seeds"]))


def pretrained_accuracies_per_seed(wildcards):
    """
    Find all accuracy files for models from one pretrain seed.

    This function:
    1. Triggers the mrnn_pretrain checkpoint for this seed
    2. Discovers all model checkpoints created during training
    3. Returns expected accuracy file paths for each checkpoint
    """

    # Trigger the checkpoint for this seed
    output_flag = checkpoints.mrnn_pretrain.get(expt=wildcards.expt,
                                                fold=wildcards.fold,
                                                SEED=wildcards.SEED).output[0]

     # Get the output directory (because the output of the checkpoint is a touch file)
    output_dir = subpath(output_flag, parent=True)

    # Find all model files for this seed
    models = glob_wildcards(
        os.path.join(output_dir, f"mRNN.model.{wildcards.SEED}.{{epoch,\\d+}}")
    )

    # Return all the expected accuracy files for this seed
    accuracy_dir = os.path.join(output_dir, "accuracy_tests")
    return expand(
        os.path.join(accuracy_dir, f"mRNN.model.{wildcards.SEED}.{{epoch}}_acc.txt"),
        epoch=models.epoch
    )


def find_best_pretrained_models(wildcards):
    """
    Find all best pretrained models selected by the checkpoint.

    This triggers the mrnn_get_best_pretrained_models checkpoint and
    returns the selected model identities as (seed, epoch) tuples.
    """
    # Trigger the checkpoint that selects best models
    best_models_dir = checkpoints.mrnn_get_best_pretrained_models.get(
        expt=wildcards.expt,
        fold=wildcards.fold
    ).output[0]

    # Find all models in the best_models directory
    models = glob_wildcards(
        os.path.join(best_models_dir, r"mRNN.model.{pre_SEED,\d+}.{pre_epoch,\d+}")
    )
    print(f"DEBUG find_best_pretrained_models: Found {len(models.pre_SEED)} best models: {models.pre_SEED}, {models.pre_epoch}")
    # Return list of (pre_SEED, pre_epoch) tuples
    return list(zip(models.pre_SEED, models.pre_epoch))


rule all_mrnn:
    input:
        expand("results/{expt}/training/{fold}/mRNN/trained/best_models", expt=config['inference_datasets'], fold=[f"fold{n}" for n in range(1,6)]),


###########################################
# CREATE DATASETS
###########################################
rule mrnn_augmented_dataset:
    """Create augmented training datasets with different random seeds"""
    input:
        pc_train="results/{expt}/training/{fold}/mRNN/datasets/mRNN_train_pc.fa",
        lnc_train="results/{expt}/training/{fold}/mRNN/datasets/mRNN_train_lnc.fa"
    output:
        pc_augmented="results/{expt}/training/{fold}/mRNN/datasets/mRNN_aug{SEED}_pc.fa",
        lnc_augmented="results/{expt}/training/{fold}/mRNN/datasets/mRNN_aug{SEED}_lnc.fa"
    conda:
        "lnc-datasets"
    log:
        "logs/{expt}/training/{fold}/mRNN/datasets/mRNN_augmented_dataset_{SEED}.log"
    benchmark:
        "benchmarks/{expt}/training/{fold}/mRNN/datasets/mRNN_augmented_dataset_{SEED}.txt"
    threads: 1
    params:
        num_augmented = 10,
        seed=lambda wc: int(wc.SEED),
    script:
        "../scripts/mrnn_create_augmented_dataset.py"


###########################################
# PRETRAIN MODELS (checkpoint)
###########################################
checkpoint mrnn_pretrain:
    """
    Pretrain models from scratch with different seeds.

    Creates multiple model checkpoint files (one per epoch) with filenames:
    mRNN.model.{SEED}.{epoch}

    The number of model checkpoint files is determined dynamically by early stopping.
    """
    input:
        pc_train="results/{expt}/training/{fold}/mRNN/datasets/mRNN_aug{SEED}_pc.fa",
        lnc_train="results/{expt}/training/{fold}/mRNN/datasets/mRNN_aug{SEED}_lnc.fa",
        pc_valid="results/{expt}/training/{fold}/mRNN/datasets/mRNN_valid_pc.fa",
        lnc_valid="results/{expt}/training/{fold}/mRNN/datasets/mRNN_valid_lnc.fa",
    output:
        touch("results/{expt}/training/{fold}/mRNN/pretrained/{SEED}.done.training")
    log:
        "logs/{expt}/training/{fold}/mRNN/pretrained/mRNN_pretrain_{fold}_{SEED}.log"
    benchmark:
        "benchmarks/{expt}/training/{fold}/mRNN/pretrained/mRNN_pretrain_{fold}_{SEED}.txt"
    conda:
        "../envs/mrnn_env.yaml"
    resources:
        mem_mb_per_cpu=lambda wc, attempt: 15000 * attempt
    retries: 2
    params:
        # File naming
        output_prefix=lambda wc: f"mRNN.model.{wc.SEED}",
        output_dir=lambda wc, output: subpath(output[0], parent=True),
        # Optimized hyperparameters
        embedding_size=128,
        recurrent_gate_size=32,
        dropout=0.4,
        # Sequence filters
        min_len=200,
        max_len=1000,
        # Other training params
        epochs=4,
        early_stop=3,
    shell:
        """
        export THEANO_FLAGS="device=cpu,floatX=float32,base_compiledir=/tmp/theano_$RANDOM"
        mkdir -p {params.output_dir}
        python software/mRNN/train_mRNN.py \
        -e {params.embedding_size} \
        -r {params.recurrent_gate_size} \
        -d {params.dropout} \
        -o {params.output_dir}/{params.output_prefix} \
        -s {params.early_stop} \
        -E {params.epochs} \
        {input.pc_train} \
        {input.lnc_train} \
        {input.pc_valid} \
        {input.lnc_valid} \
        > {log} 2>&1
        """


###########################################
# RULE TO EVALUATE ALL MODELS
###########################################
rule mrnn_test_accuracy:
    """
    Test a model on test data.

    This rule  handles both pretrained and trained models
    using the flexible {stage} and {model_info} wildcards.
    """
    input:
        model="results/{expt}/training/{fold}/mRNN/{stage}/mRNN.model.{model_info}",
        pc_valid="results/{expt}/training/{fold}/mRNN/datasets/mRNN_valid_pc.fa",
        lnc_valid="results/{expt}/training/{fold}/mRNN/datasets/mRNN_valid_lnc.fa"
    output:
        # Ouput must be in the same directory as the model
        # The -o parameter in test_mRNN.py is not taken into account
        "results/{expt}/training/{fold}/mRNN/{stage}/accuracy_tests/mRNN.model.{model_info}_acc.txt"
    conda:
        "../envs/mrnn_env.yaml"
    log:
        "logs/{expt}/training/{fold}/mRNN/mRNN_test_accuracy/{fold}.{stage}.{model_info}.log"
    params:
        output_dir=lambda wc, output: subpath(output[0], parent=True),
    threads: 1
    resources:
        mem_mb_per_cpu=lambda wc, attempt: 5000 * attempt
    retries: 2
    benchmark:
        "benchmarks/{expt}/training/{fold}/mRNN/mRNN_test_accuracy/{fold}.{stage}.{model_info}.txt"
    wildcard_constraints:
        model_info=r"[^/]+",  # Ensure model_info doesn't capture slashes
        stage="(pretrained|trained)"  # Constrain stage to expected values
    shell:
        """
        export THEANO_FLAGS="device=cpu,floatX=float32,base_compiledir=/tmp/theano_$RANDOM"
        mkdir -p {params.output_dir}
        (cd {params.output_dir} && \
        python $OLDPWD/software/mRNN/test_mRNN.py \
        -w $OLDPWD/{input.model} \
        -o . \
        $OLDPWD/{input.pc_valid} \
        $OLDPWD/{input.lnc_valid}) > {log} 2>&1
        """


###########################################
# AGGREGATE ACCURACY FILES PER SEED (enables parallel execution)
###########################################
rule mrnn_aggregate_accuracies_per_seed:
    """
    Aggregate accuracy files for all checkpoints from one pretrain seed.

    This enables parallel execution: all models from each seed can be tested
    as soon as that seed's pretraining completes.
    """
    input:
        # This input gathers all accuracy files for each seed.
        # It triggers execution of mrnn_test_accuracy for all models derived from that seed
        pretrained_accuracies_per_seed
    output:
        touch("results/{expt}/training/{fold}/mRNN/pretrained/accuracy_tests/{SEED}.done")


###########################################
# COPY BEST PRETRAINED MODELS (when all accuracy files are ready)
###########################################
checkpoint mrnn_get_best_pretrained_models:
    """
    Select the top N pretrained models based on test accuracy.

    This checkpoint:
    1. Waits for all pretrained models to be tested
    2. Copies the best N models to a new directory
    3. Triggers DAG re-evaluation for the training stage
    """
    input:
        # This input declares all expected inputs at DAG allowing parallel execution
        # of the previous rules mrnn_aggregate_seed_accuracies and mrnn_test_accuracy
        expand("results/{{expt}}/training/{{fold}}/mRNN/pretrained/accuracy_tests/{SEED}.done",
               SEED=get_pretrain_seeds())
    output:
        best_models_dir=directory("results/{expt}/training/{fold}/mRNN/pretrained/best_models")
    log:
        "logs/{expt}/training/{fold}/mRNN/pretrained/mRNN_get_best_pretrained_models_{fold}.log"
    params:
        accuracy_path=lambda wc: f"results/{wc.expt}/training/{wc.fold}/mRNN/pretrained/accuracy_tests",
        n_models=TRAIN_STAGES["pretrain"]["n_select"]
    script:
        "../scripts/mrnn_select_best_models.py"


###########################################
# TRAIN MODELS (checkpoint)
###########################################
checkpoint mrnn_train:
    input:
        pc_train="results/{expt}/training/{fold}/mRNN/datasets/mRNN_train_pc.fa",
        lnc_train="results/{expt}/training/{fold}/mRNN/datasets/mRNN_train_lnc.fa",
        pc_valid="results/{expt}/training/{fold}/mRNN/datasets/mRNN_valid_pc.fa",
        lnc_valid="results/{expt}/training/{fold}/mRNN/datasets/mRNN_valid_lnc.fa",
        pretrained_model="results/{expt}/training/{fold}/mRNN/pretrained/best_models/mRNN.model.{pre_SEED}.{pre_epoch}",
    output:
        touch("results/{expt}/training/{fold}/mRNN/trained/{pre_SEED}.{pre_epoch}.{run}.done.training")
    log:
        "logs/{expt}/training/{fold}/mRNN/trained/{fold}.{pre_SEED}.{pre_epoch}.{run}.log"
    benchmark:
        "benchmarks/{expt}/training/{fold}/mRNN/trained/{fold}_{pre_SEED}.{pre_epoch}.{run}.txt"
    conda:
        "../envs/mrnn_env.yaml"
    resources:
        mem_mb_per_cpu=lambda wc, attempt: 50000 * attempt
    retries: 2
    params:
        # File naming
        output_prefix=lambda wc: f"mRNN.model.{wc.pre_SEED}.{wc.pre_epoch}.{wc.run}",
        output_dir=lambda wc, output: subpath(output[0], parent=True),
        # Optimized hyperparameters
        embedding_size=128,
        recurrent_gate_size=32,
        dropout=0.4,
        # Sequence filters
        min_len=200,
        max_len=1000,
        # Other training params
        epochs=10,
        early_stop=3,
    shell:
        """
        export THEANO_FLAGS="device=cpu,floatX=float32,base_compiledir=/tmp/theano_$RANDOM"
        mkdir -p {params.output_dir}
        python software/mRNN/train_mRNN.py \
        -w {input.pretrained_model} \
        -e {params.embedding_size} \
        -r {params.recurrent_gate_size} \
        -d {params.dropout} \
        -o {params.output_dir}/{params.output_prefix} \
        -s {params.early_stop} \
        -E {params.epochs} \
        {input.pc_train} \
        {input.lnc_train} \
        {input.pc_valid} \
        {input.lnc_valid} \
        > {log} 2>&1
        """


def training_accuracies_per_run(wildcards):
    """
    Find all accuracy files for models from one training run.

    This function:
    1. Triggers the mrnn_train checkpoint for this run
    2. Discovers all model checkpoints created during training
    3. Returns expected accuracy file paths for each checkpoint
    """
    print(f"DEBUG training_accuracies_per_run: waiting for checkpoint for fold={wildcards.fold}, pre_SEED={wildcards.pre_SEED}, pre_epoch={wildcards.pre_epoch}, run={wildcards.run}")
    # Trigger the checkpoint for this training run
    output_flag = checkpoints.mrnn_train.get(
        expt=wildcards.expt,
        fold=wildcards.fold,
        pre_SEED=wildcards.pre_SEED,
        pre_epoch=wildcards.pre_epoch,
        run=wildcards.run
    ).output[0]

    print(f"DEBUG training_accuracies_per_run: checkpoint output flag: {output_flag}")
    # Get the output directory
    output_dir = subpath(output_flag, parent=True)

    # Find all model files for this training run
    models = glob_wildcards(
        os.path.join(output_dir, f"mRNN.model.{wildcards.pre_SEED}.{wildcards.pre_epoch}.{wildcards.run}.{{epoch,\\d+}}")
    )

    print(f"DEBUG training_accuracies_per_run: Found {len(models.epoch)} epochs: {models.epoch}")

    # Return all the expected accuracy files for this seed
    accuracy_dir = os.path.join(output_dir, "accuracy_tests")
    result = expand(
        os.path.join(accuracy_dir, f"mRNN.model.{wildcards.pre_SEED}.{wildcards.pre_epoch}.{wildcards.run}.{{epoch}}_acc.txt"),
        epoch=models.epoch
    )
    print(f"DEBUG training_accuracies_per_run: Returning {len(result)} accuracy files")
    return result


###########################################
# AGGREGATE ACCURACY FILES PER RUN (enables parallel execution)
###########################################
rule mrnn_aggregate_training_accuracies_per_run:
    """
    Aggregate accuracy files for all checkpoints from one training run.

    This enables parallel execution: each training run's models can be
    tested as soon as that run completes.
    """
    input:
        checkpoint_flag="results/{expt}/training/{fold}/mRNN/trained/{pre_SEED}.{pre_epoch}.{run}.done.training",
        accuracy_files= training_accuracies_per_run
    output:
        touch("results/{expt}/training/{fold}/mRNN/trained/accuracy_tests/{pre_SEED}.{pre_epoch}.{run}.done")
    shell:
        """
        # print accuracy file names
        echo "Aggregating accuracy files for training run {wildcards.pre_SEED}.{wildcards.pre_epoch}.{wildcards.run}" > {output}
        """


def training_accuracy_flags(wildcards):
    """
    Find all accuracy aggregation flags for all trained models.

    This cascades through:
    1. Best pretrained model selection (checkpoint)
    2. All training runs from those models
    3. Accuracy aggregation for each run
    """
    # First, get the best pretrained models
    best_models = find_best_pretrained_models(wildcards)
    print(f"DEBUG training_accuracy_flags: Got best pretrained models from checkpoint")
    print(f"DEBUG training_accuracy_flags: Best pretrained models: {best_models}")

    # For each best pretrained model and each training run, create the aggregation file path
    all_done_files = []
    for pre_seed, pre_epoch in best_models:
        for run in get_train_runs():
            all_done_files.append(
                f"results/{wildcards.expt}/training/{wildcards.fold}/mRNN/trained/accuracy_tests/{pre_seed}.{pre_epoch}.{run}.done"
            )

    return all_done_files


###########################################
# COPY BEST TRAINED MODELS (when all accuracy files are ready)
###########################################
rule mrnn_get_best_trained_models:
    """
    Select the top N trained models based on test accuracy.
    This is the final step that produces the best model(s) for actual use.
    """
    input:
        # Wait for all training runs to complete testing, which depends on the checkpoing cascade
        training_accuracy_flags
    output:
        best_models_dir=directory("results/{expt}/training/{fold}/mRNN/trained/best_models")
    log:
        "logs/{expt}/training/{fold}/mRNN/mRNN_get_best_trained_models_{fold}.log"
    conda:
        "lnc-datasets"
    params:
        accuracy_path=lambda wc: f"results/{wc.expt}/training/{wc.fold}/mRNN/trained/accuracy_tests",
        n_models=TRAIN_STAGES["train"]["n_select"]
    script:
        "../scripts/mrnn_select_best_models.py"


rule debug_training_dependencies:
    """
    Debug rule to check what files training_accuracies_per_run returns
    """
    input:
        training_accuracies_per_run
    output:
        "debug/training_deps_{fold}_{pre_SEED}_{pre_epoch}_{run}.txt"
    shell:
        """
        echo "Input files for this run:" > {output}
        echo "{input}" >> {output}
        ls -lh $(dirname {input[0]})/../mRNN.model.* >> {output} 2>&1 || echo "No model files found" >> {output}
        """


rule debug_all_training_flags:
    """
    Debug rule to check what training_accuracy_flags returns
    """
    input:
        training_accuracy_flags
    output:
        "debug/all_training_flags_{fold}.txt"
    shell:
        """
        echo "All expected .done files:" > {output}
        printf '%s\\n' {input} >> {output}
        """
