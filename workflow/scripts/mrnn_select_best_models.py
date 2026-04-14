import os
import sys
from pathlib import Path

import pandas as pd

# Enable imports from workflow directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
from utils.logging_utils import log_job_completion, setup_snakemake_logging

logger = setup_snakemake_logging(snakemake, script_name=__file__)

accuracy_path = snakemake.params["accuracy_path"]
best_models_dir = snakemake.output["best_models_dir"]
n_models = int(snakemake.params.get("n_models", 6))


# TODO: Outsource to utility function
def parse_mrnn_performance(file_path):
    results = {"metrics": [], "confusion_matrix": []}

    logger.debug(f"Parsing performance file: {file_path}")

    with open(file_path, "r") as file:
        lines = file.readlines()

        # Parse metrics
        for line in lines[:3]:  # First three lines contain metrics
            parts = line.strip().split("\t")
            weights = parts[0]
            metric = parts[1]
            value = float(parts[2])
            results["metrics"].append(
                {"weights": weights, "metric": metric, "value": value}
            )

        # Parse confusion matrix
        for line in lines[3:]:  # Last two lines contain confusion matrix
            row = list(map(int, line.strip().split("\t")))
            results["confusion_matrix"].extend(row)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(results["metrics"])

    # Convert confusion matrix to DataFrame
    # REF: https://github.com/hendrixlab/mRNN/blob/0b7e98b48385f1ee95bfe5a2f5f86c6ae4defbe8/evaluate.py#L310C5-L310C25
    confusion_labels = ["TN", "FP", "FN", "TP"]
    confusion_matrix_df = pd.DataFrame(
        [results["confusion_matrix"]], columns=confusion_labels
    )

    expanded_metrics_df = metrics_df.pivot(
        index="weights", columns="metric", values="value"
    ).reset_index()
    expanded_metrics_df = pd.concat([expanded_metrics_df, confusion_matrix_df], axis=1)

    logger.debug(f"Successfully parsed {len(metrics_df)} metrics from {file_path}")

    return expanded_metrics_df


logger.info(f"Starting model selection from: {accuracy_path}")
logger.info(f"Selecting top {n_models} models based on accuracy")

# Iterate through all files in the directory
all_metrics = pd.DataFrame()
file_count = 0
for file_name in os.listdir(accuracy_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(accuracy_path, file_name)
        logger.info(f"Processing file: {file_path}")
        expanded_metrics_df = parse_mrnn_performance(file_path)
        all_metrics = pd.concat([all_metrics, expanded_metrics_df], ignore_index=True)
        file_count += 1

logger.info(f"Processed {file_count} performance files")

# Select top models with different synthetic datasets, based on Accuracy
all_metrics["synthetic_dataset"] = all_metrics["weights"].str.rsplit(".", n=4).str[1]
all_metrics.sort_values(by="ACC", ascending=False, inplace=True)
top_models = all_metrics.groupby("synthetic_dataset").first().head(n_models)

logger.info(f"Selected top {len(top_models)} models")
logger.debug(f"Top model accuracies: {top_models['ACC'].tolist()}")

# Get files and copy to a new directory "best_models"
os.makedirs(best_models_dir, exist_ok=True)
logger.info(f"Created output directory: {best_models_dir}")

for weights_file in top_models["weights"]:
    filename = os.path.basename(weights_file)
    dst_file = os.path.join(best_models_dir, filename)
    os.system(f"cp {weights_file} {dst_file}")
    logger.info(f"Copied {weights_file} to {dst_file}")

logger.info("Model selection completed successfully")

log_job_completion(logger)
