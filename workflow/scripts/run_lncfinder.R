# Load read_ss_cache function
# (assuming execution from snakemake working directory)
source("workflow/utils/ss_utils.R")  # Load read_ss_cache function

if (!requireNamespace("LncFinder", quietly = TRUE)) {
    install.packages("LncFinder", repos = "https://cloud.r-project.org/")
}
if (!requireNamespace("logger", quietly = TRUE)) {
    install.packages("logger", repos = "https://cloud.r-project.org/")
}

run_with_SS <- snakemake@params[['ss']] == 'TRUE'
species <- snakemake@params[['species']]
if (is.null(species) || species == "") {
  species <- "human"
}
run_with_custom_model <- snakemake@params[['use_custom']] == 'TRUE'

# Load libraries
library(LncFinder)
library(seqinr)
library(logger)

# Set a log formatter that does not interpret curly braces
# This solves an error that was raised (don't know exactly which logger line does this)
log_formatter(formatter_sprintf)

# Set up logging
if (exists("snakemake") && length(snakemake@log) > 0) {
  # Set logging to file
  log_file <- snakemake@log[[1]]
  log_appender(appender_console)
  log_info(paste("Logging to file:", log_file))

  # Redirect all output to log file (this captures messages from other packages)
  con <- file(log_file, open = "wt")
  sink(con, split = TRUE)
  sink(con, type = "message")
} else {
  log_info("No log file specified, logging to console only.")
}


if (run_with_custom_model) {
  model_path <- snakemake@params[['model']]
  log_info("Using custom model for lnc_finder, path: ", model_path)

  # Check if model file exists
  if (!file.exists(model_path)) {
    log_error(paste("Model file not found:", model_path))
    stop(paste("Model file not found:", model_path))
  }

  # Create an isolated environment and load RData there
  tmp_env <- new.env()
  load(model_path, envir = tmp_env)

  # Extract the first variable from the environment
  model_obj_name <- ls(tmp_env, sorted=FALSE)[2]
  log_info(paste("Model object name:", model_obj_name))
  svm.model <- tmp_env[[model_obj_name]]

  # Extract the second variable from the environment
  freq_obj_name <- ls(tmp_env, sorted=FALSE)[1]
  log_info(paste("Frequencies object name:", freq_obj_name))
  freqs <- tmp_env[[freq_obj_name]]

  log_info(paste("Loaded custom model of class:", class(svm.model)))

  # Log the model formula to debug feature mismatches
  if (inherits(svm.model, "svm.formula")) {
    log_info(paste("Model formula:", paste(svm.model$formula, collapse=" ")))
  }
} else {
  log_info(paste("Using species model for lnc_finder:", species))
  svm.model <- species  # Set default model for species
  freqs <- species  # Set default frequencies for species
}

# Some debugging info
log_info(paste("Running 'run_lncfinder.R' in directory:", getwd()))
log_info(paste("Using threads:", snakemake@threads))
log_info(paste("Using SS features:", run_with_SS))

# Read sequences
fasta_file <- snakemake@input[['fasta']]
log_info(paste("Reading sequences from:", fasta_file))
if (run_with_SS) {
  input_sequences <- read_ss_cache(fasta_file)
} else {
  input_sequences <- read.fasta(file = fasta_file)
}

if (length(input_sequences) == 0) {
  log_error("No sequences found in the input FASTA file.")
  stop("No sequences found in the input FASTA file.")
}


log_info("Running lnc_finder...")
log_info(paste("Using SS features:", run_with_SS, "| Format:", ifelse(run_with_SS, "SS", "DNA")))

# Run lnc_finder with error handling
tryCatch({
  results <- lnc_finder(
    Sequences = input_sequences,
    SS.features = run_with_SS,
    format = ifelse(run_with_SS, "SS", "DNA"),
    frequencies.file = freqs,
    svm.model = svm.model,
    parallel.cores = snakemake@threads
  )
}, error = function(e) {
  # If error mentions missing features, provide guidance
  error_msg <- conditionMessage(e)
  if (grepl("not found|object.*not found", error_msg, ignore.case = TRUE)) {
    log_error("Feature mismatch error detected:")
    log_error(paste("The custom model expects features that were not computed."))
    log_error(paste("This likely happens when:"))
    log_error(paste("  - Running with SS=FALSE but model was trained with SS=TRUE"))
    log_error(paste("  - Running with SS=TRUE but model was trained with SS=FALSE"))
    log_error(paste("Error details:", error_msg))
  } else {
    log_error(paste("Error during lnc_finder prediction:", error_msg))
  }
  stop(e)
})

write.table(results, file = snakemake@output[['out']], sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)
log_info(paste("Results saved to:", snakemake@output[['out']]))

if (length(snakemake@log) > 0) {
  # Close the sink
  sink(type = "message")
  sink()
  close(con)
}
log_info("Script completed successfully.")
