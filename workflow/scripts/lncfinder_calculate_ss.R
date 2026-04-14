source("../utils/ss_utils.R")  # Load read_ss_cache function

if (!requireNamespace("LncFinder", quietly = TRUE)) {
    install.packages("LncFinder", repos = "https://cloud.r-project.org/")
}
if (!requireNamespace("logger", quietly = TRUE)) {
    install.packages("logger", repos = "https://cloud.r-project.org/")
}


ss_file <- snakemake@output[['out']]
fasta_file <- snakemake@input[['fasta']]

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

Seqs <- read.fasta(file = fasta_file)

log_info(paste("Running RNAfold on sequences:", fasta_file))
RNAfold.path <- file.path(Sys.getenv("CONDA_PREFIX"), "bin", "RNAfold")
SS.seq <- run_RNAfold(
  Seqs,
  RNAfold.path = RNAfold.path,
  parallel.cores = snakemake@threads
)

write.fasta(SS.seq, names = names(SS.seq), file.out = ss_file)

log_info(paste("Wrote secondary structure predictions to:", ss_file))
if (length(snakemake@log) > 0) {
  # Close the sink
  sink(type = "message")
  sink()
  close(con)
}
log_info("Script completed successfully.")
