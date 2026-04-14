# Load read_ss_cache function
# (assuming execution from snakemake working directory)
source("workflow/utils/ss_utils.R")

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

# Load libraries
library(LncFinder)
library(seqinr)
library(logger)

# Set a log formatter that does not interpret curly braces
# This solves an error that was raised (don't know exactly which logger line does this)
log_formatter(formatter_sprintf)

if (length(snakemake@log) > 0) {
    # Set logging to file
    log_file = snakemake@log[[1]]
    #log_appender(appender_tee(log_file))
    log_appender(appender_console)
    log_info(paste("Logging to file:", log_file))

    # Redirect all output to log file (this captures messages from other packages)
    con <- file(log_file, open = "wt")
    sink(con, split = TRUE)
    sink(con, type = "message")
} else {
    log_info("No log file specified, logging to console only.")
}

# Some debugging info
log_info(paste("Running 'train_lncfinder.R' in directory:", getwd()))
log_info(paste("Using threads:", snakemake@threads))
log_info(paste("Using SS features:", run_with_SS))



log_info(paste("Reading cds sequences from:", snakemake@input[['cds']]))
cds_seqs <- read.fasta(file = snakemake@input[['cds']])

if (run_with_SS) {
    # Read SS sequences
    log_info(paste("Reading lncRNA secondary structures from:", snakemake@input[['lnc']]))
    lnc_seqs <- read_ss_cache(file = snakemake@input[['lnc']])

    log_info(paste("Reading pcRNA secondary structures from:", snakemake@input[['pc']]))
    pc_seqs <- read_ss_cache(file = snakemake@input[['pc']])

    format <- "SS"
} else {
    # Read FASTA sequences
    log_info(paste("Reading lncRNA sequences from:", snakemake@input[['lnc']]))
    lnc_seqs <- read.fasta(file = snakemake@input[['lnc']])

    log_info(paste("Reading pcRNA sequences from:", snakemake@input[['pc']]))
    pc_seqs <- read.fasta(file = snakemake@input[['pc']])

    format <- "DNA"
}

log_info(paste("Sequence format set to", format))

# Informative debug output for input objects
log_info("--- INPUT SUMMARY ---")
# CDS
if (is.list(cds_seqs)) {
    log_info(sprintf("cds_seqs: %d sequences", length(cds_seqs)))
    if (length(cds_seqs) > 0) {
        log_info(sprintf("First CDS name: %s", names(cds_seqs)[1]))
        log_info(sprintf("Last CDS name: %s", names(cds_seqs)[length(cds_seqs)]))
        first_cds_str <- paste(head(cds_seqs[[1]], 40), collapse="")
        last_cds_str <- paste(head(cds_seqs[[length(cds_seqs)]], 40), collapse="")
        log_info(sprintf("First CDS preview: %s...", substr(first_cds_str, 1, 40)))
        log_info(sprintf("Last CDS preview: %s...", substr(last_cds_str, 1, 40)))
    }
}
# lncRNA
if (run_with_SS && is.data.frame(lnc_seqs)) {
    log_info(sprintf("lnc_seqs: %d sequences (columns)", ncol(lnc_seqs)))
    log_info(sprintf("Row names: %s", paste(rownames(lnc_seqs), collapse=", ")))
    if (ncol(lnc_seqs) > 0) {
        log_info(sprintf("First lncRNA name: %s", colnames(lnc_seqs)[1]))
        log_info(sprintf("Last lncRNA name: %s", colnames(lnc_seqs)[ncol(lnc_seqs)]))
        log_info(sprintf("First lncRNA rna: %s...", substr(lnc_seqs["rna",1], 1, 40)))
        log_info(sprintf("First lncRNA ss: %s...", substr(lnc_seqs["ss",1], 1, 40)))
        log_info(sprintf("First lncRNA mfe: %s", substr(lnc_seqs["mfe",1], 1, 40)))
    }
} else if (!run_with_SS && is.list(lnc_seqs)) {
    log_info(sprintf("lnc_seqs: %d sequences", length(lnc_seqs)))
    if (length(lnc_seqs) > 0) {
        log_info(sprintf("First lncRNA name: %s", names(lnc_seqs)[1]))
        log_info(sprintf("Last lncRNA name: %s", names(lnc_seqs)[length(lnc_seqs)]))
        first_lnc_str <- paste(head(lnc_seqs[[1]], 40), collapse="")
        last_lnc_str <- paste(head(lnc_seqs[[length(lnc_seqs)]], 40), collapse="")
        log_info(sprintf("First lncRNA preview: %s...", substr(first_lnc_str, 1, 40)))
        log_info(sprintf("Last lncRNA preview: %s...", substr(last_lnc_str, 1, 40)))
    }
}
# pcRNA
if (run_with_SS && is.data.frame(pc_seqs)) {
    log_info(sprintf("pc_seqs: %d sequences (columns)", ncol(pc_seqs)))
    log_info(sprintf("Row names: %s", paste(rownames(pc_seqs), collapse=", ")))
    if (ncol(pc_seqs) > 0) {
        log_info(sprintf("First pcRNA name: %s", colnames(pc_seqs)[1]))
        log_info(sprintf("Last pcRNA name: %s", colnames(pc_seqs)[ncol(pc_seqs)]))
        log_info(sprintf("First pcRNA rna: %s...", substr(pc_seqs["rna",1], 1, 40)))
        log_info(sprintf("First pcRNA ss: %s...", substr(pc_seqs["ss",1], 1, 40)))
        log_info(sprintf("First pcRNA mfe: %s", substr(pc_seqs["mfe",1], 1, 40)))
    }
} else if (!run_with_SS && is.list(pc_seqs)) {
    log_info(sprintf("pc_seqs: %d sequences", length(pc_seqs)))
    if (length(pc_seqs) > 0) {
        log_info(sprintf("First pcRNA name: %s", names(pc_seqs)[1]))
        log_info(sprintf("Last pcRNA name: %s", names(pc_seqs)[length(pc_seqs)]))
        first_pc_str <- paste(head(pc_seqs[[1]], 40), collapse="")
        last_pc_str <- paste(head(pc_seqs[[length(pc_seqs)]], 40), collapse="")
        log_info(sprintf("First pcRNA preview: %s...", substr(first_pc_str, 1, 40)))
        log_info(sprintf("Last pcRNA preview: %s...", substr(last_pc_str, 1, 40)))
    }
}
log_info("--- END INPUT SUMMARY ---")

# Helper function to trim 'N' from start and end of character vectors
trim_N <- function(seq) {
    # Collapse to string if needed
    if (is.list(seq)) seq <- unlist(seq)
    s <- paste(seq, collapse = "")
    s <- sub("^N+", "", s)
    s <- sub("N+$", "", s)
    # Return as character vector (split to single chars)
    strsplit(s, "")[[1]]
}
# Input validation functions with additional debug output
is_valid_dna_list <- function(x) {
    valid <- is.list(x) && length(x) > 0 && all(sapply(x, function(seq) is.character(seq) && length(seq) > 0))
    if (!is.list(x)) log_info("is_valid_dna_list: Input is not a list.")
    if (length(x) == 0) log_info("is_valid_dna_list: List is empty.")
    if (is.list(x)) {
        for (i in seq_along(x)) {
            if (!is.character(x[[i]])) log_info(sprintf("is_valid_dna_list: Element %d is not character.", i))
            if (length(x[[i]]) == 0) log_info(sprintf("is_valid_dna_list: Element %d is empty.", i))
        }
    }
    log_info(sprintf("is_valid_dna_list: Result = %s", valid))
    return(valid)
}
is_valid_ss_df <- function(x) {
    has_rows <- is.data.frame(x) && nrow(x) > 0 && ncol(x) > 0
    has_rna_ss_mfe_rows <- all(c("rna", "ss", "mfe") %in% rownames(x))
    has_rna_ss_mfe_cols <- all(c("rna", "ss", "mfe") %in% colnames(x))
    valid <- is.data.frame(x) && has_rows && (has_rna_ss_mfe_rows || has_rna_ss_mfe_cols)
    if (!is.data.frame(x)) log_info("is_valid_ss_df: Input is not a data.frame.")
    if (is.data.frame(x) && nrow(x) == 0) log_info("is_valid_ss_df: Data frame has zero rows.")
    if (is.data.frame(x) && ncol(x) == 0) log_info("is_valid_ss_df: Data frame has zero columns.")
    if (is.data.frame(x) && !has_rna_ss_mfe_rows && !has_rna_ss_mfe_cols) {
        log_info("is_valid_ss_df: Data frame does not have rna, ss, mfe in row or column names.")
        log_info(sprintf("Row names: %s", paste(rownames(x), collapse=", ")))
        log_info(sprintf("Col names: %s", paste(colnames(x), collapse=", ")))
    }
    log_info(sprintf("is_valid_ss_df: Result = %s", valid))
    return(valid)
}

# Validate input formats
if (run_with_SS) {
    if (!is_valid_dna_list(cds_seqs)) {
        stop("CDS input is not a non-empty list of character vectors (required for DNA input in SS mode)!")
    }
    if (!is_valid_ss_df(lnc_seqs)) {
        stop("lncRNA input is not a valid SS data.frame (should be output of read_ss_cache, with rna/ss/mfe rows or columns)!")
    }
    if (!is_valid_ss_df(pc_seqs)) {
        stop("pcRNA input is not a valid SS data.frame (should be output of read_ss_cache, with rna/ss/mfe rows or columns)!")
    }
} else {
    if (!is_valid_dna_list(cds_seqs)) {
        stop("CDS input is not a non-empty list of character vectors (required for DNA input)!")
    }
    if (!is_valid_dna_list(lnc_seqs)) {
        stop("lncRNA input is not a non-empty list of character vectors (required for DNA input)!")
    }
    if (!is_valid_dna_list(pc_seqs)) {
        stop("pcRNA input is not a non-empty list of character vectors (required for DNA input)!")
    }
}
# TODO: If I trim external IDs, I should trim the SS sequences too, which is more complex.
# This may actually be causing some of the errors I see.

#log_info("Trimming 'N' characters from the start and end of sequences...")
# Apply trimming to cds (used to calculate kmer frequencies)
# This is an educated choice which should not bias the frequencies
#cds_seqs <- lapply(cds_seqs, trim_N)
#pc_seqs <- lapply(pc_seqs, trim_N)
#lnc_seqs <- lapply(lnc_seqs, trim_N)

# NOTE: we do this here instead ot letting the make_frequencies function fail
# I would assume that make_frequencies ignores the illegal characters
# however, if the very first sequence is illegal, the underlying freq.res object
# might never be created, raising an error
log_info("Filtering CDS sequences for illegal characters...")
# Remove CDS sequences containing characters other than acgtuACGTU
cds_seqs <- cds_seqs[sapply(cds_seqs, function(seq) {
    s <- paste(seq, collapse = "")
    !grepl("[^ACGTUacgtu]", s)
})]
log_info(sprintf("Retained %d CDS sequences after filtering", length(cds_seqs)))

if (length(cds_seqs) == 0) {
    stop("No valid CDS sequences remaining after filtering for illegal characters!")
}

log_info("Calculating k-mer frequencies...")
# Get frequencies
frequencies <- make_frequencies(
    cds.seq = cds_seqs,
    mRNA.seq = pc_seqs,
    lncRNA.seq = lnc_seqs,
    SS.features = run_with_SS,
    cds.format = "DNA",
    lnc.format = format,
    check.cds = FALSE,
    ignore.illegal = TRUE
)

log_info("Building the model...")
model <- build_model(
  lncRNA.seq = lnc_seqs,
  mRNA.seq = pc_seqs,
  frequencies.file = frequencies,
  SS.features = run_with_SS,
  lncRNA.format = format,
  mRNA.format = format,
  parallel.cores = snakemake@threads,
  folds.num = 10,
  seed = 1,
  gamma.range = (2^seq(-5, 0, 1)),
  cost.range = c(1, 4, 8, 16, 24, 32),
)

# Save objects to RData file
# An RData file shipped with LncFinder contains the following objects:
## Internal.Human -> Reference frequencies
## Human.mod -> Model trained with SS features
## Human.mod.no_ss -> Model trained without SS features
# With the current implementation, each RData file only contains one of the two models:
Internal.Human <- frequencies
if (run_with_SS) {
    Human.mod <- model
    save(Internal.Human, Human.mod, file=snakemake@output[['model']])
} else {
    Human.mod.no_ss <- model
    save(Internal.Human, Human.mod.no_ss, file=snakemake@output[['model']])
}

log_info(paste("Model saved to: ", snakemake@output[['model']]))

if (length(snakemake@log) > 0) {
    # Close the sink
    sink(type = "message")
    sink()
    close(con)
}
log_info("Script completed successfully.")
