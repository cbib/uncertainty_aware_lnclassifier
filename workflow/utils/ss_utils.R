# Define readung function
read_ss_cache <- function(file) {
  log_info(paste("Reading cached secondary structure from:", file))

  # Read the file and parse sequence and secondary structure
  lines <- readLines(file)
  headers = lines[seq(1, length(lines), by = 2)]
  sequences = lines[seq(2, length(lines), by = 2)]
  sequences <- trimws(sequences)

  col_names <- sub("^>", "", headers)

  # Parse secondary structure and MFE from sequences
  # Pattern: structure (dots/parens) followed by optional MFE value
  pattern <- "^([acgtun]+)([\\.\\(\\)]+)(-?\\d+(?:\\.\\d+)?)$"
  m <- regexec(pattern, sequences, perl = TRUE)
  parts = regmatches(sequences, m)

  n <- length(sequences)
  rna <- character(n)
  ss <- character(n)
  mfe <- numeric(n)

  for (i in seq_len(n)) {
    if (length(parts[[i]]) == 4) {
      rna[i] <- parts[[i]][2]
      ss[i] <- parts[[i]][3]
      mfe[i] <- as.numeric(parts[[i]][4])
    } else {
      log_warn(paste("No match for sequence:", trimws(headers[i])))
      print(paste("No match for sequence:", trimws(headers[i])))
      rna[i] <- NA
      ss[i] <- NA
      mfe[i] <- NA
    }
 }
   #df <- data.frame(rna = rna, ss = ss, mfe = mfe, stringsAsFactors = FALSE)
  #colnames(df) <- headers
  #df<-as.data.frame(matrix(c(rna, ss, mfe), nrow = 3, byrow = TRUE))
  #colnames(df) <- headers
  #df <- data.frame(rna = rna, ss = ss, mfe = mfe, stringsAsFactors = FALSE)
  #rownames(df) <- row_names
  df <- data.frame(rna = rna, ss = ss, mfe = mfe, stringsAsFactors = FALSE)
  df <- as.data.frame(t(df))  # transpose: now rows are rna/ss/mfe, columns are sequences
  colnames(df) <- col_names

  # Verbose output after parsing
  log_info(paste("Parsed", n, "sequences."))
  log_info(paste("Number of successful parses:", sum(!is.na(df["rna", ]))))
  log_info(paste("Number of failed parses:", sum(is.na(df["rna", ]))))
  max_ids <- 5
  shown_ids <- head(col_names, max_ids)
  more_ids <- if (length(col_names) > max_ids) sprintf("... and %d more", length(col_names) - max_ids) else ""
  log_info(paste("Column names (sequence IDs):", paste(shown_ids, collapse = ", "), more_ids))

  return(df)
}
