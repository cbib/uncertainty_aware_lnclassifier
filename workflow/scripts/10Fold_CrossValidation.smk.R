# Redirect all output to the Snakemake log file
log_file <- snakemake@log[[1]]
log_conn <- file(log_file, open = "wt")
sink(log_conn, type = "output")
sink(log_conn, type = "message")

if (!require("ROCR", quietly = TRUE)) {
    options(repos = c(CRAN = "https://cran.r-project.org"))
    install.packages("ROCR")
}

library(ROCR)

# Read input file from Snakemake
data=read.table(file=snakemake@input[["training_data"]],header=T,sep="\t")
cutoff_precision = snakemake@params[["cutoff_precision"]]
print(paste("Using cutoff precision:", cutoff_precision))

# CRITICAL FIX: Randomize the data before CV
set.seed(42)  # For reproducibility
random_idx <- sample(1:nrow(data), replace=FALSE)
data <- data[random_idx, ]
attach(data)
names(data)

# Change working directory to output directory
output_prefix=snakemake@params[["output_prefix"]]
setwd(output_prefix)

#total 20000
total_lines <- nrow(data)
print(total_lines)
split <- floor(total_lines / 10)

# Store predictions
all_predictions <- list()
all_labels <- list()

for (i in 1:10) {
    split_start <- ((i-1)*split + 1)
    split_end <- (i*split)
    print(paste("Fold ", i, ": from ", split_start, " to ", split_end))
    fold_idx <- seq(split_start, split_end)

    vlabel = Label[-fold_idx]
    vmrna = mRNA[-fold_idx]
    vorf = ORF[-fold_idx]
    vfickett = Fickett[-fold_idx]
    vhexamer = Hexamer[-fold_idx]

    mylogit <- glm(vlabel ~ vmrna + vorf + vfickett + vhexamer, family=binomial(link="logit"), na.action=na.pass, control=glm.control(maxit=100))
    test <- data.frame(vmrna = mRNA[fold_idx], vorf = ORF[fold_idx], vfickett = Fickett[fold_idx], vhexamer = Hexamer[fold_idx], vlabel=Label[fold_idx])
    test$prob <- predict(mylogit, newdata=test, type="response")

    # Report statistics on predictions
        all_predictions[[i]] <- test$prob
    all_labels[[i]] <- Label[fold_idx]

    # Diagnostics
    class_dist <- sum(Label[fold_idx] == 1) / length(Label[fold_idx])
    print(paste("  Probability range:",
                round(min(test$prob, na.rm=T), 4), "-",
                round(max(test$prob, na.rm=T), 4)))
    print(paste("  Mean prob:", round(mean(test$prob, na.rm=T), 4),
                "| Class distribution (% coding):", round(class_dist*100, 2)))
    # Clip probabilities to avoid exact 0 or 1 values that cause ROCR errors
    #test$prob <- pmax(pmin(test$prob, 0.9999), 0.0001)
    output = cbind("mRNA"=test$vmrna, "ORF"=test$vorf, "Fickett"=test$vfickett, "Hexamer"=test$vhexamer, "Label"=test$vlabel, "Prob"=test$prob)
    write.table(output, file=paste0("test", i, ".xls"), quote=F, sep="\t", row.names=ID[fold_idx])
}


#ROC
test1=read.table(file="test1.xls",header=T,sep="\t")
test2=read.table(file="test2.xls",header=T,sep="\t")
test3=read.table(file="test3.xls",header=T,sep="\t")
test4=read.table(file="test4.xls",header=T,sep="\t")
test5=read.table(file="test5.xls",header=T,sep="\t")
test6=read.table(file="test6.xls",header=T,sep="\t")
test7=read.table(file="test7.xls",header=T,sep="\t")
test8=read.table(file="test8.xls",header=T,sep="\t")
test9=read.table(file="test9.xls",header=T,sep="\t")
test10=read.table(file="test10.xls",header=T,sep="\t")


Response = list(test1$Prob,test2$Prob,test3$Prob,test4$Prob,test5$Prob,test6$Prob,test7$Prob,test8$Prob,test9$Prob,test10$Prob)
Labls = list(test1$Label,test2$Label,test3$Label,test4$Label,test5$Label,test6$Label,test7$Label,test8$Label,test9$Label,test10$Label)
ROCR_data = list(predictions=Response,Labels=Labls)
pred <- prediction(ROCR_data$predictions, ROCR_data$Labels)
#perf <- performance(pred,"auc")
#avergae AUC = 0.9927


pdf("Figure_3.pdf")
par(mfrow=c(2,2),mar=c(5,4,2,2),cex.axis=1.2, cex.lab=1.2)
#ROC curve
#pdf("Human_10fold.ROC.pdf")
perf <- performance(pred,"tpr","fpr")
plot(perf,col="blue",lty=3,xlab="1-Specificity",ylab="Sensitivity",ylim=c(0.7,1),xlim=c(0,0.3),main="",cex.axis=1.5,cex.label=1.5)	#AUC = 0.9927
plot(perf,lwd=2,avg="vertical",add=TRUE,col="red",xlab="1-specificity",ylab="sensitivity",main="",cex.axis=1.2,cex.label=1.2)
abline(v=0,lty="dashed",lwd=0.5)
abline(h=1.0,lty="dashed",lwd=0.5)
abline(v=0.05,lty="dashed",lwd=0.5)
abline(h=0.95,lty="dashed",lwd=0.5)
#dev.off()

#precision
#pdf("Human_10fold.precision_vs_recall.pdf")
d=performance(pred,measure="prec", x.measure="rec")
plot(d,col="blue",lty=3,xlab="Recall (TPR)",ylab="Precision (PPV)",xlim=c(0.7,1),ylim=c(0.7,1),cex.axis=1.2,cex.label=1.2)
plot(d,lwd=2,avg="vertical",col="red",xlab="Recall (TPR)",ylab="Precision (PPV)",add=T,cex.axis=1.2,cex.label=1.2)
abline(v=1.0,lty="dashed",lwd=0.5)
abline(h=1.0,lty="dashed",lwd=0.5)
abline(v=0.95,lty="dashed",lwd=0.5)
abline(h=0.95,lty="dashed",lwd=0.5)
#dev.off()


#Accuracy
#pdf("Human_10fold.Accuracy.pdf")
perf <- performance(pred,"acc")
plot(perf,col="blue",lty=3,xlab="Coding probability cutoff",ylab="Accuracy",ylim=c(0.7,1),cex.axis=1.2,cex.label=1.2)
plot(perf,lwd=2,avg="vertical",add=TRUE,col="red",cex.axis=1.2,cex.label=1.2)
abline(h=1,lty="dashed",lwd=0.5)
abline(h=0.95,lty="dashed",lwd=0.5)
#dev.off()


#sensitivity vs specificity
pred <- prediction(ROCR_data$predictions, ROCR_data$Labels)
S <- performance(pred,measure="sens")
P <- performance(pred,measure="spec")
#pdf("Human_10fold_sens_vs_spec.pdf")
plot(S,col="blue",lty=3,ylab="Performance",xlab="Coding Probability Cutoff",ylim=c(0.8,1),cex.axis=1.2,cex.label=1.2)
plot(S,lwd=2,avg="vertical",add=TRUE,col="blue")
plot(P,col="red",lty=3, add=TRUE,)
plot(P,lwd=2,avg="vertical",add=TRUE,col="red")
abline(h=0.966,lty="dashed",lwd=0.5)
abline(v=0.364,lty="dashed",lwd=0.5)
legend(0.4,0.85,col=c("blue","red"),lwd=2,legend=c("Sensitivity","Specificity"))
dev.off()

# Try to compute AUC
perf_auc <- tryCatch({
    performance(pred, measure="auc")
}, error=function(e) {
    print("Could not compute AUC due to probability distribution")
    return(NULL)
})

if(!is.null(perf_auc)) {
    auc_vals <- unlist(perf_auc@y.values)
    print(paste("Mean AUC:", round(mean(auc_vals), 4)))
    print(paste("Individual fold AUCs:", paste(round(auc_vals, 4), collapse=", ")))
}

pred <- prediction(all_predictions, all_labels)

# ============================================================================
# PART 2: OPTIMAL CUTOFF SELECTION (Multiple Strategies)
# ============================================================================

# Aggregate all predictions and labels for cutoff analysis
all_pred_vec <- unlist(all_predictions)
all_label_vec <- unlist(all_labels)

# Define cutoff range to test
cutoffs_to_test <- seq(0.0, 1.0, by=cutoff_precision)

# Initialize results data frame
cutoff_analysis <- data.frame(
    Cutoff = cutoffs_to_test,
    TP = numeric(length(cutoffs_to_test)),
    FP = numeric(length(cutoffs_to_test)),
    TN = numeric(length(cutoffs_to_test)),
    FN = numeric(length(cutoffs_to_test)),
    Sensitivity = numeric(length(cutoffs_to_test)),
    Specificity = numeric(length(cutoffs_to_test)),
    Precision = numeric(length(cutoffs_to_test)),
    Accuracy = numeric(length(cutoffs_to_test)),
    F1_Score = numeric(length(cutoffs_to_test)),
    MCC = numeric(length(cutoffs_to_test)),
    Youden = numeric(length(cutoffs_to_test))
)

# Calculate metrics for each cutoff
for (i in 1:length(cutoffs_to_test)) {
    cutoff <- cutoffs_to_test[i]

    # Confusion matrix
    tp <- sum(all_pred_vec >= cutoff & all_label_vec == 1)
    tn <- sum(all_pred_vec <  cutoff & all_label_vec == 0)
    fp <- sum(all_pred_vec >= cutoff & all_label_vec == 0)
    fn <- sum(all_pred_vec <  cutoff & all_label_vec == 1)

    tp <- as.numeric(tp); tn <- as.numeric(tn)
    fp <- as.numeric(fp); fn <- as.numeric(fn)

    # Sensitivity
    if (tp + fn == 0) {
    sensitivity <- NA
    } else {
    sensitivity <- tp / (tp + fn)
    }

    # Specificity
    if (tn + fp == 0) {
    specificity <- NA
    } else {
    specificity <- tn / (tn + fp)
    }

    # Precision
    if (tp + fp == 0) {
    precision <- NA
    } else {
    precision <- tp / (tp + fp)
    }

    # Accuracy
    if ((tp + tn + fp + fn) == 0) {
    accuracy <- NA
    } else {
    accuracy <- (tp + tn) / (tp + tn + fp + fn)
    }

    # F1 (guard against NA)
    if (is.na(precision) || is.na(sensitivity) || (precision + sensitivity) == 0) {
    f1 <- NA
    } else {
    f1 <- 2 * (precision * sensitivity) / (precision + sensitivity)
    }

    # MCC (with overflow and NA protection)
    mcc_num   <- (tp * tn) - (fp * fn)
    mcc_denom <- (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if (is.na(mcc_denom) || mcc_denom <= 0) {
    mcc <- NA
    } else {
    mcc <- mcc_num / sqrt(mcc_denom)
    }

    # Youden (only if sens/spec not NA)
    if (is.na(sensitivity) || is.na(specificity)) {
    youden <- NA
    } else {
    youden <- sensitivity + specificity - 1
    }

    cutoff_analysis$Sensitivity[i] <- sensitivity
    cutoff_analysis$Specificity[i] <- specificity
    cutoff_analysis$Precision[i]   <- precision
    cutoff_analysis$Accuracy[i]    <- accuracy
    cutoff_analysis$F1_Score[i]    <- f1
    cutoff_analysis$MCC[i]         <- mcc
    cutoff_analysis$Youden[i]      <- youden

}

# ============================================================================
# PART 3: FIND OPTIMAL CUTOFFS BY DIFFERENT STRATEGIES
# ============================================================================

optimal_cutoffs <- data.frame(Strategy = character(), Cutoff = numeric(),
                              Sensitivity = numeric(), Specificity = numeric(),
                              Accuracy = numeric(), Score = numeric())

# Strategy 1: Maximum Accuracy
idx_max_acc <- which.max(cutoff_analysis$Accuracy)
optimal_cutoffs <- rbind(optimal_cutoffs, data.frame(
    Strategy = "Max Accuracy",
    Cutoff = cutoff_analysis$Cutoff[idx_max_acc],
    Sensitivity = cutoff_analysis$Sensitivity[idx_max_acc],
    Specificity = cutoff_analysis$Specificity[idx_max_acc],
    Accuracy = cutoff_analysis$Accuracy[idx_max_acc],
    Score = cutoff_analysis$Accuracy[idx_max_acc]
))

# Strategy 2: Maximum Youden Index
idx_max_youden <- which.max(cutoff_analysis$Youden)
optimal_cutoffs <- rbind(optimal_cutoffs, data.frame(
    Strategy = "Maximum Youden",
    Cutoff = cutoff_analysis$Cutoff[idx_max_youden],
    Sensitivity = cutoff_analysis$Sensitivity[idx_max_youden],
    Specificity = cutoff_analysis$Specificity[idx_max_youden],
    Accuracy = cutoff_analysis$Accuracy[idx_max_youden],
    Score = cutoff_analysis$Youden[idx_max_youden]
))

# Strategy 3: Maximum F1 Score
idx_max_f1 <- which.max(cutoff_analysis$F1_Score)
optimal_cutoffs <- rbind(optimal_cutoffs, data.frame(
    Strategy = "Maximum F1",
    Cutoff = cutoff_analysis$Cutoff[idx_max_f1],
    Sensitivity = cutoff_analysis$Sensitivity[idx_max_f1],
    Specificity = cutoff_analysis$Specificity[idx_max_f1],
    Accuracy = cutoff_analysis$Accuracy[idx_max_f1],
    Score = cutoff_analysis$F1_Score[idx_max_f1]
))

# Strategy 4: Maximum MCC
idx_max_mcc <- which.max(cutoff_analysis$MCC)
optimal_cutoffs <- rbind(optimal_cutoffs, data.frame(
    Strategy = "Maximum MCC",
    Cutoff = cutoff_analysis$Cutoff[idx_max_mcc],
    Sensitivity = cutoff_analysis$Sensitivity[idx_max_mcc],
    Specificity = cutoff_analysis$Specificity[idx_max_mcc],
    Accuracy = cutoff_analysis$Accuracy[idx_max_mcc],
    Score = cutoff_analysis$MCC[idx_max_mcc]
))

# Strategy 5: Balanced Sensitivity/Specificity (closest to 0.5 difference)
sensitivity_specificity_diff <- abs(cutoff_analysis$Sensitivity - cutoff_analysis$Specificity)
idx_balanced <- which.min(sensitivity_specificity_diff)
optimal_cutoffs <- rbind(optimal_cutoffs, data.frame(
    Strategy = "Balanced Sens/Spec",
    Cutoff = cutoff_analysis$Cutoff[idx_balanced],
    Sensitivity = cutoff_analysis$Sensitivity[idx_balanced],
    Specificity = cutoff_analysis$Specificity[idx_balanced],
    Accuracy = cutoff_analysis$Accuracy[idx_balanced],
    Score = sensitivity_specificity_diff[idx_balanced]
))

# Strategy 6: 0.5 Default (classical threshold)
idx_default <- which.min(abs(cutoff_analysis$Cutoff - 0.5))
optimal_cutoffs <- rbind(optimal_cutoffs, data.frame(
    Strategy = "Default (0.5)",
    Cutoff = cutoff_analysis$Cutoff[idx_default],
    Sensitivity = cutoff_analysis$Sensitivity[idx_default],
    Specificity = cutoff_analysis$Specificity[idx_default],
    Accuracy = cutoff_analysis$Accuracy[idx_default],
    Score = cutoff_analysis$Youden[idx_default]
))

# ============================================================================
# PART 4: VISUALIZE CUTOFF ANALYSIS
# ============================================================================

pdf("cutoff_analysis.pdf", width=14, height=10)

# Plot 1: Accuracy, Sensitivity, Specificity vs Cutoff
par(mfrow=c(2,3))
plot(cutoff_analysis$Cutoff, cutoff_analysis$Accuracy,
     type="l", lwd=2, xlab="Cutoff", ylab="Accuracy",
     main="Accuracy vs Cutoff", col="blue")
abline(v=optimal_cutoffs$Cutoff[1], col="red", lty=2)
points(optimal_cutoffs$Cutoff[1], optimal_cutoffs$Accuracy[1], col="red", pch=16, cex=2)

# Plot 2: Youden Index vs Cutoff
plot(cutoff_analysis$Cutoff, cutoff_analysis$Youden,
     type="l", lwd=2, xlab="Cutoff", ylab="Youden Index",
     main="Youden Index vs Cutoff", col="brown")
abline(v=optimal_cutoffs$Cutoff[2], col="red", lty=2)
points(optimal_cutoffs$Cutoff[2], optimal_cutoffs$Score[2], col="red", pch=16, cex=2)

# Plot 3: F1 Score vs Cutoff
plot(cutoff_analysis$Cutoff, cutoff_analysis$F1_Score,
     type="l", lwd=2, xlab="Cutoff", ylab="F1 Score",
     main="F1 Score vs Cutoff", col="purple")
abline(v=optimal_cutoffs$Cutoff[3], col="red", lty=2)
points(optimal_cutoffs$Cutoff[3], optimal_cutoffs$Score[3], col="red", pch=16, cex=2)

# Plot 4: MCC vs Cutoff
plot(cutoff_analysis$Cutoff, cutoff_analysis$MCC,
     type="l", lwd=2, xlab="Cutoff", ylab="MCC",
     main="Matthews Correlation Coefficient vs Cutoff", col="orange")
abline(v=optimal_cutoffs$Cutoff[4], col="red", lty=2)
points(optimal_cutoffs$Cutoff[4], optimal_cutoffs$Score[4], col="red", pch=16, cex=2)

# Plot 5: Sensitivity vs Specificity
plot(cutoff_analysis$Cutoff, cutoff_analysis$Sensitivity,
     type="l", lwd=2, xlab="Cutoff", ylab="Rate",
     main="Sensitivity & Specificity vs Cutoff", col="blue")
lines(cutoff_analysis$Cutoff, cutoff_analysis$Specificity, lwd=2, col="green")
abline(v=optimal_cutoffs$Cutoff[5], col="red", lty=2)
points(optimal_cutoffs$Cutoff[5], optimal_cutoffs$Score[5], col="red", pch=16, cex=2)
legend("right", c("Sensitivity", "Specificity"), col=c("blue", "green"), lwd=2)

# Plot 6: Precision vs Cutoff
plot(cutoff_analysis$Cutoff, cutoff_analysis$Precision,
     type="l", lwd=2, xlab="Cutoff", ylab="Precision",
     main="Precision vs Cutoff", col="darkgreen")
abline(v=0.5, col="gray", lty=2)

dev.off()

# ============================================================================
# PART 5: SAVE RESULTS AND PRINT SUMMARY
# ============================================================================

# Save detailed cutoff analysis
write.table(cutoff_analysis, file="cutoff_analysis_detailed.tsv",
            quote=F, sep="\t", row.names=F)

# Save optimal cutoffs summary
write.table(optimal_cutoffs, file="optimal_cutoffs_summary.tsv",
            quote=F, sep="\t", row.names=F)

# Print summary to console
print(strrep("=", 80))
print("OPTIMAL CUTOFF ANALYSIS RESULTS")
print(strrep("=", 80))
print("")
print(optimal_cutoffs)
print("")

# Detailed recommendations
print(strrep("=", 80))
print("RECOMMENDATIONS")
print(strrep("=", 80))
print("")
print("For most use cases, choose from these top candidates:")
print("")
print("1. MAXIMUM YOUDEN INDEX (Recommended for balanced classification)")
print(paste("   Cutoff:", round(optimal_cutoffs$Cutoff[2], 4)))
print(paste("   Sensitivity:", round(optimal_cutoffs$Sensitivity[2], 4)))
print(paste("   Specificity:", round(optimal_cutoffs$Specificity[2], 4)))
print(paste("   Youden Index:", round(optimal_cutoffs$Score[2], 4)))
print("")

print("2. MAXIMUM F1 SCORE (If coding genes are more important)")
print(paste("   Cutoff:", round(optimal_cutoffs$Cutoff[3], 4)))
print(paste("   Sensitivity:", round(optimal_cutoffs$Sensitivity[3], 4)))
print(paste("   Specificity:", round(optimal_cutoffs$Specificity[3], 4)))
print(paste("   F1 Score:", round(optimal_cutoffs$Score[3], 4)))
print("")

print("3. MAXIMUM ACCURACY (Simplest, balances all errors)")
print(paste("   Cutoff:", round(optimal_cutoffs$Cutoff[1], 4)))
print(paste("   Accuracy:", round(optimal_cutoffs$Accuracy[1], 4)))
print("")

print("See 'cutoff_analysis.pdf' and 'optimal_cutoffs_summary.tsv' for full results")
print("")

# ============================================================================
# PART 6: SAVE FINAL CUTOFF TO FILE
# ============================================================================

# While Youden Index might be better, CPAT uses intersection of sensitivity and specificity
final_cutoff <- optimal_cutoffs$Cutoff[5]  # Balanced Sens/Spec
write.table(data.frame(Cutoff=final_cutoff), file="optimal_cutoff.txt",
            quote=F, sep="\t", row.names=F, col.names=T)

print(paste("Final recommended cutoff for CPAT:", round(final_cutoff, 4)))
print("This cutoff is based on the point where sensitivity and specificity are most balanced.")
print("See 'optimal_cutoff.txt' for the exact value.")

sink(type = "message")
sink(type = "output")
close(log_conn)
