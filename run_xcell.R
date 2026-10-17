#!/usr/bin/env Rscript
# =============================================================================
# 用官方 xCell（R 包）对 GSE65682 做细胞组成反卷积
# xCell 要求输入：基因 × 样本 的线性尺度矩阵，行名为人类基因 symbol
# =============================================================================
suppressMessages({
  library(GSVA); library(GSEABase); library(pracma); library(MASS)
})
source("/tmp/xCell-master/R/xCell.R")
load("/tmp/xCell-master/data/xCell.data.rda")   # 提供 xCell.data（signatures / genes / spill）
cat("xCell.data 组件:", names(xCell.data), "\n")

args <- commandArgs(trailingOnly = TRUE)
infile  <- ifelse(length(args) >= 1, args[1], "GSE65682_gene_matrix_linear.csv")
outfile <- ifelse(length(args) >= 2, args[2], "xcell_results.csv")

cat("[1/3] 读取线性表达矩阵：", infile, "\n")
expr <- as.matrix(read.csv(infile, row.names = 1, check.names = FALSE))
cat("    dim =", dim(expr)[1], "genes x", dim(expr)[2], "samples\n")
cat("    range =", range(expr), "\n")

# xCell 需要非负；线性 CPM 已非负
if (min(expr) < 0) { expr[expr < 0] <- 0; cat("    负值已截断为 0\n") }

cat("[2/3] 运行 xCellAnalysis（rnaseq=TRUE，与 RNA-seq/芯片 CPM 尺度匹配）...\n")
set.seed(2026)
scores <- xCellAnalysis(expr, rnaseq = TRUE, parallel.sz = 4, parallel.type = "SOCK")
cat("    输出 dim =", dim(scores)[1], "cell types x", dim(scores)[2], "samples\n")

# xCell 输出：细胞类型 × 样本 → 转成 样本 × 细胞类型
out <- as.data.frame(t(scores))
out <- cbind(sample = rownames(out), out)
cat("[3/3] 写入：", outfile, "\n")
write.csv(out, outfile, row.names = FALSE)
cat("完成。细胞类型：\n"); print(colnames(scores))
