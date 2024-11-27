# 加载包
library(Seurat)
library(reticulate)
library(GapClust)
library(Matrix)
library(irlba)
library(FNN)
library(moments)

# 使用reticulate来加载h5ad文件
# 指定使用已有的conda环境
use_condaenv("scrna", required = TRUE)

# 导入anndata Python包来处理h5ad文件
anndata <- import("anndata")

# 加载h5ad数据
adata <- anndata$read_h5ad("../output/pbmc_benchmark_data/pbmc_benchmark_s1d1.h5ad")

# 确保变量名称唯一
adata$var_names_make_unique()

# 将h5ad转换为Seurat对象
# 获取表达矩阵（通常保存在X中）
expression_matrix <- t(adata$X)  # 转置以匹配gene*cell的格式

# 创建Seurat对象
seurat_object <- CreateSeuratObject(counts = expression_matrix)

# 标准数据预处理（数据归一化和高变基因选择等）
seurat_object <- NormalizeData(seurat_object)
seurat_object <- FindVariableFeatures(seurat_object)

# 提取标准化的表达矩阵，确保数据为基因*细胞格式
normalized_data <- as.matrix(GetAssayData(seurat_object, slot = "data"))

# 修改后的 GapClust 函数，去除对 meta.features 的依赖
GapClust <- function(data, k = 200) {
  ## Fano genes for clustering
  pbmc <- CreateSeuratObject(counts = data)
  pbmc <- FindVariableFeatures(object = pbmc, selection.method = 'vst', nfeatures = dim(data)[1], verbose = FALSE)
  features.vst <- VariableFeatures(pbmc)  # 直接使用 VariableFeatures 函数获取高变基因

  if (length(features.vst) == 0) {
    stop("Unable to find variable features in the Seurat object.")
  }

  tmp <- data[rownames(data) %in% features.vst, ]
  tmp <- log2(as.matrix(tmp) + 1)
  pca <- irlba::irlba(t(tmp), nv = min(c(50, dim(tmp) - 1)))
  pca$pca <- t(pca$d * t(pca$u))
  knn.res <- FNN::get.knn(pca$pca, k = k)

  distance.diff <- (knn.res$nn.dist[, -1, drop = FALSE] - knn.res$nn.dist[, -ncol(knn.res$nn.dist), drop = FALSE])
  diff.left <- distance.diff[, -1, drop = FALSE] - distance.diff[, -ncol(distance.diff), drop = FALSE]
  diff.both <- diff.left[, -ncol(diff.left), drop = FALSE] - diff.left[, -1, drop = FALSE]
  diff.both[, 1] <- diff.both[, 1] + distance.diff[, 1]  # Very important due to distance variation to the first neighbor.

  v1.k <- matrix(NA, dim(data)[2], k - 3)
  skew <- c()
  top.values.ave <- c()
  for (j in 1:dim(diff.both)[2]) {
    v <- diff.both[, j]
    v1 <- v
    for (m in 1:length(v)) {
      v1[m] <- (v[m] + v[knn.res$nn.index[m, 2]]) / 2
    }
    v1.k[, j] <- (v1)
    v2 <- v1[order(v1, decreasing = TRUE)[(j + 2):length(v1)]]
    v2[is.na(v2)] <- 0
    top.values <- v1[knn.res$nn.index[which.max(v1), 1:(j + 1)]]
    v2 <- c(v2[v2 <= (quantile(v2, 0.75) + 1.5 * IQR(v2)) & v2 >= (quantile(v2, 0.25) - 1.5 * IQR(v2))], rep(sum(top.values[top.values > 0]) / length(top.values), (2)))
    skew <- c(skew, moments::skewness(v2))
    top.values.ave <- c(top.values.ave, mean(top.values))
  }

  ids <- which(skew > 2)
  col.mat <- matrix(0, length(ids), dim(tmp)[2])
  for (i in 1:length(ids)) {
    top.cell <- which.max(v1.k[, (ids[i])])
    col.mat[i, knn.res$nn.index[top.cell, 1:(ids[i] + 1)]] <- skew[ids[i]] * top.values.ave[ids[i]]
  }

  id.max <- apply(col.mat, 2, which.max)
  max.val <- apply(col.mat, 2, max)
  id.max[max.val == 0] <- 0
  cnt <- table(id.max)
  cnt <- cnt[names(cnt) != '0']
  id.max.match <- cnt[which(cnt == (ids[as.integer(names(cnt))] + 1))] - 1

  cls <- rep(0, dim(tmp)[2])
  for (id.match in id.max.match) {
    cls[id.max == (id.match)] <- which(id.max.match %in% id.match)
  }

  rare.cells <- list()
  for (id.match in id.max.match) {
    rare.cells[[as.character(id.match)]] <- knn.res$nn.index[which.max(v1.k[, id.match]), 1:(id.match + 1)]
  }
  results <- list(skewness = skew, rare_cell_indices = rare.cells, rare_score = v1.k)
  return(results)
}

# 使用修改后的 GapClust 函数识别稀有细胞群
# 使用标准化且未标度化的表达矩阵作为输入
gapclust_results <- GapClust(normalized_data, k = 200)

# 输出GapClust的结果
print(gapclust_results)

# 将稀有细胞的索引添加到Seurat对象的meta data中
rare_cells_indices <- unlist(gapclust_results$rare_cell_indices)  # 提取稀有细胞的索引

# 添加标签到meta数据中
seurat_object$rare_cells <- "non-rare"
seurat_object$rare_cells[rare_cells_indices] <- "rare"

# 保存稀有细胞和非稀有细胞的索引到CSV文件
rare_status <- data.frame(Cell = colnames(seurat_object), Status = seurat_object$rare_cells)
write.csv(rare_status, file = "GapClust_rare_cells_status.csv", row.names = FALSE)

# 运行PCA进行降维
seurat_object <- ScaleData(seurat_object)
seurat_object <- RunPCA(seurat_object, features = VariableFeatures(seurat_object))

# 运行UMAP进行可视化
seurat_object <- RunUMAP(seurat_object, dims = 1:10)

# 可视化稀有细胞群
DimPlot(seurat_object, group.by = "rare_cells")
