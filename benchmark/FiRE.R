# 加载包
library(Seurat)
library(reticulate)
library(FiRE)

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

# 使用FiRE进行罕见细胞群识别
# 将数据转换为样本 * 特征的格式（细胞 * 基因）
fire_data <- t(normalized_data)  # 转置以匹配FiRE的需求

# 创建数据矩阵列表（包含数据和基因符号）
genes <- 1:ncol(fire_data)  # 可以替换为原始基因名称
data_mat <- list(mat = fire_data, gene_symbols = genes)


# 创建FiRE模型
model <- new(FiRE::FiRE, L = 100, M = 50, H = 1017881, seed = 5489, verbose = 0)

# 对数据集应用FiRE模型
model$fit(fire_data)

# 计算每个细胞的FiRE得分
score <- model$score(fire_data)

# 使用IQR准则选择高FiRE得分的细胞，识别稀有细胞
q3 <- quantile(score, 0.75)
iqr <- IQR(score)
th <- q3 + (1.5 * iqr)
rare_cells_indices <- which(score >= th)

# 将稀有细胞的索引添加到Seurat对象的meta data中
seurat_object$rare_cells <- "non-rare"
seurat_object$rare_cells[rare_cells_indices] <- "rare"

# 保存稀有细胞和非稀有细胞的索引到CSV文件
rare_status <- data.frame(Cell = colnames(seurat_object), Status = seurat_object$rare_cells)
write.csv(rare_status, file = "rare_cells_status_fire.csv", row.names = FALSE)

# 运行PCA进行降维
seurat_object <- ScaleData(seurat_object)
seurat_object <- RunPCA(seurat_object, features = VariableFeatures(seurat_object))

# 运行UMAP进行可视化
seurat_object <- RunUMAP(seurat_object, dims = 1:10)

# 可视化稀有细胞群
DimPlot(seurat_object, group.by = "rare_cells")
