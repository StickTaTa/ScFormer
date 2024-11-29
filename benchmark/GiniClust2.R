# 加载必要的包
library(Seurat)
library(reticulate)
library(umap)

# 设置 GiniClust2 的函数文件夹路径
Rfundir <- "C:/code/SpatialTransformer/benchmark/GiniClust2/GiniClust2_download/Rfunction"  # 修改为 GiniClust2 的路径
workdir <- "C:/code/SpatialTransformer/benchmark"  # 设置你的工作目录路径

# 设置工作目录
setwd(workdir)
dir.create(file.path(workdir, "results"), showWarnings = FALSE)  # 创建结果文件夹
dir.create(file.path(workdir, "figures"), showWarnings = FALSE)  # 创建图形文件夹

# 使用reticulate来加载h5ad文件
use_condaenv("scrna", required = TRUE)

# 导入anndata Python包来处理h5ad文件
anndata <- import("anndata")

# 加载h5ad数据
adata <- anndata$read_h5ad("../output/pbmc_benchmark_data/pbmc_benchmark_s1d1.h5ad")

# 确保基因名称唯一
adata$var_names_make_unique()

# 将h5ad转换为Seurat对象
expression_matrix <- t(adata$X)  # 转置以匹配gene*cell的格式
seurat_object <- CreateSeuratObject(counts = expression_matrix)

# 标准数据预处理（数据归一化和高变基因选择等）
seurat_object <- NormalizeData(seurat_object)
seurat_object <- FindVariableFeatures(seurat_object)

# 提取标准化的表达矩阵，确保数据为基因*细胞格式
normalized_data <- as.matrix(GetAssayData(seurat_object, slot = "data"))

# 使用GiniClust2进行稀有细胞群识别
# 将数据转换为样本 * 特征的格式（细胞 * 基因）
fire_data <- t(normalized_data)  # 转置以匹配GiniClust2的需求

# 设置GiniClust2路径
# Rfundir <- "C:/code/SpatialTransformer/benchmark/GiniClust2/Rfunction"  # 修改为GiniClust2的路径

# 加载GiniClust2所需的函数
source(paste(Rfundir, "GiniClust2_packages.R", sep = "/"))
source(paste(Rfundir, "GiniClust2_functions.R", sep = "/"))
source(paste(Rfundir, "GiniClust2_preprocess.R", sep = "/"))
source(paste(Rfundir, "GiniClust2_filtering_RawCounts.R", sep = "/"))

# 运行数据预处理和过滤
source(paste(Rfundir, "GiniClust2_preprocess.R", sep = "/"))
source(paste(Rfundir, "GiniClust2_filtering_RawCounts.R", sep = "/"))

# 运行 Gini 聚类
source(paste(Rfundir, "GiniClust2_fitting.R", sep = "/"))
source(paste(Rfundir, "GiniClust2_Gini_clustering.R", sep = "/"))

# 查看 Gini 聚类的结果
table(P_G)  # P_G 是 Gini 聚类的结果

