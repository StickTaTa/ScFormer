# 加载包
library(Seurat)
library(reticulate)
library(RaceID)

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

# 使用RaceID进行稀有细胞群识别
# 将Seurat对象中的数据转换为RaceID输入格式
sc <- SCseq(normalized_data)

# 过滤低质量细胞
sc <- filterdata(sc, mintotal = 1000, minexpr = 5, minnumber = 5)

# 距离计算和聚类
sc <- compdist(sc, metric = "pearson")
sc <- clustexp(sc, cln = 10, sat = TRUE, bootnr = 100)

# 运行findoutliers以便进行稀有细胞类型识别
sc <- findoutliers(sc)

# 稀有细胞类型识别
# 通过设置稀有细胞群体的比例来控制稀有细胞的定义
rare_proportion <- 0.05  # 设置稀有细胞的最大比例，比如5%
rare_threshold <- ceiling(rare_proportion * length(sc@cpart))

# 识别稀有细胞群
rare_clusters <- which(table(sc@cpart) < rare_threshold)
rare_cells <- names(sc@cpart)[sc@cpart %in% rare_clusters]
cat("检测到的稀有细胞群数量：", length(rare_clusters), "\n")
cat("检测到的稀有细胞数量：", length(rare_cells), "\n")

# 将稀有细胞的索引添加到Seurat对象的meta data中
seurat_object$rare_cells <- "non-rare"
seurat_object$rare_cells[rare_cells] <- "rare"

# 保存稀有细胞和非稀有细胞的索引到CSV文件
rare_status <- data.frame(Cell = colnames(seurat_object), Status = seurat_object$rare_cells)
write.csv(rare_status, file = "RaceID3_rare_cells_status.csv", row.names = FALSE)

# 运行PCA进行降维
seurat_object <- ScaleData(seurat_object)
seurat_object <- RunPCA(seurat_object, features = VariableFeatures(seurat_object))

# 运行UMAP进行可视化
seurat_object <- RunUMAP(seurat_object, dims = 1:10)

# 可视化稀有细胞群
DimPlot(seurat_object, group.by = "rare_cells")
