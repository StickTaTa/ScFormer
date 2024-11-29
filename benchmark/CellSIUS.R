# 加载必要的R包
library(Seurat)
library(reticulate)
library(CellSIUS)

# 使用reticulate加载h5ad文件
use_condaenv("scrna", required = TRUE)
anndata <- import("anndata")

# 加载h5ad数据
adata <- anndata$read_h5ad("../output/pbmc_benchmark_data/pbmc_benchmark_s1d1.h5ad")

# 确保变量名称唯一
adata$var_names_make_unique()

# 转换为Seurat对象
expression_matrix <- t(adata$X)  # 转置为基因*细胞格式
seurat_object <- CreateSeuratObject(counts = expression_matrix)

# 数据预处理（归一化和高变基因选择）
seurat_object <- NormalizeData(seurat_object)
seurat_object <- FindVariableFeatures(seurat_object)

# 聚类：先进行降维和邻域检测
seurat_object <- ScaleData(seurat_object)
seurat_object <- RunPCA(seurat_object, features = VariableFeatures(seurat_object))
seurat_object <- FindNeighbors(seurat_object, dims = 1:10)
seurat_object <- FindClusters(seurat_object)

# 准备CellSIUS输入数据
clusters <- Idents(seurat_object)  # 聚类分组信息
mat.norm <- as.matrix(GetAssayData(seurat_object, slot = "data"))  # 标准化表达矩阵

# 确保输入符合CellSIUS格式
cellsius_data <- list(
  mat.norm = mat.norm, 
  group_id = as.character(clusters)  # 转为字符型向量
)

# 运行CellSIUS算法
mcl_path <- "~/local/bin/mcl"  # 指定MCL工具路径
cellsius_out <- CellSIUS(
  mat.norm = cellsius_data$mat.norm,
  group_id = cellsius_data$group_id,
  min_n_cells = 10,          # 每个簇中模式的最小细胞数
  min_fc = 1,                # 基因表达差异倍数，增加至更高值以提高敏感性
  corr_cutoff = NULL,        # 自动设置相关性阈值
  iter = 0,                  # 基于第一模式的分配
  max_perc_cells = 5,        # 限制稀有细胞比例为5%
  fc_between_cutoff = 2,     # 子群和其他细胞的最低差异倍数
  mcl_path = mcl_path
)

# 检查结果是否生成
if (is.null(cellsius_out) || nrow(cellsius_out) == 0) {
  stop("CellSIUS未能识别到稀有细胞亚群或返回了空结果。")
}

# 分析CellSIUS输出，提取稀有细胞信息
rare_cells_indices <- which(cellsius_out$sub_cluster != 0)

# 检查稀有细胞的比例
total_cells <- ncol(seurat_object)
rare_cells <- length(rare_cells_indices)
rare_cells_ratio <- rare_cells / total_cells

if (rare_cells_ratio > 0.05) {
  warning("稀有细胞的比例超过5%。请检查参数设置是否过于宽松。")
}

# 将稀有细胞状态添加到Seurat对象
seurat_object$rare_cells <- "non-rare"
if (rare_cells_ratio <= 0.05) {
  seurat_object$rare_cells[rare_cells_indices] <- "rare"
}

# 保存稀有细胞和非稀有细胞状态到CSV文件
rare_status <- data.frame(Cell = colnames(seurat_object), Status = seurat_object$rare_cells)
write.csv(rare_status, file = "rare_cells_status_cellsius.csv", row.names = FALSE)

# 可视化稀有细胞群
seurat_object <- RunUMAP(seurat_object, dims = 1:10)
DimPlot(seurat_object, group.by = "rare_cells", label = TRUE, pt.size = 1)
