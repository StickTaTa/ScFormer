import pandas as pd
from warnings import filterwarnings
import random

import torch.cuda as cuda
from scipy.sparse import csr_matrix
from scipy.io import mmread

from scformer.model import *
from scformer.utils import *

filterwarnings("ignore")
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

adata = sc.read_h5ad('../data/pbmc_benchmark_data/pbmc.h5ad')
mapping_dict = {
    'Naive CD20+ B IGKC+': 'Naïve CD20+ B',
    'Naive CD20+ B IGKC-': 'Naïve CD20+ B',
    'CD14+ Mono': 'CD14+ Mono',
    'HSC': 'HSC',
    'Reticulocyte': 'Erythroblast',
    'Normoblast': 'Normoblast',
    'NK CD158e1+': 'NK',
    'CD4+ T naive': 'CD4+ T naïve',
    'Erythroblast': 'Erythroblast',
    'NK': 'NK',
    'B1 B IGKC+': 'B1 B',
    'B1 B IGKC-': 'B1 B',
    'CD4+ T activated': 'CD4+ T activated',
    'CD4+ T activated integrinB7+': 'CD4+ T activated',
    'pDC': 'pDC',
    'Proerythroblast': 'Proerythroblast',
    'Transitional B': 'Transitional B',
    'MAIT': 'CD8+ T',
    'CD8+ T naive': 'CD8+ T naïve',
    'CD8+ T naive CD127+ CD26- CD101-': 'CD8+ T naïve',
    'T reg': 'CD4+ T activated',
    'CD8+ T CD49f+': 'CD8+ T',
    'CD8+ T TIGIT+ CD45RO+': 'CD8+ T',
    'gdT TCRVD2+': 'CD8+ T',  # gdT 类型被归入 CD8+ T
    'CD8+ T CD57+ CD45RA+': 'CD8+ T',
    'Lymph prog': 'Lymph prog',
    'Plasmablast IGKC+': 'Plasma cell',
    'Plasmablast IGKC-': 'Plasma cell',
    'CD8+ T CD69+ CD45RO+': 'CD8+ T',
    'CD8+ T TIGIT+ CD45RA+': 'CD8+ T',
    'CD8+ T CD69+ CD45RA+': 'CD8+ T',
    'ILC1': 'ILC',
    'cDC2': 'cDC2',
    'CD16+ Mono': 'CD16+ Mono',
    'G/M prog': 'G/Mprog',
    'MK/E prog': 'MK/E prog',
    'Plasma cell IGKC+': 'Plasma cell',
    'Plasma cell IGKC-': 'Plasma cell',
    'CD4+ T CD314+ CD45RA+': 'CD4+ T naïve',
    'ILC': 'ILC',
    'gdT CD158b+': 'CD8+ T',  # gdT 类型被归入 CD8+ T
    'CD8+ T CD57+ CD45RO+': 'CD8+ T',
    'dnT': None,  # 不需要的类型设为 None
    'cDC1': 'cDC2',  # 映射到 cDC2
    'T prog cycling': 'Lymph prog',  # 映射到 Lymph prog
}
adata.obs['Annotation'] = adata.obs['cell_type'].map(mapping_dict)
adata = adata[~adata.obs['Annotation'].isna()]

output_dir = '../output/pbmc_benchmark_data/'
os.makedirs(os.path.dirname(output_dir), exist_ok=True)

adata.write(output_dir + '/pbmc_benchmark.h5ad')

adata_train = adata[adata.obs['batch'].isin(['s1d1', 's2d1', 's3d1'])]
adata_test = adata[adata.obs['batch'].isin(['s4d1'])]

adata_train.write(output_dir + '/adata_train.h5ad')
adata_test.write(output_dir + '/adata_test.h5ad')

