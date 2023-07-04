import anndata
import h5py
import pandas as pd
import numpy as np
from csn import upperlower
from scipy.sparse import csr_matrix
from scipy.stats import norm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

input_file = r"./inputdata/Biase_k3/"
sample_name = "E-MTAB-3321"
alpha = 0.1

if input_file.endswith("/"): # 处理文件夹（txt/tsv/csv）
    gem_file = input_file + "Biase_k3_FPKM_scRNA_less.csv"
    gem_df = pd.read_csv(gem_file, sep=",", index_col=0)

elif input_file.endswith(".h5"):
    h5file = h5py.File(input_file, 'r')
    dataset = {}
    for group in h5file.keys():
        group = h5file[group]
        dataset[group.name] = group[:]
elif input_file.endswith(".h5ad"):
    adata = anndata.read(input_file)
    X = pd.DataFrame(adata.X.todense())
    obs = adata.obs
    var = adata.var
    # 获取细胞数据
    cell_columns = obs.columns # 细胞元数据包含内容
    cell_type = np.array(obs.cell_type1) # 细胞类型
    label = []
    for item in cell_type:
        label.append(list(set(cell_type)).index(item))
    # label = torch.tensor(np.array(label).reshape(len(label), 1), dtype=torch.int64)
    # one_hot = torch.zeros(cell_type.shape[0], len(set(cell_type))).scatter_(1, label, 1)
    # 获取表达量矩阵转
    cell_name = adata.obs.index
    chr_name = adata.var.index
    gem = pd.DataFrame(adata.X.todense())
    gem.index = cell_name
    gem.columns = chr_name
    gem = gem.T

# 生成torch_geometric.data.data.Data数据作为输入
gem, gene, sample = gem_df.values, gem_df.index.values, gem_df.columns.values
(n1, n2) = gem.shape
eps = np.finfo(float).eps
(upper, lower) = upperlower(gem, boxsize=0.1)

dataset_raw_path = f"PyGDataset\\{sample_name}\\raw"
dataset_raw_file = [os.path.join(dataset_raw_path, f"{sample_name}_edge_index.txt"),
                    os.path.join(dataset_raw_path, f"{sample_name}_edge_attributes.txt"),
                    os.path.join(dataset_raw_path, f"{sample_name}_node_labels.txt"),
                    os.path.join(dataset_raw_path, f"{sample_name}_graph_indicator.txt"),
                    # os.path.join(dataset_raw_path, f"{sample_name}_graph_labels.txt")
                    ]
if not os.path.exists(dataset_raw_path):
    os.makedirs(dataset_raw_path, exist_ok=True)
else:
    if os.path.exists(dataset_raw_path):
        for file in os.listdir(dataset_raw_path):
            os.remove(dataset_raw_path + "\\" + file)

matrix_start_index = 0
for k in range(0, n2):
    sampleID = sample[k]
    B = np.zeros((n1, n2), dtype=np.float32)
    for j in range(0, n2):
        B[:, j] = (gem[:, j] <= upper[:, k]) & (gem[:, j] >= lower[:, k]) & (gem[:, k] > 0)
    a = B.sum(axis=1)
    a = np.reshape(a, (n1, 1))
    temp = (B @ B.T * n2 - a @ a.T) / np.sqrt((a @ a.T) * ((n2 - a) @ (n2 - a).T) / (n2 - 1) + eps)
    np.fill_diagonal(temp, 0)
    matrix = csr_matrix(temp).tocoo()
    matrix = matrix.multiply(matrix >= norm.ppf(1 - alpha)) # filter the data

    with open(dataset_raw_file[0], 'a') as f:
        edge_index = np.array(list(zip(matrix.nonzero()[0] + matrix_start_index, matrix.nonzero()[1] + matrix_start_index)))
        np.savetxt(f, edge_index, delimiter=', ', fmt='%d')
    with open(dataset_raw_file[1], 'a') as f:
        edge_attributes = np.zeros(matrix.count_nonzero())
        for edge in range(matrix.count_nonzero()):
            edge_attributes[edge] = matrix.data[edge]
        np.savetxt(f, edge_attributes, delimiter='\n', fmt='%.4f')
    with open(dataset_raw_file[2], 'a') as f:
        node_labels = np.array(range(n1))
        np.savetxt(f, node_labels, delimiter='\n', fmt='%d')
    with open(dataset_raw_file[3], 'a') as f:
        graph_indicator = np.full(n1, k + 1)
        np.savetxt(f, graph_indicator, delimiter='\n', fmt='%d')
    # with open(dataset_raw_file[4], 'a') as f:
    #     f.write(str(label[k]) + "\n")

    matrix_start_index += n1
    print(f'The sample {k + 1} get net end! node num : {matrix.count_nonzero()}')
