import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data.data import Data
from torch_geometric.data import Dataset
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from MyPyGDataset import MyPyGDataset
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sample_name = "E-MTAB-3321"

dataset = MyPyGDataset(root="PyGDataset", name=f"{sample_name}")
data1 = dataset[0]
data2 = dataset[1] # Get the first graph object.

G = to_networkx(data1, edge_attrs=['edge_attr'], to_undirected=True)
# plt.figure(figsize=(50, 50))
# nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
# plt.show()

adjacency_matrix = nx.adjacency_matrix(G, weight='edge_attr')

print("ddd")

