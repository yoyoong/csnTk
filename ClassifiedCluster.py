import numpy as np
import pandas as pd
from MyPyGDataset import MyPyGDataset
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn import manifold
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, MeanShift, \
    estimate_bandwidth, SpectralClustering, DBSCAN, Birch, OPTICS, AffinityPropagation
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")

sample_name = "E-MTAB-3321"

dataset = MyPyGDataset(root="PyGDataset", name=f"{sample_name}")

x = np.zeros((dataset.len(), dataset[0].num_nodes * dataset[0].num_nodes))
x_3dim = np.zeros((dataset.len(), dataset[0].num_nodes, dataset[0].num_nodes))
i = 0
for data in dataset:
    G = to_networkx(data, edge_attrs=['edge_attr'], to_undirected=True)
    adjacency_matrix = nx.adjacency_matrix(G, weight='edge_attr')
    n1, n2 = adjacency_matrix.shape
    x[i, :] = adjacency_matrix.todense().reshape(1, n1 * n2)
    x_3dim[i, :, :] = adjacency_matrix.todense()
    i += 1

# 层次聚类
Agglomerative = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward').fit(x)
print('Agglomerative y_pred:' + str(Agglomerative.labels_))

# KMeans聚类
KMeans = KMeans(n_clusters = 3, random_state = 0).fit(x)
print('KMeans y_pred:' + str(KMeans.labels_))

# MiniBatchKMeans
MiniBatchKMeans = MiniBatchKMeans(n_clusters=3).fit(x)
print('MiniBatchKMeans y_pred:' + str(MiniBatchKMeans.labels_))

# mean-shift
# bandwidth = estimate_bandwidth(x, quantile=0.2, n_samples=100)
# MeanShift = MeanShift(bin_seeding=True).fit(x)
# print('MeanShift y_pred:' + str(MeanShift.labels_))

# SpectralClustering
# Spectral = SpectralClustering(n_clusters=3, affinity='precomputed', assign_labels='discretize').fit_predict(x)
# print('Spectral y_pred:' + str(Spectral.labels_))

# DBSCAN
DBSCAN = DBSCAN(eps=0.3, min_samples=5).fit(x)
print('DBSCAN y_pred:' + str(DBSCAN.labels_))

# Birch
birch = Birch(n_clusters = 3).fit(x)
print('Birch y_pred:' + str(birch.labels_))

# OPTICS
OPTICS = OPTICS(eps=0.8, min_samples=5).fit(x)
print('OPTICS y_pred:' + str(OPTICS.labels_))

# GaussianMixture
# GaussianMixture = GaussianMixture(n_components=3, covariance_type='full').fit(x)
# print('GaussianMixture y_pred:' + str(GaussianMixture.labels_))

# AffinityPropagation
AffinityPropagation = AffinityPropagation(damping=0.5, max_iter=100).fit(x)
print('AffinityPropagation y_pred:' + str(AffinityPropagation.labels_))


