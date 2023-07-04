import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, one_hot, remove_self_loops
import os
import os.path as osp
import glob

class MyPyGDataset(InMemoryDataset):
    def __init__(self, root: str, name: str,transform=None, pre_transform=None):
        self.name = name
        super(MyPyGDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, f'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, f'processed')

    @property
    def raw_file_names(self):
        return osp.join(self.root, self.name, f'raw')

    @property
    def processed_file_names(self):
        return 'data.pt'

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
    #     ...

    def process(self):
        files = glob.glob(osp.join(self.raw_file_names, f'{self.name}_*.txt'))
        names = [f.split(os.sep)[-1][len(self.name) + 1:-4] for f in files]

        edge_index = read_file(self.raw_file_names, self.name, 'edge_index', torch.long).t()
        batch = read_file(self.raw_file_names, self.name, 'graph_indicator', torch.long) - 1

        node_attributes = torch.empty((batch.size(0), 0))
        if 'node_attributes' in names:
            node_attributes = read_file(self.raw_file_names, self.name, 'node_attributes')
            if node_attributes.dim() == 1:
                node_attributes = node_attributes.unsqueeze(-1)

        node_labels = torch.empty((batch.size(0), 0))
        if 'node_labels' in names:
            node_labels = read_file(self.raw_file_names, self.name, 'node_labels', torch.long)
            if node_labels.dim() == 1:
                node_labels = node_labels.unsqueeze(-1)
            node_labels = node_labels - node_labels.min(dim=0)[0]
            node_labels = node_labels.unbind(dim=-1)
            node_labels = [one_hot(x) for x in node_labels]
            if len(node_labels) == 1:
                node_labels = node_labels[0]
            else:
                node_labels = torch.cat(node_labels, dim=-1)

        edge_attributes = torch.empty((edge_index.size(1), 0))
        if 'edge_attributes' in names:
            edge_attributes = read_file(self.raw_file_names, self.name, 'edge_attributes')
            if edge_attributes.dim() == 1:
                edge_attributes = edge_attributes.unsqueeze(-1)

        edge_labels = torch.empty((edge_index.size(1), 0))
        if 'edge_labels' in names:
            edge_labels = read_file(self.raw_file_names, self.name, 'edge_labels', torch.long)
            if edge_labels.dim() == 1:
                edge_labels = edge_labels.unsqueeze(-1)
            edge_labels = edge_labels - edge_labels.min(dim=0)[0]
            edge_labels = edge_labels.unbind(dim=-1)
            edge_labels = [one_hot(e) for e in edge_labels]
            if len(edge_labels) == 1:
                edge_labels = edge_labels[0]
            else:
                edge_labels = torch.cat(edge_labels, dim=-1)

        x = cat([node_attributes, node_labels])
        edge_attr = cat([edge_attributes, edge_labels])

        y = None
        if 'graph_attributes' in names:  # Regression problem.
            y = read_file(self.raw_file_names, self.name, 'graph_attributes')
        elif 'graph_labels' in names:  # Classification problem.
            y = read_file(self.raw_file_names, self.name, 'graph_labels', torch.long)
            _, y = y.unique(sorted=True, return_inverse=True)

        # num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
        # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        # edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        self.data, self.slices = split(data, batch)

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]
        #
        # data, slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

def read_file(folder, prefix, name, dtype=None):
    path = osp.join(folder, f'{prefix}_{name}.txt')
    return read_txt_array(path, sep=',', dtype=dtype)

def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item for item in seq if item.numel() > 0]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    else:
        # Imitate `collate` functionality:
        data._num_nodes = torch.bincount(batch).tolist()
        data.num_nodes = batch.numel()
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices
