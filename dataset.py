import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, extract_zip
from custom_io import read_tcp_data

from utils import download_url

class TCPDataset(InMemoryDataset):
    """
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): 'train' or 'valid_query' or 'test_query'
    """

    def __init__(self, root, f_name='train'):
        assert f_name in ['train', 'valid_query', 'test_query']
        self.f_name = f_name
        super(TCPDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = 'raw'
        return osp.join(self.root, name)

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['train_000', 'train_001']
        return ['{}.txt'.format(name) for name in names]

    @property
    def processed_file_names(self):
        return f'data_{self.f_name}.pt'

    def download(self):
        try:
            assert len(os.listdir(osp.join(self.root, 'raw')))==4
        except:
            raise ValueError(f'Save four directory to {self.root}/raw by unzipping given zip files')

    def process(self):
        self.data, self.slices = read_tcp_data(self.raw_dir, self.f_name)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

if __name__=='__main__':
    print('train')
    dataset = TCPDataset(root='./data')
    print('valid_qry')
    dataset_val = TCPDataset(root='./data', f_name='valid_query')
    print('test_qry')
    dataset_test = TCPDataset(root='./data', f_name='test_query')