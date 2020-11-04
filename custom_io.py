import os
import os.path as osp
import glob
import random

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data

from constant import ATTACK_TYPES

COLUMNS = ['source', 'destination', 'port', 'time', 'attack']
COLUMNS2 = ['source', 'destination', 'port', 'time']
EMBEDDING_DIM = 100
TOP_PORT = 40 # max port per a batch 31 < 40
FOR_IDENTIFY = 30000 # max node number 23397 < 30000


def read_tcp_data(folder):
    """To read and process raw data.

    Parameters
    ----------
    folder : list of directory(:str:)
        Root directory where the data `TCPConnection` exists.
    """
    # TODO November 04, 2020: folder will be 'train' ==> ['train', 'valid_qry', 'valid_answer', 'test_qry']
    folder = osp.join('data', 'raw', 'train')
    files = glob.glob(osp.join(folder, '*.txt'))

    # aggregate all files to pandas.DataFrame
    df_list = []
    for idx, file_name in enumerate(files):
        df = pd.read_csv(file_name, delimiter='\t', header=None)
        df.columns = COLUMNS
        df['graph_idx'] = idx
        df_list.append(df)
    df = pd.concat(df_list)

    # node embedding is different in some part, so we should identify nodes
    df['auxiliary_source'] = df.source + df.graph_idx * FOR_IDENTIFY
    df['auxiliary_destination'] = df.destination + df.graph_idx * FOR_IDENTIFY

    edge_index = np.array([df.auxiliary_source.values, df.auxiliary_destination.values])
    edge_index = torch.from_numpy(edge_index)

    #
    # STEP 1: Node attributes
    #
    # TODO November 05, 2020: 'train' ==> ['train', 'valid_qry', 'valid_answer', 'test_qry']
    torch.manual_seed(999)
    set_nodes_source = df.source.unique()
    set_nodes_destination = df.destination.unique()
    set_nodes = set(set_nodes_source) | set(set_nodes_destination)
    
    # node_att1 : random vector allocated for each ip
    embedding_values = torch.rand(len(set_nodes), EMBEDDING_DIM)
    table_embedding = {}
    for nth, key in enumerate(sorted(set_nodes)):
        table_embedding[key] = embedding_values[nth]

    # node_att2 : port frequency vector for each source ip in a graph
    # node_att3 : port frequency vector for each destination ip in a graph
    top_port_score_per_graph = df.groupby('graph_idx').apply(lambda x: tps_p_g(x))
    top_port_score_per_graph_source = df.groupby(['graph_idx', 'source']).\
                        apply(lambda x: tps_p_gn(x, top_port_score_per_graph))
    top_port_score_per_graph_destination = df.groupby(['graph_idx', 'destination']).\
                        apply(lambda x: tps_p_gn(x, top_port_score_per_graph))

    # Aggregate the node_attributes
    node_attributes = []
    
    set_nodes_source = df.auxiliary_source.unique()
    set_nodes_destination = df.auxiliary_destination.unique()
    set_nodes = set(set_nodes_source) | set(set_nodes_destination)
    set_nodes = sorted(set_nodes)

    for aux_node in set_nodes:
        double_flag = 0 # just exception check
        node_id = aux_node % FOR_IDENTIFY
        graph_id = aux_node // FOR_IDENTIFY
        # node_att1
        node_att1 = table_embedding[node_id]
        # node_att2
        try:
            node_att2 = top_port_score_per_graph_source[graph_id][node_id].copy()
            node_att2 += [0.] * (TOP_PORT-len(node_att2)) # uniform shape of tensor
        except KeyError:
            node_att2 = [0.] * TOP_PORT
            double_flag += 1
        # node_att3
        try:
            node_att3 = top_port_score_per_graph_destination[graph_id][node_id].copy()
            node_att3 += [0.] * (TOP_PORT-len(node_att3)) # uniform shape of tensor
        except KeyError:
            node_att3 = [0.] * TOP_PORT
            double_flag += 1
        # exception
        if double_flag==2:
            raise KeyError(f'There is not {node_id} node')
        # Aggregate
        node_attr = torch.cat([node_att1, torch.tensor(node_att2), torch.tensor(node_att3)])
        node_attributes.append(node_attr.clone())
    
    node_attributes = torch.stack(node_attributes)

    #
    # STEP 2: Edge attributes
    #
    # edge_att1 : the number of connection occurred in a graph(during 30 min)
    edge_count = df.groupby(['auxiliary_source', 'auxiliary_destination']).apply(lambda x: len(x))
    edge_att1 = torch.from_numpy(edge_count.values) # edge count

    # edge_att2 : time...
    # TODO November 04, 2020: edge_att2 will be added (containing time information)

    # edge_label : what types of attack happend in this connection path in a graph
    edge_labels = []
    temp = df.groupby(['auxiliary_source', 'auxiliary_destination']).apply(lambda x: x.attack)
    for edge in edge_count.index:
        edge_attack_types = temp[edge].unique()
        label = torch.zeros(len(ATTACK_TYPES))
        for edge_atk in edge_attack_types:
            a_type = ATTACK_TYPES.index(edge_atk)
            label[a_type] = 1.
        edge_labels.append(label)
    edge_labels = torch.stack(edge_labels)

    # aggregate
    edge_attr = torch.cat([edge_att1.unsqueeze(1), edge_labels], axis=1)

    #
    # STEP 3: Graph label
    #
    y = df.groupby('graph_idx').apply(lambda x: x.attack.unique())
    y = y.apply(lambda x: atk_to_onehot(x))
    y = np.array([*y.values])
    y = torch.from_numpy(y)

    #
    # STEP 4: Load graph data on `torch_geometric.data.Data`
    #
    node_table = {old_id: new_id for new_id, old_id in enumerate(set_nodes)}
    edge_index = torch.from_numpy(np.array([*edge_count.index.values]).T).long()
    edge_index.apply_(lambda x: node_table[x])
    sorted_index = torch.sort(edge_index[0]).indices
    edge_index = edge_index[:, sorted_index]

    data = Data(x=node_attributes, edge_index=edge_index, edge_attr=edge_attr, y=y)
    batch = torch.tensor([i//30000 for i in set_nodes]).long()
    data, slices = split(data, batch)

    return data, slices

def tps_p_g(x):
    """compute TopPortScore for each graph_id

    It returns 40-dimensional embedding s.t. n-th element means
    the portion of top n-th port occurred in the graph
    """
    top_port_score = x['port'].value_counts().nlargest(TOP_PORT) / len(x)
    return top_port_score

def tps_p_gn(x, top_port_score_per_graph):
    """compute TopPortScore for (graph_id, node)

    It returns 40-dimensional embedding s.t. n-th element means
    the portion of top n-th port occurred in the node
    """
    relative_freq_list = []
    graph_idx = x.graph_idx.unique()[0]
    top_port_score = top_port_score_per_graph[graph_idx]
    
    for top_port in top_port_score.index:
        freq = sum(x.port == top_port)
        relative_freq = freq / len(x)
        relative_freq_list.append(relative_freq)
        
    return relative_freq_list

def atk_to_onehot(attack_array):
    """Given attack_type array, it returns 25-dimensional one-hot encoding vector
    """
    y = np.zeros(len(ATTACK_TYPES))
    for attack in attack_array:
        a_type = ATTACK_TYPES.index(attack)
        y[a_type] = 1.
    return y

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices
