# Created by lcquan@nwsuaf.edu.cn on 2022/10/16

import torch
import random
from torch_geometric.data import Data
import json
from torch.nn.functional import normalize
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from bgrl import *

def get_lasftm_asia(dataset_name):
    graph_edges = "./data/lasftm-asia/lastfm_asia_edges.txt"
    graph_node_feature = "./data/lasftm-asia/lastfm_asia_features.json"
    graph_node_label = "./data/lasftm-asia/lastfm_asia_target.txt"

    start = []
    to = []
    for line in open(graph_edges):
        strlist = line.split()
        start.append(int(strlist[0]))
        to.append(int(strlist[1]))
    edge_index = torch.tensor([start, to], dtype=torch.int64)

    label_list = [int(line.split()[1]) for line in open(graph_node_label)]
    y = torch.tensor(label_list)

    x_values = []
    with open(graph_node_feature, 'r') as fp:
        json_data = json.load(fp)
    max_index = 7841 # max([max(v) for v in json_data.values() if len(v)>0])

    for raw_feat in json_data.values():
        mask = torch.tensor(raw_feat)
        x_value = torch.zeros(max_index+1)
        if len(raw_feat)>0:
           x_value[mask] = 1
        x_values.append(x_value.tolist())
    x = torch.tensor(x_values, dtype=torch.float32)
    x = normalize(x, p=2.0,dim=0)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def get_planetoid(dataset_name):
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '', 'data', dataset_name)
    dataset = Planetoid("./data", dataset_name, transform=T.TargetIndegree())
    data = dataset[0]
    return data


def get_wiki(dataset_name):
    dataset, _, _,_, = get_wiki_cs('./data/wiki-cs')
    data = dataset[0]
    return data


def get_common_dataset(dataset_name):
    dataset = get_dataset('./data', dataset_name)
    data = dataset[0]  # all dataset include one graph
    return data


def load_dataset(dataset_name):

    loader_dict = {"amazon-computers":get_common_dataset,
                   "amazon-photos": get_common_dataset,
                   "coauthor-cs": get_common_dataset,
                   "coauthor-physics": get_common_dataset,
                   "wiki-cs": get_wiki,
                   "cora": get_planetoid,
                   "citeseer": get_planetoid,
                   "pubmed": get_planetoid,
                   "lasftm-asia": get_lasftm_asia
                   }

    data = loader_dict[dataset_name.lower()](dataset_name.lower())

    node_degree_path = "./data/node-degree/" + dataset_name + "-node-degrees.txt"
    node_degree_list = [(int(line.split()[1]), int(line.split()[0])) for line in open(node_degree_path)]
    node_degrees = torch.tensor(node_degree_list)
    data.node_degrees = node_degrees

    node_degree_list_without_id = [int(line.split()[1]) if int(line.split()[1])>0 else 1 for line in open(node_degree_path)]
    node_degrees_without_id = torch.tensor(node_degree_list_without_id)
    data.node_degrees_without_id = node_degrees_without_id

    data.num_classes = torch.max(data.y, dim=0)[0].item() + 1
    # c_num_list = [(data.y == c).long().sum(0).item() for c in range(data.num_classes)]

    return data


def make_pu_dataset(data, pos_index=[0],
                    use_high_degree=False, fixed_seed=True,
                    sample_seed=10, seed_nodes_num=5, test_pct=1.0):
    # transform into positive-negative dataset
    data.y = sum([data.y == idx for idx in pos_index]).long()

    # created train_mask and val_mask
    data.train_mask = torch.zeros(data.y.size()).bool().fill_(False)
    # return indices of positive nodes, view(-1): as a 1-dim tuple
    pos_idx = data.y.nonzero(as_tuple=False).view(-1)
    pos_num = pos_idx.size(0)

    # sort or permutate pos_idx, output as a list
    # pos_idx = idx[torch.randperm(pos_num)]
    if use_high_degree:
        pos_node_degrees = data.node_degrees[pos_idx]
        pos_node_degrees_list = pos_node_degrees.tolist()
        pos_node_degrees_list.sort(reverse=True)
        pos_idx_list = [x[1] for x in pos_node_degrees_list]
    else:
        pos_idx_list = pos_idx.tolist()
        if fixed_seed:
            random.seed(sample_seed)
        random.shuffle(pos_idx_list)
    # sample seed nodes
    pos_idx = torch.tensor(pos_idx_list)
    train_idx = pos_idx[:seed_nodes_num]
    data.train_mask[train_idx] = True      # seed nodes mask
    data.pos_train_mask = data.train_mask  #re-name, for easy use
    data.train_seed_nodes = data.train_mask.nonzero(as_tuple=False).view(-1)

    # negative train mask: neg_train_mask, only contains negative nodes. Usually, this is only available for baseline.
    data.neg_train_mask = torch.zeros(data.y.size()).bool().fill_(False)
    neg_idx = (data.y == 0).nonzero(as_tuple=False).view(-1)
    perm_neg_idx = neg_idx[torch.randperm(neg_idx.size(0))]
    neg_train_idx = perm_neg_idx[:seed_nodes_num] # The number is the same as that of pos_train_mask
    data.neg_train_mask[neg_train_idx] = True

    # un_train_mask--unlabeled train mask (may contain both P and N nodes), and test_mask
    data.un_train_mask = torch.zeros(data.y.size()).bool().fill_(False)
    data.test_mask = torch.zeros(data.y.size()).bool().fill_(False)
    # remaining index and permutate
    remaining_idx = (~data.train_mask).nonzero(as_tuple=False).view(-1)
    perm_remaining_idx = remaining_idx[torch.randperm(remaining_idx.size(0))]
    # sample
    data.un_train_mask[perm_remaining_idx[:seed_nodes_num]] = True # The number is the same as that of pos_train_mask
    data.test_mask[perm_remaining_idx[:round(perm_remaining_idx.size(0) * test_pct)]] = True  # may overlap un_train_mask

    data.prior = data.y.sum().item() / data.y.size(0)
    return data
