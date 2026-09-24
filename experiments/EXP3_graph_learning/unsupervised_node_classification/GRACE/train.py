import argparse
import os.path as osp
import random
import sys
from pathlib import Path
from time import perf_counter as t
import numpy as np
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
from eval import label_classification

_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from evaluation_protocol import get_split


def train(model: Model, x, edge_index):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y, split):
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
    return label_classification(z, y, split)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DBLP')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=15)
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation_name = config['activation']
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
        name = 'dblp' if name == 'DBLP' else name

        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            transform=T.NormalizeFeatures())

    path = osp.abspath(osp.expanduser(args.data_root))
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    results = []
    for run in range(args.runs):
        run_seed = args.seed + run
        seed_everything(run_seed)
        split = get_split(data.num_nodes, run_seed)
        activation = F.relu if activation_name == 'relu' else nn.PReLU()
        encoder = Encoder(
            dataset.num_features, num_hidden, activation,
            base_model=base_model, k=num_layers,
        ).to(device)
        model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        start = t()
        prev = start
        for epoch in range(1, num_epochs + 1):
            loss = train(model, data.x, data.edge_index)
            now = t()
            print(
                f'(T run {run + 1}/{args.runs}) | Epoch={epoch:03d}, '
                f'loss={loss:.4f}, this epoch {now - prev:.4f}, '
                f'total {now - start:.4f}',
                flush=True,
            )
            prev = now

        result = test(model, data.x, data.edge_index, data.y, split)
        results.append(result)
        print(
            f'(E-run) | run={run + 1} seed={run_seed} '
            f'TestAcc={result["TestAcc"]:.4f}, '
            f'F1Mi={result["F1Mi"]:.4f}, F1Ma={result["F1Ma"]:.4f}, '
            f'ValAcc={result["ValAcc"]:.4f}, C={result["C"]:.6g}',
            flush=True,
        )

    for key in ('TestAcc', 'F1Mi', 'F1Ma'):
        values = np.asarray([result[key] for result in results])
        print(
            f'(E-final) | {key}={values.mean():.4f}+-{values.std():.4f} '
            f'({args.runs} pretraining runs)',
            flush=True,
        )
