"""Citation data loader using the repository's existing PyG datasets."""

from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

try:
    from ..utils import compute_ppr, normalize_adj, preprocess_features
except ImportError:  # Support direct execution from the original repository layout.
    from utils import compute_ppr, normalize_adj, preprocess_features


def _mask_indices(mask):
    return np.flatnonzero(mask.cpu().numpy())


def _build_cache(dataset, data_root, cache_dir):
    pyg_name = "CiteSeer" if dataset == "citeseer" else "Cora"
    data = Planetoid(
        str(data_root), pyg_name, transform=T.NormalizeFeatures()
    )[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(data.edge_index.t().cpu().numpy().tolist())

    arrays = {
        "adj": nx.to_numpy_array(graph, dtype=np.float32),
        "diff": compute_ppr(graph, 0.2),
        "feat": data.x.cpu().numpy(),
        "labels": data.y.cpu().numpy(),
        "idx_train": _mask_indices(data.train_mask),
        "idx_val": _mask_indices(data.val_mask),
        "idx_test": _mask_indices(data.test_mask),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(cache_dir / f"{name}.npy", value)
    return arrays


def load(dataset, data_root):
    if dataset not in {"cora", "citeseer"}:
        raise ValueError("Only cora and citeseer are supported.")
    data_root = Path(data_root).expanduser().resolve()
    cache_dir = data_root / "MVGRL" / dataset
    names = ("adj", "diff", "feat", "labels", "idx_train", "idx_val", "idx_test")
    if all((cache_dir / f"{name}.npy").exists() for name in names):
        arrays = {name: np.load(cache_dir / f"{name}.npy") for name in names}
    else:
        arrays = _build_cache(dataset, data_root, cache_dir)

    adj = arrays["adj"]
    diff = arrays["diff"]
    feat = arrays["feat"]
    if dataset == "citeseer":
        feat = preprocess_features(feat)
        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([
            abs(avg_degree - np.argwhere(diff >= value).shape[0] / diff.shape[0])
            for value in epsilons
        ])]
        diff[diff < epsilon] = 0.0
        diff = MinMaxScaler().fit_transform(diff)

    normalized_adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    return (
        normalized_adj, diff, feat, arrays["labels"], arrays["idx_train"],
        arrays["idx_val"], arrays["idx_test"],
    )
