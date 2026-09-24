"""Dataset naming and loading for bundled PyG experiments."""

import gzip
import urllib.request
from pathlib import Path

import numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import Actor, Amazon, Planetoid, WebKB

from .io.graph import edge_index_from_edges
from .paths import DATASET_DIR


WEBKB_NAMES = {"texas": "Texas", "cornell": "Cornell", "wisconsin": "Wisconsin"}
PLANETOID_NAMES = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
AMAZON_NAMES = {"photo": "Photo", "computers": "Computers"}
ACTOR_NAMES = {"actor": "Actor", "film": "Actor"}
SNAP_NAMES = {"cagrqc": "ca-GrQc"}
SNAP_URLS = {"ca-GrQc": "https://snap.stanford.edu/data/ca-GrQc.txt.gz"}


def _key(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def canonical_dataset_name(name: str) -> str:
    key = _key(name)
    for names in (WEBKB_NAMES, PLANETOID_NAMES, AMAZON_NAMES, ACTOR_NAMES, SNAP_NAMES):
        if key in names:
            return names[key]
    supported = (
        list(WEBKB_NAMES.values()) + list(PLANETOID_NAMES.values())
        + list(AMAZON_NAMES.values()) + sorted(set(ACTOR_NAMES.values()))
        + list(SNAP_NAMES.values())
    )
    raise ValueError(f"Unsupported dataset: {name}. Use one of: {', '.join(supported)}.")


def load_graph_dataset(name: str):
    canonical = canonical_dataset_name(name)
    key = _key(canonical)
    if key in WEBKB_NAMES:
        return WebKB(root=str(DATASET_DIR), name=canonical)[0]
    if key in PLANETOID_NAMES:
        return Planetoid(root=str(DATASET_DIR), name=canonical)[0]
    if key in AMAZON_NAMES:
        return Amazon(root=str(DATASET_DIR), name=canonical)[0]
    if key in ACTOR_NAMES:
        return Actor(root=str(DATASET_DIR))[0]
    if key in SNAP_NAMES:
        return _load_snap_undirected_graph(canonical)
    raise ValueError(f"Unsupported dataset: {name}")


def _load_snap_undirected_graph(name: str) -> Data:
    """Download and load a SNAP undirected edge list as a PyG graph."""

    url = SNAP_URLS[name]
    raw_dir = DATASET_DIR / name / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / Path(url).name
    if not raw_path.exists():
        print(f"[download] {url}", flush=True)
        urllib.request.urlretrieve(url, raw_path)

    edges: set[tuple[int, int]] = set()
    with gzip.open(raw_path, "rt", encoding="utf-8") as file:
        for line in file:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            source, target = map(int, fields[:2])
            if source != target:
                edges.add((min(source, target), max(source, target)))

    if not edges:
        raise ValueError(f"SNAP dataset {name} contains no usable edges: {raw_path}")
    external_edges = np.asarray(sorted(edges), dtype=np.int64)
    node_ids, inverse = np.unique(external_edges.reshape(-1), return_inverse=True)
    contiguous_edges = inverse.reshape(-1, 2)
    edge_index = edge_index_from_edges(
        ((int(u), int(v)) for u, v in contiguous_edges), len(node_ids)
    )
    return Data(edge_index=edge_index, num_nodes=len(node_ids))
