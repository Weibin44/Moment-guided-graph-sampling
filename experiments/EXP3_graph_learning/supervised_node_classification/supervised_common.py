"""Snapshot and supervised-GCN evaluation utilities for EXP3."""

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from mggs.datasets import PLANETOID_NAMES, canonical_dataset_name
from mggs.io import append_csv_row, load_csv_rows, safe_torch_load
from mggs.moments import TopologyMomentState, estimate_moments
from mggs.paths import SUPERVISED_NODE_CLASSIFICATION_RECORDS_DIR
from mggs.reproducibility import seed_everything


class SupervisedGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def load_planetoid(dataset_name, root, split):
    name = canonical_dataset_name(dataset_name)
    if name not in PLANETOID_NAMES.values():
        raise ValueError(f"{name} is not a Planetoid dataset.")
    return Planetoid(
        str(root),
        name=name,
        split=split,
        transform=T.NormalizeFeatures(),
    )


def parse_device(device_arg):
    if device_arg == "cpu":
        device = torch.device("cpu")
        print("[device] using cpu")
        return device

    if device_arg.isdigit():
        requested = f"cuda:{device_arg}"
    else:
        requested = device_arg

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{requested}' was requested, but torch.cuda.is_available() "
                "is False in this process. Check the launch environment, CUDA driver, "
                "and whether /dev/nvidia* is visible."
            )
        device = torch.device(requested)
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {requested}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        print(f"[device] using cuda:{index} - {torch.cuda.get_device_name(index)}")
        return device

    device = torch.device(requested)
    print(f"[device] using {device}")
    return device


def one_dim_mask(mask):
    return mask[:, 0] if mask.dim() > 1 else mask


def run_root(dataset_name, run_name):
    return (
        SUPERVISED_NODE_CLASSIFICATION_RECORDS_DIR
        / canonical_dataset_name(dataset_name)
        / run_name
    )


def resolve_run_name(dataset_name, run_name, split=None):
    if run_name:
        return run_name
    dataset_root = (
        SUPERVISED_NODE_CLASSIFICATION_RECORDS_DIR
        / canonical_dataset_name(dataset_name)
    )
    candidates = [p for p in dataset_root.glob("*") if p.is_dir()]
    if split is not None:
        split_key = f"_{split}_"
        split_candidates = [p for p in candidates if split_key in p.name]
        if split_candidates:
            candidates = split_candidates
    if not candidates:
        raise FileNotFoundError(
            f"No runs found under {dataset_root}. Run the generate command first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime).name


def base_graph_from_data(data):
    """Collapse PyG edge_index into the simple undirected graph used by TopologyMomentState."""
    base_state = TopologyMomentState(data.edge_index.cpu(), data.num_nodes)
    base_edge_index = base_state.to_edge_index().cpu()
    _, m2, m3, _ = estimate_moments(base_edge_index, data.num_nodes, method="exact")
    base_state.m2 = m2
    base_state.m3 = m3
    return base_state, base_edge_index, float(m2), float(m3)


def protected_delete_edges(data, base_state):
    """Allow every deletion except same-label train-to-train edges."""
    labels = data.y.cpu().numpy()
    train_mask = one_dim_mask(data.train_mask).cpu().numpy().astype(bool)
    return {
        (int(u), int(v))
        for u, v in base_state.edge_list
        if not (train_mask[u] and train_mask[v] and labels[u] == labels[v])
    }


def save_snapshot(edge_index, snapshot_id, metadata, snapshots_dir):
    """Save graph topology while keeping its metadata in the run-level CSV."""
    row = {"snapshot_id": snapshot_id, **metadata}
    snapshot_path = snapshots_dir / f"{snapshot_id}.pt"
    row["snapshot_path"] = str(snapshot_path.relative_to(snapshots_dir.parent))
    torch.save({"edge_index": edge_index.cpu()}, snapshot_path)
    return row


def metric_from_logits(logits, y, mask):
    pred = logits.argmax(dim=-1)
    return float((pred[mask] == y[mask]).float().mean().item())


def train_one_snapshot(data, edge_index, dataset, args, seed, device):
    seed_everything(seed)
    model = SupervisedGCN(
        in_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=dataset.num_classes,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    x = data.x.to(device)
    y = data.y.to(device)
    train_mask = one_dim_mask(data.train_mask).to(device)
    val_mask = one_dim_mask(data.val_mask).to(device)
    test_mask = one_dim_mask(data.test_mask).to(device)
    edge_index = edge_index.to(device)

    best = {
        "best_epoch": 0,
        "val_acc": -1.0,
        "test_acc": 0.0,
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        logits = model(x, edge_index)
        val = metric_from_logits(logits, y, val_mask)
        test = metric_from_logits(logits, y, test_mask)
        if val > best["val_acc"]:
            best = {
                "best_epoch": int(epoch),
                "val_acc": val,
                "test_acc": test,
            }
        elif (
            args.early_stop > 0
            and epoch - best["best_epoch"] >= args.early_stop
        ):
            break

    return best


def eval_snapshots(args):
    args.run_name = resolve_run_name(args.dataset, args.run_name, args.split)
    out_root = run_root(args.dataset, args.run_name)
    metadata_path = out_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    dataset = load_planetoid(args.dataset, args.dataset_root, args.split)
    data = dataset[0]
    device = parse_device(args.device)
    metadata = load_csv_rows(metadata_path)
    if args.max_eval_snapshots > 0:
        metadata = metadata[:args.max_eval_snapshots]

    num_shards = args.eval_num_shards
    shard_index = args.eval_shard_index
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(
            "--eval-num-shards must be positive and --eval-shard-index must "
            "be in [0, eval_num_shards)."
        )
    if num_shards > 1:
        metadata = [
            row
            for index, row in enumerate(metadata)
            if index % num_shards == shard_index
        ]

    eval_filename = (
        f"eval_results.shard_{shard_index:02d}_of_{num_shards:02d}.csv"
        if num_shards > 1
        else "eval_results.csv"
    )
    eval_path = out_root / eval_filename
    existing = set()
    if eval_path.exists() and not args.overwrite_eval:
        for row in load_csv_rows(eval_path):
            existing.add((row["snapshot_id"], int(row["seed"])))
    elif eval_path.exists() and args.overwrite_eval:
        eval_path.unlink()

    fieldnames = [
        "dataset", "split", "run_name", "snapshot_id", "direction_id", "snapshot_idx",
        "seed", "best_epoch", "val_acc", "test_acc",
    ]
    total = len(metadata) * len(args.eval_seeds)
    if num_shards > 1:
        print(
            f"[eval] shard {shard_index + 1}/{num_shards}: "
            f"{len(metadata)} snapshots, {total} trainings"
        )
    done = 0
    for meta in metadata:
        snapshot_path = out_root / meta["snapshot_path"]
        snapshot = safe_torch_load(snapshot_path, map_location="cpu")
        edge_index = snapshot["edge_index"]
        for seed in args.eval_seeds:
            key = (meta["snapshot_id"], int(seed))
            if key in existing:
                continue
            metrics = train_one_snapshot(
                data, edge_index, dataset, args, int(seed), device
            )
            row = {
                "dataset": canonical_dataset_name(args.dataset),
                "split": args.split,
                "run_name": args.run_name,
                "snapshot_id": meta["snapshot_id"],
                "direction_id": meta["direction_id"],
                "snapshot_idx": meta["snapshot_idx"],
                "seed": int(seed),
                **metrics,
            }
            append_csv_row(eval_path, row, fieldnames)
            done += 1
            print(
                f"[eval] {done}/{total} {meta['snapshot_id']} seed={seed} "
                f"val={metrics['val_acc']:.4f} test={metrics['test_acc']:.4f}"
            )
    print(f"[eval] results saved to {eval_path}")
