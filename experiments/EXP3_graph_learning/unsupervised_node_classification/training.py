"""Graph-bank training and seeded linear evaluation for EXP3."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
import numpy as np

import sys
import torch
from torch.optim import Adam
from mggs.io import safe_torch_load, write_csv_rows as write_rows
from mggs.moments import TopologyMomentState
from mggs.reproducibility import seed_everything
from .encoder import OriginalMomentEncoder
from .graph_bank import prepare_graph_bank

# The vendored baseline uses absolute sibling imports; keep it outside mggs.
_SPAN_DIR = Path(__file__).resolve().parent / "GCL-SPAN"
if str(_SPAN_DIR) not in sys.path:
    sys.path.insert(0, str(_SPAN_DIR))
from ContrastMode import DualBranchContrast
from Evaluator import LREvaluator, get_split
from Loss import JSD

Row = dict[str, Any]


def evaluate_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> Row:
    split = get_split(embeddings.size(0), train_ratio=0.1, test_ratio=0.8, seed=seed)
    best = None
    for decay in args.eval_weight_decays:
        result = LREvaluator(
            num_epochs=args.eval_epochs,
            learning_rate=args.eval_learning_rate,
            weight_decay=decay,
            test_interval=args.eval_interval,
            seed=seed,
        )(embeddings, labels, split)
        print(
            f"[linear-eval] weight_decay={decay:g} "
            f"val={result['accuracy_val']:.4f} test={result['accuracy']:.4f}",
            flush=True,
        )
        if best is None or result["accuracy_val"] > best["accuracy_val"]:
            best = {**result, "eval_weight_decay": decay}
    assert best is not None
    return {
        "test_accuracy": best["accuracy"],
        "val_accuracy": best["accuracy_val"],
        **{key: best[key] for key in ("micro_f1", "macro_f1", "eval_weight_decay")},
    }

def train_point(
    x: torch.Tensor,
    labels: torch.Tensor,
    original_edges: torch.Tensor,
    row: Row,
    seed: int,
    args: argparse.Namespace,
    graph_bank: list[tuple[torch.Tensor, Row]],
) -> Row:
    if not graph_bank:
        raise RuntimeError(f"Missing graph bank for {row['point_id']}.")
    seed_everything(seed)
    model = OriginalMomentEncoder(x.size(1), args.hidden_dim, args.pf).to(x.device)
    contrast = DualBranchContrast(loss=JSD(), mode="G2L").to(x.device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    bank_rng = np.random.default_rng(args.sampling_seed + seed * 1_000_003)
    trace = []
    for epoch in range(args.epochs):
        bank_index = int(bank_rng.integers(len(graph_bank)))
        augmented_edges, entry = graph_bank[bank_index]
        trace.append([
            float(entry[key]) for key in ("m2", "m3", "add_count", "delete_count")
        ])
        model.train()
        optimizer.zero_grad()
        z1, z2, g1, g2, z1n, z2n = model(x, original_edges, augmented_edges)
        loss = contrast(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
        loss.backward()
        optimizer.step()
        if epoch == 0 or (epoch + 1) % args.log_interval == 0:
            print(f"[epoch] {epoch + 1}/{args.epochs} loss={loss.item():.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        z1, z2 = model.encode_clean(x, original_edges)
        embeddings = torch.cat((z1, z2), dim=1).detach().cpu()
    metrics = evaluate_embeddings(embeddings.to(x.device), labels, args, seed)

    observations = np.asarray(trace)
    for index, key in enumerate(("m2", "m3", "add_count", "delete_count")):
        metrics[f"observed_{key}"] = float(observations[:, index].mean())
        if key in ("m2", "m3"):
            metrics[f"observed_{key}_std"] = float(observations[:, index].std())
    metrics["encoder_epoch"] = args.epochs
    return metrics

def resolve_device(device_name: str) -> torch.device:
    requested = (
        device_name
        if device_name == "cpu" or device_name.startswith("cuda")
        else f"cuda:{device_name}"
    )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested} requested but CUDA is unavailable."
        )
    return torch.device(requested)

def evaluate_points(
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[Row],
    data,
    base_state: TopologyMomentState,
) -> list[Row]:
    device = resolve_device(args.device)
    x = data.x.to(device)
    labels = data.y.to(device)
    original_edges = data.edge_index.to(device)
    results: list[Row] = []
    for point_index, row in enumerate(rows):
        bank_entries = prepare_graph_bank(output_dir, row, base_state, args)
        bank_dir = output_dir / "graph_bank" / str(row["point_id"])
        graph_bank = [
            (
                safe_torch_load(bank_dir / str(entry["snapshot"]))["edge_index"].to(device),
                entry,
            )
            for entry in bank_entries
        ]
        for seed in args.seeds:
            print(
                f"[train] {point_index + 1}/{len(rows)} "
                f"{row['point_id']} seed={seed}",
                flush=True,
            )
            metrics = train_point(x, labels, original_edges, row, seed, args, graph_bank)
            results.append({
                **row,
                "base_m2": base_state.m2,
                "base_m3": base_state.m3,
                "base_edge_count": len(base_state.edge_list),
                "seed": seed,
                **metrics,
            })
    write_rows(output_dir / "evaluations.csv", results)
    return results
