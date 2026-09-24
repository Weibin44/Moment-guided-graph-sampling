"""Exact target-moment graph banks for GCL node classification.

Train with an original-graph view and a bank-sampled view, then evaluate
unmasked embeddings on the original graph. Targets form a rectangular
grid or are specified explicitly as one (m2, m3) pair."""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mggs_matplotlib_cache")
os.environ.setdefault("PYG_HOME", "/tmp/mggs_pyg_cache")
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mggs.io import load_csv_rows as read_rows, write_csv_rows as write_rows
from mggs.io.results import write_rows_atomic, write_json, write_json_atomic
from mggs.paths import UNSUPERVISED_NODE_CLASSIFICATION_RECORDS_DIR
from mggs.reproducibility import seed_everything
from experiments.EXP3_graph_learning.unsupervised_node_classification.config import (
    build_parser, validate_args,
)
from experiments.EXP3_graph_learning.unsupervised_node_classification.targets import (
    load_graph, rectangular_grid_targets, resolve_budgets, select_points, prepare_targets,
)
from experiments.EXP3_graph_learning.unsupervised_node_classification.sampling_protocol import (
    update_state, candidate_edit, best_target_edit, sample_target,
)
from experiments.EXP3_graph_learning.unsupervised_node_classification.graph_bank import (
    prepare_graph_bank,
)
from experiments.EXP3_graph_learning.unsupervised_node_classification.encoder import (
    GConv, drop_feature, OriginalMomentEncoder,
)
from experiments.EXP3_graph_learning.unsupervised_node_classification.training import (
    evaluate_embeddings, train_point, resolve_device, evaluate_points,
)
from experiments.EXP3_graph_learning.unsupervised_node_classification.reporting import (
    analyze, plot_landscape,
)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    args.output_dir = args.output_dir or (
        UNSUPERVISED_NODE_CLASSIFICATION_RECORDS_DIR
        / args.dataset
        / "moments_landscape"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", vars(args))

    data, base_state = load_graph(args.dataset, args.data_root)
    rows = prepare_targets(args, args.output_dir, base_state)
    if args.prepare_only:
        return
    if args.prepare_bank_only:
        for row in rows:
            prepare_graph_bank(args.output_dir, row, base_state, args)
        return
    evaluations = evaluate_points(
        args, args.output_dir, rows, data, base_state
    )
    if not args.defer_analysis:
        analyze(args, args.output_dir, evaluations)


if __name__ == "__main__":
    main()
