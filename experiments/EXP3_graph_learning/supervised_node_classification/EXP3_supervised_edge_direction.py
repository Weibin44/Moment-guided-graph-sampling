"""Evaluate moment-delta edge directions against matched random deletion."""
from __future__ import annotations
import time
from mggs.datasets import canonical_dataset_name
from experiments.EXP3_graph_learning.supervised_node_classification.supervised_common import eval_snapshots
from experiments.EXP3_graph_learning.supervised_node_classification.direction_snapshots import (
    DELETE_LABEL_SCOPE, classify_original_edges, filtered_direction_pools, sampled_snapshot, generate_snapshots,
)
from experiments.EXP3_graph_learning.supervised_node_classification.direction_reporting import (
    paired_results, summarize_groups, merge_eval_shards, analyze,
)
from experiments.EXP3_graph_learning.supervised_node_classification.direction_config import build_parser

def main():
    args = build_parser().parse_args()
    args.dataset = canonical_dataset_name(args.dataset)
    if args.run_name is None and args.stage in {"all", "generate"}:
        args.run_name = (
            f"{args.dataset}_{args.split}_{DELETE_LABEL_SCOPE}_edge_direction_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )
    if args.directions < 1 or args.graph_repeats < 1:
        raise ValueError("Directions and repeat counts must be positive.")
    if args.direction_ids is not None:
        invalid = [
            direction_id
            for direction_id in args.direction_ids
            if direction_id < 0 or direction_id >= args.directions
        ]
        if invalid:
            raise ValueError(f"Invalid --direction-ids: {invalid}")
    if args.stage == "all" and args.eval_num_shards > 1:
        raise ValueError(
            "Sharded evaluation requires separate generate, eval, and analyze stages."
        )
    if args.stage in {"all", "generate"}:
        generate_snapshots(args)
    if args.stage in {"all", "eval"}:
        eval_snapshots(args)
    if args.stage in {"all", "analyze"}:
        analyze(args)


if __name__ == "__main__":
    main()
