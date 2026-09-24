"""Repository-local paths used by reproducible experiment scripts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "data"
RECORDS_DIR = REPO_ROOT / "results"
SUPERVISED_NODE_CLASSIFICATION_RECORDS_DIR = (
    RECORDS_DIR / "EXP3_graph_learning" / "supervised_node_classification"
)
UNSUPERVISED_NODE_CLASSIFICATION_RECORDS_DIR = (
    RECORDS_DIR / "EXP3_graph_learning" / "unsupervised_node_classification"
)

__all__ = [
    "REPO_ROOT",
    "DATASET_DIR",
    "RECORDS_DIR",
    "SUPERVISED_NODE_CLASSIFICATION_RECORDS_DIR",
    "UNSUPERVISED_NODE_CLASSIFICATION_RECORDS_DIR",
]
