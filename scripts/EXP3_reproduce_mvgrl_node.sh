#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MVGRL_DIR="${REPO_ROOT}/experiments/EXP3_graph_learning/unsupervised_node_classification/MVGRL"
CONDA_ENV="${CONDA_ENV:-CGC}"
RUNS="${RUNS:-10}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/results/EXP3_graph_learning/unsupervised_node_classification/logs}"
CHECKPOINT_ROOT="${REPO_ROOT}/results/EXP3_graph_learning/unsupervised_node_classification/MVGRL/checkpoints_10_10_80"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_ROOT}"
launch() {
  local dataset="$1"
  local gpu_id="$2"
  local log_file="${LOG_DIR}/MVGRL_${dataset}_$(date -u +%Y%m%d_%H%M%S_%N).log"
  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -u "${MVGRL_DIR}/node/train.py" \
      --dataset "${dataset}" \
      --data-root "${REPO_ROOT}/data" \
      --gpu-id 0 \
      --runs "${RUNS}" \
      --seed 15 \
      --checkpoint-dir "${CHECKPOINT_ROOT}/${dataset}" \
      --verbose \
    >"${log_file}" 2>&1 < /dev/null &
  echo "[started] method=MVGRL dataset=${dataset} gpu=${gpu_id} pid=$! log=${log_file}"
}

launch cora 6
launch citeseer 7
