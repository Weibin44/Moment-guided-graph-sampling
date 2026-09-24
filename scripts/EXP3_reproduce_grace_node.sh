#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GRACE_DIR="${REPO_ROOT}/experiments/EXP3_graph_learning/unsupervised_node_classification/GRACE"
CONDA_ENV="${CONDA_ENV:-CGC}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/results/EXP3_graph_learning/unsupervised_node_classification/logs}"
CORA_GPU_ID="${CORA_GPU_ID:-4}"
CITESEER_GPU_ID="${CITESEER_GPU_ID:-5}"
DATASETS="${DATASETS:-Cora CiteSeer}"

mkdir -p "${LOG_DIR}"
launch() {
  local dataset="$1"
  local gpu_id="$2"
  local log_file="${LOG_DIR}/GRACE_${dataset}_$(date -u +%Y%m%d_%H%M%S_%N).log"
  nohup env CUDA_VISIBLE_DEVICES="${gpu_id}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${GRACE_DIR}/train.py" \
      --dataset "${dataset}" \
      --gpu_id 0 \
      --config "${GRACE_DIR}/config.yaml" \
      --data-root "${REPO_ROOT}/data" \
      --runs 10 \
      --seed 15 \
    >"${log_file}" 2>&1 < /dev/null &
  echo "[started] method=GRACE dataset=${dataset} gpu=${gpu_id} pid=$! log=${log_file}"
}

for dataset in ${DATASETS}; do
  case "${dataset}" in
    Cora)
      launch Cora "${CORA_GPU_ID}"
      ;;
    CiteSeer)
      launch CiteSeer "${CITESEER_GPU_ID}"
      ;;
    *)
      echo "Unsupported dataset: ${dataset}. Expected Cora or CiteSeer." >&2
      exit 2
      ;;
  esac
done
