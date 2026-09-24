#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPAN_DIR="${REPO_ROOT}/experiments/EXP3_graph_learning/unsupervised_node_classification/GCL-SPAN"

CONDA_ENV="${CONDA_ENV:-CGC}"
GPU_ID="${GPU_ID:-}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASETS="${DATASETS:-Cora CiteSeer}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/results/EXP3_graph_learning/unsupervised_node_classification/logs}"
RUNS="${RUNS:-10}"
EVAL_VIEW="${EVAL_VIEW:-official}"

if [ "${GCL_SPAN_WORKER:-0}" != "1" ]; then
  mkdir -p "${LOG_DIR}"
  read -r -a DATASET_ARRAY <<< "${DATASETS}"
  for dataset in "${DATASET_ARRAY[@]}"; do
    case "${dataset,,}" in
      cora)
        dataset="Cora"
        dataset_gpu="${CORA_GPU_ID:-${GPU_ID:-2}}"
        ;;
      citeseer)
        dataset="CiteSeer"
        dataset_gpu="${CITESEER_GPU_ID:-${GPU_ID:-5}}"
        ;;
      *)
        echo "Unsupported dataset: ${dataset}" >&2
        exit 2
        ;;
    esac
    log_file="${LOG_DIR}/gcl_span_${dataset}_$(date -u +%Y%m%d_%H%M%S_%N).log"
    nohup env GCL_SPAN_WORKER=1 DATASETS="${dataset}" GPU_ID="${dataset_gpu}" \
      bash "$0" >"${log_file}" 2>&1 < /dev/null &
    echo "[started] dataset=${dataset} gpu=${dataset_gpu} pid=$! log=${log_file}"
  done
  exit 0
fi

cd "${SPAN_DIR}"
read -r -a DATASET_ARRAY <<< "${DATASETS}"
for dataset in "${DATASET_ARRAY[@]}"; do
  case "${dataset,,}" in
    cora)
      dataset="Cora"
      seed="${CORA_SEED:-15}"
      epoch="${CORA_EPOCH:-1000}"
      aug_lr1="${CORA_AUG_LR1:-100}"
      aug_lr2="${CORA_AUG_LR2:-0.1}"
      aug_iter="${CORA_AUG_ITER:-80}"
      pf="${CORA_PF:-0.}"
      pe="${CORA_PE:-0.1}"
      classifier_lr="${CORA_CLASSIFIER_LR:-${CLASSIFIER_LR:-0.001}}"
      classifier_epoch="${CORA_CLASSIFIER_EPOCH:-${CLASSIFIER_EPOCH:-5000}}"
      ;;
    citeseer)
      dataset="CiteSeer"
      seed="${CITESEER_SEED:-15}"
      epoch="${CITESEER_EPOCH:-1000}"
      aug_lr1="${CITESEER_AUG_LR1:-100}"
      aug_lr2="${CITESEER_AUG_LR2:-0.1}"
      aug_iter="${CITESEER_AUG_ITER:-80}"
      pf="${CITESEER_PF:-0.4}"
      pe="${CITESEER_PE:-0.3}"
      classifier_lr="${CITESEER_CLASSIFIER_LR:-${CLASSIFIER_LR:-0.001}}"
      classifier_epoch="${CITESEER_CLASSIFIER_EPOCH:-${CLASSIFIER_EPOCH:-500}}"
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      exit 2
      ;;
  esac

  echo "[GCL-SPAN] dataset=${dataset} gpu=${GPU_ID} seed=${seed} runs=${RUNS} epoch=${epoch} aug_lr1=${aug_lr1} aug_lr2=${aug_lr2} aug_iter=${aug_iter} pf=${pf} pe=${pe} classifier_lr=${classifier_lr} classifier_epoch=${classifier_epoch} eval_view=${EVAL_VIEW}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" conda run --no-capture-output -n "${CONDA_ENV}" \
    python unsupervised_node.py \
      --dataset "${dataset}" \
      --data-root "${DATA_ROOT}" \
      --device 0 \
      --seed "${seed}" \
      --epoch "${epoch}" \
      --runs "${RUNS}" \
      --classifier_lr "${classifier_lr}" \
      --classifier_epoch "${classifier_epoch}" \
      --eval-view "${EVAL_VIEW}" \
      --aug_lr1 "${aug_lr1}" \
      --aug_lr2 "${aug_lr2}" \
      --aug_iter "${aug_iter}" \
      --pf "${pf}" \
      --pe "${pe}"
done
