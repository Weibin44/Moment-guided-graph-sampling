#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-CGC}"
DATASETS="${DATASETS:-Cora}"
GPU_IDS="${GPU_IDS:-0 1 2}"
SHARDS_PER_DATASET="${SHARDS_PER_DATASET:-3}"
SPLIT="${SPLIT:-full}"
DIRECTIONS="${DIRECTIONS:-8}"
BUDGET_RATIOS="${BUDGET_RATIOS:-0.3 0.6 0.9}"
RUN_TAG="${RUN_TAG:-EXP3_edge_direction_$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-results/EXP3_graph_learning/supervised_node_classification/logs}"
MODULE="experiments.EXP3_graph_learning.supervised_node_classification.EXP3_supervised_edge_direction"

read -r -a DATASET_ARRAY <<< "${DATASETS}"
read -r -a GPU_ARRAY <<< "${GPU_IDS}"
read -r -a BUDGET_RATIO_ARRAY <<< "${BUDGET_RATIOS}"
required_gpus=$(( ${#DATASET_ARRAY[@]} * SHARDS_PER_DATASET ))
if [ "${#GPU_ARRAY[@]}" -lt "${required_gpus}" ]; then
  echo "Need ${required_gpus} GPU IDs, received ${#GPU_ARRAY[@]}." >&2
  exit 1
fi
mkdir -p "${LOG_DIR}"

if [ "${EXP3_EDGE_DIRECTION_WORKER:-0}" != "1" ]; then
  supervisor_log="${LOG_DIR}/${RUN_TAG}.supervisor.log"
  nohup env EXP3_EDGE_DIRECTION_WORKER=1 bash "$0" \
    >"${supervisor_log}" 2>&1 < /dev/null &
  echo "[started] pid=$! log=${supervisor_log}"
  exit 0
fi

run_stage() {
  local dataset="$1"
  local run_name="$2"
  local stage="$3"
  shift 3
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m "${MODULE}" \
    --stage "${stage}" \
    --dataset "${dataset}" \
    --split "${SPLIT}" \
    --run-name "${run_name}" \
    --directions "${DIRECTIONS}" \
    --budget-ratios "${BUDGET_RATIO_ARRAY[@]}" \
    --graph-repeats 5 \
    --eval-seeds 0 1 2 3 4 \
    --epochs 400 \
    --early-stop 50 \
    --eval-num-shards "${SHARDS_PER_DATASET}" \
    "$@"
}

run_names=()
generation_pids=()
for dataset in "${DATASET_ARRAY[@]}"; do
  run_name="${dataset}_${SPLIT}_protect_train_same_label_${RUN_TAG}"
  run_names+=("${run_name}")
  log_file="${LOG_DIR}/${run_name}.generate.log"
  run_stage "${dataset}" "${run_name}" generate --device cpu \
    >"${log_file}" 2>&1 &
  generation_pids+=("$!")
  echo "[generate started] ${dataset} pid=$! log=${log_file}"
done

failed=0
for pid in "${generation_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  echo "Snapshot generation failed; evaluation was not started." >&2
  exit 1
fi

evaluation_pids=()
gpu_cursor=0
for dataset_index in "${!DATASET_ARRAY[@]}"; do
  dataset="${DATASET_ARRAY[dataset_index]}"
  run_name="${run_names[dataset_index]}"
  for ((shard_index=0; shard_index<SHARDS_PER_DATASET; shard_index++)); do
    gpu_id="${GPU_ARRAY[gpu_cursor]}"
    gpu_cursor=$((gpu_cursor + 1))
    log_file="${LOG_DIR}/${run_name}.eval_shard_${shard_index}.log"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      run_stage "${dataset}" "${run_name}" eval \
        --device 0 \
        --eval-shard-index "${shard_index}"
    ) >"${log_file}" 2>&1 &
    evaluation_pids+=("$!")
    echo "[eval started] ${dataset} shard=${shard_index}/${SHARDS_PER_DATASET} gpu=${gpu_id} pid=$! log=${log_file}"
  done
done

failed=0
for pid in "${evaluation_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [ "${failed}" -ne 0 ]; then
  echo "At least one evaluation shard failed; merge was not started." >&2
  exit 1
fi

analysis_pids=()
for dataset_index in "${!DATASET_ARRAY[@]}"; do
  dataset="${DATASET_ARRAY[dataset_index]}"
  run_name="${run_names[dataset_index]}"
  log_file="${LOG_DIR}/${run_name}.analyze.log"
  run_stage "${dataset}" "${run_name}" analyze --device cpu \
    >"${log_file}" 2>&1 &
  analysis_pids+=("$!")
  echo "[analyze started] ${dataset} pid=$! log=${log_file}"
done

for pid in "${analysis_pids[@]}"; do
  wait "${pid}"
done

for dataset_index in "${!DATASET_ARRAY[@]}"; do
  echo "[complete] results/EXP3_graph_learning/supervised_node_classification/${DATASET_ARRAY[dataset_index]}/${run_names[dataset_index]}"
done
