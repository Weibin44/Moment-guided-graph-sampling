#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-CGC}"
MODE="${MODE:-main}"
DATASETS="${DATASETS:-Cora CiteSeer}"
GPU_IDS="${GPU_IDS:-2 5 6 7 8 9}"
SEEDS="${SEEDS:-15 16 17 18 19 20 21 22 23 24}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/results/EXP3_graph_learning/unsupervised_node_classification}"

NUM_TARGETS=37
ORIGINAL_MOMENTS_TARGET=36
GRID_M2_COUNT=6
GRID_M3_COUNT=6
M2_RELATIVE_RANGE=0.5
M3_RELATIVE_RANGE=0.5
EDIT_RATIO=0.2
GRAPH_BANK_SIZE=16
CANDIDATE_ADD=1000
CANDIDATE_DEL=1000
SAMPLING_SEED=15
HIDDEN_DIM=512
ENCODER_LR=0.0001
PF=0.4
EVAL_EPOCHS=5000
EVAL_LR=0.001
EVAL_WEIGHT_DECAYS="0.0 0.0005 0.001 0.005"

read -r -a DATASET_ARRAY <<<"${DATASETS}"
read -r -a GPU_ARRAY <<<"${GPU_IDS}"
read -r -a SEED_ARRAY <<<"${SEEDS}"
read -r -a EVAL_WD_ARRAY <<<"${EVAL_WEIGHT_DECAYS}"

case "${MODE}" in
  main) TARGET_ARRAY=("${ORIGINAL_MOMENTS_TARGET}") ;;
  landscape)
    TARGET_ARRAY=()
    for ((target_index = 0; target_index < NUM_TARGETS; target_index++)); do
      TARGET_ARRAY+=("${target_index}")
    done
    ;;
  *) echo "MODE must be main or landscape, received: ${MODE}" >&2; exit 2 ;;
esac
if [ "${#GPU_ARRAY[@]}" -eq 0 ] || [ "${#SEED_ARRAY[@]}" -eq 0 ]; then
  echo "GPU_IDS and SEEDS must not be empty." >&2
  exit 2
fi

if [ "${EXP3_MOMENTS_ORIGINAL_WORKER:-0}" != "1" ]; then
  mkdir -p "${RUN_ROOT}/logs"
  supervisor_log="${RUN_ROOT}/logs/${MODE}_$(date -u +%Y%m%d_%H%M%S).log"
  nohup env EXP3_MOMENTS_ORIGINAL_WORKER=1 bash "$0" \
    >"${supervisor_log}" 2>&1 < /dev/null &
  echo "[started] mode=${MODE} pid=$! log=${supervisor_log} results=${RUN_ROOT}"
  exit 0
fi

dataset_epochs() {
  case "$1" in
    Cora) printf '2500' ;;
    CiteSeer) printf '1000' ;;
    *) echo "Unsupported dataset: $1" >&2; return 2 ;;
  esac
}

cache_root() {
  printf '%s/data/%s/moments_gcl/mixed20_grid37_bank16' "${REPO_ROOT}" "$1"
}

prepare_points() {
  local dataset="$1" root points row_count
  root="$(cache_root "${dataset}")"
  points="${root}/points.csv"
  if [ -f "${points}" ]; then
    row_count="$(($(wc -l < "${points}") - 1))"
    if [ "${row_count}" -ne "${NUM_TARGETS}" ]; then
      echo "${points}: expected ${NUM_TARGETS} targets, found ${row_count} rows." >&2
      return 2
    fi
    echo "[bank:points:reuse] ${dataset} ${points}"
    return
  fi

  mkdir -p "${root}"
  echo "[bank:points:create] ${dataset} grid=6x6+original-moments"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m experiments.EXP3_graph_learning.unsupervised_node_classification.EXP3_moments_gcl \
      --dataset "${dataset}" \
      --data-root "${REPO_ROOT}/data" \
      --output-dir "${root}" \
      --device cpu \
      --grid-m2-count "${GRID_M2_COUNT}" \
      --grid-m3-count "${GRID_M3_COUNT}" \
      --m2-relative-range "${M2_RELATIVE_RANGE}" \
      --m3-relative-range "${M3_RELATIVE_RANGE}" \
      --include-origin-target \
      --max-edit-ratio "${EDIT_RATIO}" \
      --candidate-add "${CANDIDATE_ADD}" \
      --candidate-del "${CANDIDATE_DEL}" \
      --graph-bank-size "${GRAPH_BANK_SIZE}" \
      --sampling-seed "${SAMPLING_SEED}" \
      --prepare-only
}

run_target() {
  local gpu_id="$1" dataset="$2" target_index="$3"
  local target_name output_dir cache points epochs log
  printf -v target_name 'target_%03d' "${target_index}"
  output_dir="${RUN_ROOT}/${dataset}/bank/ratio_0.2/shards/${target_name}"
  cache="$(cache_root "${dataset}")"
  points="${cache}/points.csv"
  epochs="$(dataset_epochs "${dataset}")"
  log="${output_dir}/run.log"

  if [ -f "${output_dir}/.done" ]; then
    echo "[skip] ${dataset} ${target_name}"
    return
  fi
  mkdir -p "${output_dir}"
  if [ ! -e "${output_dir}/graph_bank" ] && [ ! -L "${output_dir}/graph_bank" ]; then
    ln -s "${cache}/graph_bank" "${output_dir}/graph_bank"
  fi
  echo "[run] gpu=${gpu_id} ${dataset} ${target_name} seeds=${SEEDS} epochs=${epochs}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m experiments.EXP3_graph_learning.unsupervised_node_classification.EXP3_moments_gcl \
      --dataset "${dataset}" \
      --data-root "${REPO_ROOT}/data" \
      --output-dir "${output_dir}" \
      --prepared-points "${points}" \
      --device 0 \
      --grid-m2-count "${GRID_M2_COUNT}" \
      --grid-m3-count "${GRID_M3_COUNT}" \
      --m2-relative-range "${M2_RELATIVE_RANGE}" \
      --m3-relative-range "${M3_RELATIVE_RANGE}" \
      --include-origin-target \
      --target-indices "${target_index}" \
      --max-edit-ratio "${EDIT_RATIO}" \
      --candidate-add "${CANDIDATE_ADD}" \
      --candidate-del "${CANDIDATE_DEL}" \
      --graph-bank-size "${GRAPH_BANK_SIZE}" \
      --sampling-seed "${SAMPLING_SEED}" \
      --seeds "${SEED_ARRAY[@]}" \
      --epochs "${epochs}" \
      --hidden-dim "${HIDDEN_DIM}" \
      --learning-rate "${ENCODER_LR}" \
      --pf "${PF}" \
      --log-interval 100 \
      --eval-epochs "${EVAL_EPOCHS}" \
      --eval-learning-rate "${EVAL_LR}" \
      --eval-interval 20 \
      --eval-weight-decays "${EVAL_WD_ARRAY[@]}" \
      --defer-analysis >"${log}" 2>&1
  touch "${output_dir}/.done"
  echo "[done] ${dataset} ${target_name}"
}

run_queue() {
  local slot="$1" gpu_id="$2" task_index=0 dataset target_index
  for dataset in "${DATASET_ARRAY[@]}"; do
    for target_index in "${TARGET_ARRAY[@]}"; do
      if [ "$((task_index % ${#GPU_ARRAY[@]}))" -eq "${slot}" ]; then
        run_target "${gpu_id}" "${dataset}" "${target_index}"
      fi
      task_index=$((task_index + 1))
    done
  done
}

for dataset in "${DATASET_ARRAY[@]}"; do
  prepare_points "${dataset}"
done

echo "[config] mode=${MODE} datasets=${DATASETS} GPUs=${GPU_IDS} seeds=${SEEDS}"
worker_pids=()
for slot in "${!GPU_ARRAY[@]}"; do
  run_queue "${slot}" "${GPU_ARRAY[slot]}" &
  worker_pids+=("$!")
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [ "${status}" -ne 0 ]; then
  echo "[failed] Inspect target logs under ${RUN_ROOT}." >&2
  exit "${status}"
fi

for dataset in "${DATASET_ARRAY[@]}"; do
  merge_args=()
  if [ "${MODE}" = "main" ]; then
    merge_args=(--target-indices "${ORIGINAL_MOMENTS_TARGET}")
  fi
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m experiments.EXP3_graph_learning.unsupervised_node_classification.EXP3_merge_moments_landscape \
      --root "${RUN_ROOT}" \
      --datasets "${dataset}" \
      --modes bank \
      --ratios "${EDIT_RATIO}" \
      --num-targets "${NUM_TARGETS}" \
      --num-seeds "${#SEED_ARRAY[@]}" \
      "${merge_args[@]}"
done

if [ "${MODE}" = "landscape" ]; then
  touch "${RUN_ROOT}/.landscape.done"
else
  touch "${RUN_ROOT}/.main.done"
fi
echo "[complete] mode=${MODE} results=${RUN_ROOT}"
