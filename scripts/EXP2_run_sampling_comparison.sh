#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
MODE="${1:-efficiency}"
case "$MODE" in efficiency|preservation|all|smoke) ;; *)
  echo "Usage: bash $0 [efficiency|preservation|all|smoke]" >&2; exit 2 ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-results/EXP2_sampling_comparison/$(date -u +%Y%m%d_%H%M%S)}"
CONDA_ENV="${CONDA_ENV:-CGC}"
THREADS="${THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mggs_matplotlib_cache}"
NORMALIZATION="${NORMALIZATION:-graph_relative}"
read -r -a EFFICIENCY_ORDERS_ARRAY <<< "${EFFICIENCY_ORDERS:-2,3 2,3,4}"
read -r -a PRESERVATION_ORDERS_ARRAY <<< "${PRESERVATION_ORDERS:-2,3,4}"
read -r -a BENCHMARK_TOP_KS_ARRAY <<< "${BENCHMARK_TOP_KS:-5 25 50}"
read -r -a TOP_KS_ARRAY <<< "${TOP_KS:-5 25 50}"
read -r -a SEEDS_ARRAY <<< "${SEEDS:-0 1 2}"
mkdir -p "$OUTPUT_ROOT/logs"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

# Detach the preservation pipeline once; datasets remain sequential in the worker.
if [[ "$MODE" == preservation && "${PRESERVATION_FOREGROUND:-0}" != 1 ]]; then
  setsid env OUTPUT_ROOT="$OUTPUT_ROOT" PRESERVATION_FOREGROUND=1 \
    bash "$PROJECT_ROOT/scripts/EXP2_run_sampling_comparison.sh" preservation \
    >"$OUTPUT_ROOT/logs/preservation.log" 2>&1 < /dev/null &
  worker_pid=$!
  echo "$worker_pid" > "$OUTPUT_ROOT/preservation.pid"
  echo "[started] pid=$worker_pid"
  echo "[log] $OUTPUT_ROOT/logs/preservation.log"
  echo "[output] $OUTPUT_ROOT"
  exit 0
fi

run() {
  local name="$1"
  shift
  echo "[run] $name"
  conda run --no-capture-output -n "$CONDA_ENV" python "$@" 2>&1 | tee "$OUTPUT_ROOT/logs/$name.log"
}

efficiency() (
  export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
  run efficiency -m experiments.sampling_efficiency --benchmark \
    --dataset "${DATASET:-Cora}" --delete-budget "${DELETE_BUDGET:-100}" \
    --direct-trace-delete-budget "${DIRECT_TRACE_DELETE_BUDGET:-2}" \
    --moment-groups "${EFFICIENCY_ORDERS_ARRAY[@]}" --top-ks "${BENCHMARK_TOP_KS_ARRAY[@]}" \
    --threads "$THREADS" --normalization "$NORMALIZATION" \
    --output-dir "$OUTPUT_ROOT/efficiency"
)

preserve() {
  local dataset="$1" metric="$2" folder="$3"
  local thread_args=()
  if [[ -n "${PRESERVATION_THREADS:-}" ]]; then
    thread_args=(--threads "$PRESERVATION_THREADS")
  fi
  run "preservation_$folder" -m experiments.EXP2_graph_property_preservation \
    --compare-samplers --datasets "$dataset" --properties "$metric" \
    --moment-groups "${PRESERVATION_ORDERS_ARRAY[@]}" --backend "${PRESERVATION_BACKEND:-low_rank}" \
    --top-ks "${TOP_KS_ARRAY[@]}" --seeds "${SEEDS_ARRAY[@]}" \
    --max-remove-ratio "${MAX_REMOVE_RATIO:-0.70}" --checkpoint-step "${CHECKPOINT_STEP:-0.05}" \
    --candidate-batch-size "${PRESERVATION_CANDIDATE_BATCH_SIZE:-2048}" \
    --normalization "$NORMALIZATION" "${thread_args[@]}" \
    --output-dir "$OUTPUT_ROOT/preservation/$folder"
}

if [[ "$MODE" == smoke ]]; then
  DATASET=Texas DELETE_BUDGET=3 efficiency
  run preservation_smoke -m experiments.EXP2_graph_property_preservation \
    --compare-samplers --datasets Texas --moment-groups 2,3 2,3,4 \
    --backend low_rank --top-ks 5 25 50 --seeds 0 1 \
    --max-remove-ratio 0.02 --checkpoint-step 0.01 --normalization "$NORMALIZATION" \
    --output-dir "$OUTPUT_ROOT/preservation/smoke"
else
  if [[ "$MODE" == efficiency || "$MODE" == all ]]; then efficiency; fi
  if [[ "$MODE" == preservation || "$MODE" == all ]]; then
    preserve ca-GrQc spectrum_rmse ca_grqc
    preserve CiteSeer mean_triangle_weighted_clustering_abs_error citeseer
    preserve Cora mean_neighbor_inverse_degree_abs_error cora
    preserve Actor normalized_estrada_index_abs_error actor
  fi
fi
echo "[done] $OUTPUT_ROOT"
