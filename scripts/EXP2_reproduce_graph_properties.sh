#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CONDA_ENV="${CONDA_ENV:-CGC}"

RESULT_ROOT="results/EXP2_graph_property_preservation"
LOG_DIR="${RESULT_ROOT}/logs"
RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

launch() {
  local run_name="$1"
  shift
  local log_file="${LOG_DIR}/${run_name}_${RUN_TIMESTAMP}.log"

  nohup conda run --no-capture-output -n "${CONDA_ENV}" "$@" \
    >"${log_file}" 2>&1 < /dev/null &
  echo "[started] ${run_name} pid=$! log=${ROOT}/${log_file}"
}

launch wisconsin_spectrum \
  python -m experiments.EXP2_graph_property_preservation \
  --datasets wisconsin \
  --methods anchor_m2 anchor_m2_m3 anchor_m2_m3_m4 random_edge \
  --properties spectrum_rmse \
  --seeds 0 \
  --normalization graph_relative \
  --max-remove-ratio 0.70 \
  --checkpoint-step 0.05 \
  --output-dir "${RESULT_ROOT}/wisconsin/spectrum"

launch wisconsin_properties \
  python -m experiments.EXP2_graph_property_preservation \
  --datasets wisconsin \
  --methods anchor_m2_m3_m4 random_edge \
  --properties mean_neighbor_inverse_degree_abs_error mean_triangle_weighted_clustering_abs_error normalized_estrada_index_abs_error \
  --seeds 0 \
  --normalization graph_relative \
  --max-remove-ratio 0.70 \
  --checkpoint-step 0.05 \
  --output-dir "${RESULT_ROOT}/wisconsin/properties"

# launch actor_spectrum \
#   python -m experiments.EXP2_graph_property_preservation \
#   --datasets actor \
#   --methods anchor_m2 anchor_m2_m3 anchor_m2_m3_m4 random_edge \
#   --properties spectrum_rmse \
#   --seeds 0 \
#   --normalization graph_relative \
#   --max-remove-ratio 0.70 \
#   --checkpoint-step 0.05 \
#   --output-dir "${RESULT_ROOT}/actor/spectrum"

# launch actor_properties \
#   python -m experiments.EXP2_graph_property_preservation \
#   --datasets actor \
#   --methods anchor_m2_m3_m4 random_edge \
#   --properties mean_neighbor_inverse_degree_abs_error mean_triangle_weighted_clustering_abs_error normalized_estrada_index_abs_error \
#   --seeds 0 \
#   --normalization graph_relative \
#   --max-remove-ratio 0.70 \
#   --checkpoint-step 0.05 \
#   --output-dir "${RESULT_ROOT}/actor/properties"


# launch cornell_spectrum \
#   python -m experiments.EXP2_graph_property_preservation \
#   --datasets cornell \
#   --methods anchor_m2 anchor_m2_m3 anchor_m2_m3_m4 random_edge \
#   --properties spectrum_rmse \
#   --seeds 0 \
#   --normalization graph_relative \
#   --max-remove-ratio 0.70 \
#   --checkpoint-step 0.05 \
#   --output-dir "${RESULT_ROOT}/cornell/spectrum"

# launch cornell_properties \
#   python -m experiments.EXP2_graph_property_preservation \
#   --datasets cornell \
#   --methods anchor_m2_m3_m4 random_edge \
#   --properties mean_neighbor_inverse_degree_abs_error mean_triangle_weighted_clustering_abs_error normalized_estrada_index_abs_error \
#   --seeds 0 \
#   --normalization graph_relative \
#   --max-remove-ratio 0.70 \
#   --checkpoint-step 0.05 \
#   --output-dir "${RESULT_ROOT}/cornell/properties"

# launch ca_grqc_properties \
#   python -m experiments.EXP2_graph_property_preservation \
#   --datasets ca-GrQc \
#   --methods anchor_m2_m3_m4 random_edge \
#   --properties mean_neighbor_inverse_degree_abs_error mean_triangle_weighted_clustering_abs_error normalized_estrada_index_abs_error \
#   --seeds 0 \
#   --normalization graph_relative \
#   --candidate-batch-size 2048 \
#   --max-remove-ratio 0.70 \
#   --checkpoint-step 0.05 \
#   --output-dir "${RESULT_ROOT}/ca_grqc/properties"

# launch ca_grqc_spectrum \
#   python -m experiments.EXP2_graph_property_preservation \
#   --datasets ca-GrQc \
#   --methods anchor_m2 anchor_m2_m3 anchor_m2_m3_m4 random_edge \
#   --properties spectrum_rmse \
#   --seeds 0 \
#   --normalization graph_relative \
#   --candidate-batch-size 2048 \
#   --max-remove-ratio 0.70 \
#   --checkpoint-step 0.05 \
#   --output-dir "${RESULT_ROOT}/ca_grqc/spectrum"

# launch cora_properties \
#   python -m experiments.EXP2_graph_property_preservation \
#   --datasets Cora \
#   --methods anchor_m2_m3_m4 random_edge \
#   --properties mean_neighbor_inverse_degree_abs_error mean_triangle_weighted_clustering_abs_error normalized_estrada_index_abs_error \
#   --seeds 0 \
#   --normalization graph_relative \
#   --max-remove-ratio 0.70 \
#   --checkpoint-step 0.05 \
#   --output-dir "${RESULT_ROOT}/cora/properties"

# launch cora_spectrum \
#   python -m experiments.EXP2_graph_property_preservation \
#   --datasets Cora \
#   --methods anchor_m2 anchor_m2_m3 anchor_m2_m3_m4 random_edge \
#   --properties spectrum_rmse \
#   --seeds 0 \
#   --normalization graph_relative \
#   --max-remove-ratio 0.70 \
#   --checkpoint-step 0.05 \
#   --output-dir "${RESULT_ROOT}/cora/spectrum"

# launch citeseer_properties \
#   python -m experiments.EXP2_graph_property_preservation \
#   --datasets CiteSeer \
#   --methods anchor_m2_m3_m4 random_edge \
#   --properties mean_neighbor_inverse_degree_abs_error mean_triangle_weighted_clustering_abs_error normalized_estrada_index_abs_error \
#   --seeds 0 \
#   --normalization graph_relative \
#   --max-remove-ratio 0.70 \
#   --checkpoint-step 0.05 \
#   --output-dir "${RESULT_ROOT}/citeseer/properties"

# launch citeseer_spectrum \
#   python -m experiments.EXP2_graph_property_preservation \
#   --datasets CiteSeer \
#   --methods anchor_m2 anchor_m2_m3 anchor_m2_m3_m4 random_edge \
#   --properties spectrum_rmse \
#   --seeds 0 \
#   --normalization graph_relative \
#   --max-remove-ratio 0.70 \
#   --checkpoint-step 0.05 \
#   --output-dir "${RESULT_ROOT}/citeseer/spectrum"
