#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
python -m experiments.EXP1_edge_direction_density \
  --scale-mode graph_relative \
  --output-dir results/EXP1_edge_direction_density
