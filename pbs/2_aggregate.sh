#!/bin/bash
# STEP 2: combine every FOV into experiment tables and the calibration reports.
#
#   bash pbs/2_aggregate.sh config/250806/experiment.yaml
#
# Writes <output_dir>/aggregated/: spots, cells, cell_counts_long / _wide, tss, tss_per_cell (parquet),
# references.csv, and calibration/ (negative-control curves, proposed thresholds, odds/evens detection,
# chromatic offsets, clumps of non-specific staining, ...). Spots are counted with the thresholds in the config.
# Next: notebooks/1_calibration.ipynb. If it proposes different thresholds, paste them into the config and
# run this step again (cheap: no image processing).
set -euo pipefail

CONFIG=${1:?usage: bash pbs/2_aggregate.sh <config.yaml>}
cd "$(dirname "$0")/.."
mkdir -p logs
qsub -o logs/ -v "CONFIG=$CONFIG,CONDA_ENV=${CONDA_ENV:-smfish}" pbs/aggregate.pbs
