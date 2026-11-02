#!/bin/bash
# STEP 1: process every field of view (FOV), one PBS array element each.
# Run on the cluster login node from the repository root, with the smfish conda env active:
#
#   bash pbs/1_process_fovs.sh config/250806/experiment.yaml           # all FOVs (finished ones are skipped)
#   bash pbs/1_process_fovs.sh config/250806/experiment.yaml pending   # only FOVs not done yet (missing / failed)
#   OVERWRITE=1 bash pbs/1_process_fovs.sh <config>                    # re-process everything
#
# Each FOV job (pbs/run_fov.pbs): load -> segment cells and nuclei (Cellpose-SAM) -> detect spots (LoG, stored
# down to a permissive floor) -> measure intensities -> assign spots to cells / compartments. It writes
# <output_dir>/fov/<fov_id>/{spots.parquet, cells.parquet, labels.tif, qc.json, overlay.png, DONE.json}.
# Check progress with `python -m smfish.cli check -c <config>`; when every FOV is done, go to step 2.
#
# Settings (environment variables): CONDA_ENV (default smfish), CELLPOSE_LOCAL_MODELS_PATH (where
# `python -m smfish.cli check-env --download` put the Cellpose-SAM weights; compute nodes cannot download).
# The manifest (<output_dir>/manifest.csv, the list of FOVs) is built on first use. Each submission snapshots it
# next to its index list, so rebuilding the manifest while jobs are queued cannot make them process the wrong
# FOVs. PBS -J only takes a range, so array element k processes the manifest index on line k+1 of the list.
set -euo pipefail

CONFIG=${1:?usage: bash pbs/1_process_fovs.sh <config.yaml> [all|pending]}
MODE=${2:-all}
cd "$(dirname "$0")/.."
mkdir -p logs

OUT=$(python -c "from smfish.config import load_config; print(load_config('$CONFIG')['paths']['output_dir'])")
MANIFEST="$OUT/manifest.csv"
[ -f "$MANIFEST" ] || python -m smfish.cli manifest -c "$CONFIG"
if [ -n "${CELLPOSE_LOCAL_MODELS_PATH:-}" ] && [ ! -f "$CELLPOSE_LOCAL_MODELS_PATH/cpsam_v2" ]; then
  echo "No Cellpose-SAM weights in $CELLPOSE_LOCAL_MODELS_PATH: run python -m smfish.cli check-env --download" >&2
  exit 1
fi

STAMP="$OUT/submit_$(date +%Y%m%d_%H%M%S)_${MODE}"
INDEX_FILE="$STAMP.txt"
SNAPSHOT="$STAMP.manifest.csv"
cp "$MANIFEST" "$SNAPSHOT"
case "$MODE" in
  all) tail -n +2 "$SNAPSHOT" | cut -d, -f1 > "$INDEX_FILE" ;;
  pending) python -m smfish.cli check -c "$CONFIG" --manifest "$SNAPSHOT" --write-pending "$INDEX_FILE" > /dev/null || true ;;
  *) echo "mode must be all or pending" >&2; exit 2 ;;
esac

N=$(grep -c . "$INDEX_FILE" || true)
if [ "$N" -eq 0 ]; then
  echo "Nothing to submit."
  exit 0
fi
VARS="CONFIG=$CONFIG,INDEX_FILE=$INDEX_FILE,MANIFEST=$SNAPSHOT,CONDA_ENV=${CONDA_ENV:-smfish}"
VARS="$VARS${CELLPOSE_LOCAL_MODELS_PATH:+,CELLPOSE_LOCAL_MODELS_PATH=$CELLPOSE_LOCAL_MODELS_PATH}${OVERWRITE:+,OVERWRITE=1}"
if [ "$N" -eq 1 ]; then
  qsub -o logs/ -v "$VARS" pbs/run_fov.pbs
else
  qsub -o logs/ -J "0-$((N - 1))" -v "$VARS" pbs/run_fov.pbs
fi
echo "Submitted $N FOV job(s); index list: $INDEX_FILE"
