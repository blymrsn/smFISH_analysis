# smFISH: mRNA per cell and size scaling from 3D smFISH images

This code counts single mRNA molecules (smFISH spots) in 3D confocal stacks of HCT116 cells, assigns each spot to
a cell and its nucleus or cytoplasm, and measures how mRNA number scales with cell size.

- **Probes and channels** are read from the plate layout, so the code never assumes which probe is in which channel.
- **Spot thresholds** are calibrated on negative-control probes imaged on the same plate. They are set after
  detection: every candidate spot is stored with its score.
- **Uncertainty** comes from resampling whole fields of view.

Two plates with the same layout are included: **250806** and its replicate **250929**.

## The workflow

| Step | Where | What | Command | Output |
|---|---|---|---|---|
| 0 | cluster, once | install, download the Cellpose-SAM weights | see [Install](#0-install) | conda env `smfish` |
| 1 | cluster (PBS, GPU) | per field of view: segment cells and nuclei, detect and measure spots, assign them to cells | `bash pbs/1_process_fovs.sh config/250806/experiment.yaml` | `outputs/250806/fov/<field>/` |
| 2 | cluster (PBS) | combine all fields; count spots with the thresholds in the config; calibration reports | `bash pbs/2_aggregate.sh config/250806/experiment.yaml` | `outputs/250806/aggregated/` |
| 3 | laptop | **how each threshold was chosen**, and checks that the result holds | [`notebooks/1_calibration.ipynb`](notebooks/1_calibration.ipynb) | `results/250806/thresholds.csv` and other tables |
| 4 | | only if step 3 proposes different thresholds: paste them into the config, re-run step 2 | | |
| 5 | laptop | **mRNA per cell, size scaling, noise** | [`notebooks/2_analysis.ipynb`](notebooks/2_analysis.ipynb) | `results/250806/*.csv`, figures |

Replace `250806` with `250929` for the replicate (`SMFISH_PLATE=250929` for the notebooks). The notebooks in the
repository have been run on 250806, so their figures can be read on GitHub without running anything.

## Repository layout

```
config/
  probes.yaml                  probe name -> gene, role (exon / intron / neg_control), odds / evens set,
                               cell lines lacking the target (those wells become negative controls)
  250806/experiment.yaml       every setting for one plate: channels, segmentation, detection, thresholds
  250806/plate_layout.csv      well -> probe in each channel, cell line
  250929/...                   the replicate plate
pbs/                           steps 1 and 2 on a PBS cluster
notebooks/                     steps 3 and 5 (.ipynb to read and run; .py is the same notebook as plain text)
smfish/                        the Python package
results/<plate>/               small tables written by the notebooks
tests/                         pytest suite on synthetic images and tables
data/<plate>/OME-TIFF/         raw images (not in the repository)
outputs/<plate>/               pipeline outputs (not in the repository)
```

The configs use paths relative to the repository root. Put (or link) each plate's raw OME-TIFFs in
`data/<plate>/OME-TIFF/`, one file per field of view with `Well<N>` in the name. Outputs go to `outputs/<plate>/`.

## 0. Install

On the cluster:

```bash
conda env create -f environment.yml
conda activate smfish
pip install -e .
export CELLPOSE_LOCAL_MODELS_PATH=<scratch>/cellpose_models   # also in your job environment
python -m smfish.cli check-env --download    # once, on a node with internet: Cellpose-SAM weights (~1.2 GB)
python -m smfish.cli check-env               # on a GPU node: versions, CUDA, a small 3D segmentation test
```

The PBS scripts activate the env with `source ~/.bashrc; conda activate smfish`. If your cluster provides conda
another way, edit those two lines in `pbs/run_fov.pbs` and `pbs/aggregate.pbs`.

For the notebooks alone (no GPU): a Python 3.10 environment and `pip install -e ".[notebooks]"`, then copy or link a
plate's `aggregated/` folder from the cluster to `outputs/<plate>/aggregated/`.

## 1. Process every field of view (PBS)

```bash
bash pbs/1_process_fovs.sh config/250806/experiment.yaml            # submit all fields (an array job)
python -m smfish.cli check -c config/250806/experiment.yaml         # progress, per well
bash pbs/1_process_fovs.sh config/250806/experiment.yaml pending    # resubmit fields that failed or are missing
```

Each field (about 10 min on one GPU) goes through these stages:
1. **Segment**: Cellpose-SAM in 3D. Nuclei come from DAPI; cells from ConA plus DAPI, refined by a watershed on ConA
   ridges. The stack is cropped to the cell layer, without the coverslip plane.
2. **Detect**: a Laplacian-of-Gaussian (LoG) filter, with every local maximum above a low storage floor kept as a
   candidate with its score. Transcription-site probes (introns) use a larger kernel.
3. **Measure**: each spot's intensity minus its local background.
4. **Assign**: each spot to a cell and compartment (nucleus, perinuclear shell, cytoplasm), with its distances to the
   nuclear envelope and the membrane.

Outputs per field: `spots.parquet`, `cells.parquet`, `labels.tif`, `qc.json` (thresholds, elbow curve, segmentation
QC) and `overlay.png`.

## 2. Aggregate (PBS)

```bash
bash pbs/2_aggregate.sh config/250806/experiment.yaml
```

This step combines every field and applies one threshold per probe and channel (`aggregation.thresholds` in the
config). It flags clumps of non-specific staining, and counts spots per cell and compartment. Its outputs, in
`outputs/<plate>/aggregated/`:

| File | Content |
|---|---|
| `spots.parquet` | every candidate spot. `passes` marks a counted spot; `in_clump` marks non-specific staining |
| `cells.parquet` | every cell: volumes, border flags, `qc_pass` |
| `cell_counts_long.parquet` | spots per cell, probe and compartment, including zeros |
| `references.csv` | single-mRNA intensity per well and probe |
| `calibration/` | negative-control curves, proposed thresholds, odds / evens detection, chromatic offsets, clumps per well |

## 3. Calibrate: `notebooks/1_calibration.ipynb`

The notebook shows:
- **the rule**: per probe and channel, the lowest score at which the negative control gives at most 1 false spot per
  cell (at most 0.1 per nucleus for transcription sites), as a median over fields;
- **the controls against real signal**, and the threshold each control gives;
- **clumps of non-specific staining**, and how many spots they remove;
- **detection efficiency**, from the odds / evens wells;
- **robustness**: whether the main readouts move when every threshold is scaled from 0.6x to 1.6x;
- **position in the field**: readouts by distance from the field centre, where illumination is brightest.

It ends by checking that the thresholds in the config are the ones this run's controls give. If they are not, it
prints the block to paste into the config; re-run step 2 afterwards (no image processing, minutes).

## 5. Analyse: `notebooks/2_analysis.ipynb`

1. QC cells per well.
2. mRNA per cell for every probe, with the negative controls.
3. Size scaling of POLR2A mRNA in the knock-in and the parental line: log-log slopes against cell volume, nucleus
   and cytoplasm separately and against their own volumes.
4. Expression noise (Fano factor, excess CV²) by cell-volume tercile.

## A new plate

1. Copy `config/250806/` to `config/<plate>/`. Then edit `plate_layout.csv`, and in `experiment.yaml` edit the
   `paths`, the `channels` (index in the OME-TIFF -> name) and the `conditions`.
2. Add any new probe to `config/probes.yaml`.
3. Run steps 1-2. Set the thresholds from notebook 1 (step 3), re-run step 2, then run notebook 2.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Licence

MIT: see [LICENSE](LICENSE).
