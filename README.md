# NullBlend

Self-supervised seismic deblending with hierarchical multiscale feature
extraction, physically aligned feature aggregation, and hard data consistency.

## The deblending problem

![Simultaneous-source acquisition, blending, and pseudo-deblending at a single receiver](figs/fig2.png)

**Blending and pseudo-deblending.** At a single receiver, the blending operator
$A$ combines traces from different shots into one continuous recording.
Applying its transpose $A^{\top}$ produces a pseudo-deblended gather with
interference from overlapping shots. The colored traces follow shots 112,
113, and 114 through the process.

## Method

![NullBlend architecture with range-space and null-space reconstruction and a self-supervised training loop](figs/fig3.png)

**NullBlend workflow.** The deblending network adds a learned null-space
component to the minimum-norm range-space reconstruction. For self-supervised
training, NullBlend shifts the estimated wavefield along the shot axis to form
a target, reblends it, and trains the network to recover the target from the
reblended measurements.

The minimum-norm data-consistent reconstruction initializes the teacher.
The reconstruction network has four independently parameterized stages,
each consisting of a three-level MW-UNet followed by time-aligned feature
aggregation (TAFA). Network width is 32; total trainable parameters are
**1,321,121**.

Eight input channels contain five receiver-context amplitude channels, overlap
order, normalized relative time and normalized shot index. TAFA concatenates
the sample's own feature and the valid-slot aligned mean, then applies a
zero-initialized 64-to-32 pointwise projection and a residual update.
Acquisition timing defines alignment. The output residual is gated to overlap
regions, added to the minimum-norm reconstruction and projected onto the
observations.

Training uses shifted teacher targets and reblended virtual observations.
The shot shifts cycle through `+1,-1,+2,-2,+4,-4,+8,-8`, one shift per epoch.
The loss is collision-frequency-weighted L1. Every eight epochs the teacher
is directly replaced by the current reconstruction. There is no temporal
averaging or pretrained model. The final reconstruction is the last teacher;
the last student reconstruction is saved separately.

Defaults: 200 epochs, seed 20260818, receiver batch size one, receiver radius
two, backbone/input learning rate 2e-4, TAFA/head rates 5e-6. The initialization
uses a fixed random-stream budget per aggregation layer, preserving the seeded
initialization used by the bundled runs. These temporary initialization draws
are not learned parameters or input features.

Clean data are stored separately and loaded only after training for evaluation.
They are never used for loss, checkpoint selection or early stopping.

## Installation and training

Python 3.10+ with PyTorch and Matplotlib is required. The bundled runs used
Python 3.14 and PyTorch 2.11.0+cu128 on an NVIDIA RTX 5090.

```bash
python -m pip install -r requirements.txt
python run_nullblend.py --dataset mobil --device cuda:0
python run_nullblend.py --dataset seam --device cuda:0
python run_nullblend.py --dataset marmousi --device cuda:0
```

Run from this folder, or use an absolute script path. Data locations resolve
relative to the package. CUDA requests fail explicitly when unavailable.
`--epochs` is intended for smoke checks; use 200 for the reported configuration.

Each dataset contains `data/<dataset>/observed.pt` (keys `continuous`, `dithers`,
`meta`) and `reference_clean.pt` (key `clean`). Training can also finish without
a reference file, in which case evaluation is skipped.

## Included results

| Dataset | Overall S/N (dB) |
|---|---:|
| Mobil | 26.764458 |
| SEAM | 22.932487 |
| Marmousi | 30.255144 |

`results/<dataset>/` contains the final teacher and student reconstructions,
`hnsln_final.pt`, `checkpoints/latest.pt`, run configuration, 200-epoch history,
training log, evaluation tables and figures. `reference_results.json` records
the configuration and scores. All bundled results use the configuration
described here. Floating-point differences can occur across devices/runtimes.

New runs write to `outputs/<dataset>/nullblend_<timestamp>/` by default and do
not overwrite bundled results. Existing checkpoints contain optimizer state
for inspection; automatic interrupted-training resume is not implemented.

## Inference and verification

```bash
python infer.py --checkpoint results/mobil/hnsln_final.pt --device cuda:0 --output prediction.pt
python verify.py --device cpu
python verify_results.py
```

Inference reconstructs the student's prediction from observations. It accepts
both final weights and latest checkpoints and reads the adjacent configuration.
The saved `deblended_crg_final.pt` is the reported teacher output. No clean
reference is loaded by inference. The package is self-contained and requires
no sibling project folders.
