# Phase 18: frozen event-coded MNIST

Phase 18 is the first AMMC benchmark outside the embodied foraging worlds. It
asks a narrow question: does a frozen sparse spiking reservoir create a useful
representation of event-coded handwritten digits?

This is not an end-to-end AMMC learning result. The sparse reservoir topology
and every STW/LTW value are frozen. Only the small classifier heads are trained.

## Experimental design

MNIST images are resized from `28x28` to `8x8`. Each of the 64 normalized
pixels emits at most one latency-coded event across eight timesteps: brighter
pixels fire earlier and near-black pixels remain silent.

The default reservoir contains:

- 64 sensor neurons;
- 64 recurrent hidden neurons;
- 128 sensor-to-hidden edges;
- 256 hidden recurrent edges;
- 384 active edges in a fixed pool of 512 slots;
- frozen LTWs and no STW updates, sprouting, pruning, or backpropagation into
  the reservoir.

Four models are trained and evaluated on identical official MNIST splits:

| Model | Purpose |
|---|---|
| `raw_pixel_linear` | Linear baseline on the 64 downsampled intensities. |
| `raw_pixel_mlp` | Strong dense baseline width-matched to the AMMC MLP's trainable parameter budget. |
| `frozen_ammc_linear` | Linear probe on frozen spike-count and membrane traces. |
| `frozen_ammc_mlp` | Nonlinear readout on the same frozen traces. |

The official test split is never used to fit the readouts. The default run uses
three independent reservoir/readout seeds and reports mean plus population
standard deviation. It also records hidden-neuron spike rate; an inactive
reservoir must not be interpreted as useful temporal computation.

## Google Colab cells

Use a GPU runtime. Mount Drive and update the existing checkout:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/A-SNN
!git pull
```

Check that PyTorch and torchvision are ABI-compatible:

```python
import torch, torchvision
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())
```

Run the default evidence configuration:

```python
!python gen5/examples/sprint18_event_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --image-size 8 \
  --timesteps 8 \
  --hidden-neurons 64 \
  --max-edges 512 \
  --epochs 15 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/event_mnist_cuda
```

Nothing must be uploaded for the first run: torchvision downloads MNIST into
`gen5_data`. The generated evidence files are:

- `event_mnist.json`
- `event_mnist_records.csv`
- `event_mnist_summary.csv`
- `event_mnist_summary.png`

Download those four files, or zip the directory, after the run.

## Decision rules

- `frozen_ammc_linear > raw_pixel_linear`: the frozen event dynamics improve
  linear separability over the same downsampled input.
- `frozen_ammc_mlp > raw_pixel_mlp`: evidence that the sparse reservoir adds
  value beyond a parameter-budget-matched dense nonlinear head.
- `frozen_ammc_mlp > frozen_ammc_linear`: the reservoir contains useful but
  nonlinearly arranged class information.
- Frozen models below raw baselines: the current event code/topology does not
  justify added complexity; tune coding and reservoir dynamics before scaling.

Do not compare only against chance. MNIST is easy for conventional models, so
the raw-pixel MLP is the minimum credible performance baseline.

## First CUDA result

The first three-seed run failed both representation decision rules:

| Model | Mean test accuracy |
|---|---:|
| Raw pixel linear | 85.94% |
| Frozen AMMC linear | 79.31% |
| Raw pixel MLP | 95.14% |
| Frozen AMMC MLP | 86.11% |

The frozen reservoir was active at a mean hidden spike rate of `2.37%`, but it
lost `6.63` percentage points against raw linear features and `9.03` points
against the parameter-matched raw MLP. The next phase decomposes raw, event,
sensor, and hidden representations before any topology scaling or plasticity.

Full analysis:
`gen5/outputs/event_mnist_cuda_2026-08-09/analysis.md`.
