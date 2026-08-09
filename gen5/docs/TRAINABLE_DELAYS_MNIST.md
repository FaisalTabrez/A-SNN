# Phase 30: Trainable Delay Assignment

Phase 29 produced the strongest conventional-task temporal-mechanism result in
Gen-5 so far: heterogeneous recurrent delays improved every seed by about
`+8.09` linear and `+7.59` MLP accuracy points over paired no-delay LIF.
Uniform delay did not transfer to the MLP, showing that delay diversity—not
generic slowing—was responsible.

Phase 30 is the final row-sequential MNIST diagnostic. It asks whether delay
assignment can be optimized while preserving the same 272-edge graph.

## Arms

- `lif_no_delay_warm_all`: no-delay reference;
- `fixed_distance012_warm_all`: winning Phase 29 fixed-delay control;
- `learned_soft_distance_init`: soft delay mixtures initialized near the fixed
  winner;
- `learned_st_distance_init`: hard forward delays with straight-through
  gradients;
- `learned_soft_flat_init`: neutral one-third probability per recurrent delay;
- `raw`: flattened-pixel ceiling.

Each recurrent edge receives three delay logits for buckets 0, 1, and 2.
Sensor edges remain immediate and their delay logits are gradient-masked. The
readout trains for ten warmup epochs before LTWs and delay logits activate.

## Colab CUDA run

```python
%cd /content/A-SNN
!git pull

!python gen5/examples/sprint30_trainable_delays_mnist.py \
  --device cuda \
  --seeds 42 43 44 \
  --train-samples 20000 \
  --test-samples 5000 \
  --epochs 15 \
  --warmup-epochs 10 \
  --delay-learning-rate 0.003 \
  --entropy-regularization 0.001 \
  --batch-size 512 \
  --data-root /content/drive/MyDrive/A-SNN/gen5_data \
  --output-dir /content/drive/MyDrive/A-SNN/gen5_outputs/trainable_delays_mnist_cuda
```

## Decision gate

Learned assignment passes only with at least `+0.5` points over the paired
fixed-distance control, two improved seeds, stable event activity, and no LTW
saturation. Regardless of the result, this ends MNIST mechanism tuning. Phase
31 carries either the learned winner or the fixed Phase 29 winner to SHD.
