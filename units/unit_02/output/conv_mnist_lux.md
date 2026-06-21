::: {.cell-output .cell-output-stdout}
```text
================================================================
Unit 2.7 — MNIST convolutional network (LeNet-5, Lux.jl)
================================================================
GPU available: true  (NVIDIA L4)
loading MNIST … train=60000  test=10000  (28×28×1 image tensors)
LeNet-5: 2 conv blocks + 3 dense layers, 61706 parameters

Per-epoch wall-clock (batch=128, full 60k train set):
device    sec / epoch
------------------------
CPU             21.18
GPU              1.33

GPU speed-up: 16.0x per epoch (vs the MLP's ~3x — convolutions are far more compute-dense)

Full training run on GPU — 10 epochs:
  epoch  1   test accuracy = 0.9721
  epoch  2   test accuracy = 0.9789
  epoch  3   test accuracy = 0.9843
  epoch  4   test accuracy = 0.9864
  epoch  5   test accuracy = 0.9870
  epoch  6   test accuracy = 0.9848
  epoch  7   test accuracy = 0.9895
  epoch  8   test accuracy = 0.9861
  epoch  9   test accuracy = 0.9886
  epoch 10   test accuracy = 0.9875

Final MNIST test accuracy: 0.9875  (forest ~0.97, softmax ~0.92, MLP ~0.98)
```
:::
