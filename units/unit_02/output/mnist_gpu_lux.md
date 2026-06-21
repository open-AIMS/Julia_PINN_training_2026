::: {.cell-output .cell-output-stdout}
```text
================================================================
Unit 2.7 — MNIST MLP training on CPU vs GPU (Lux.jl)
================================================================
GPU available: true  (NVIDIA L4)
loading MNIST … train=60000  test=10000  (28x28 → 784)

Per-epoch wall-clock (batch=128, full 60k train set):
device    sec / epoch
------------------------
CPU              2.01
GPU              0.67

GPU speed-up: 3.0x per epoch

Full training run on GPU — 10 epochs:
  epoch  1   test accuracy = 0.9616
  epoch  2   test accuracy = 0.9680
  epoch  3   test accuracy = 0.9768
  epoch  4   test accuracy = 0.9780
  epoch  5   test accuracy = 0.9766
  epoch  6   test accuracy = 0.9784
  epoch  7   test accuracy = 0.9791
  epoch  8   test accuracy = 0.9826
  epoch  9   test accuracy = 0.9801
  epoch 10   test accuracy = 0.9796

Final MNIST test accuracy: 0.9796
```
:::
