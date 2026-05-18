"""
Toy KAN example in pykan, the reference implementation
from Liu et al. (2024).

Fit f(x1, x2) = exp(sin(pi x1) + x2^2) with a small 2-layer KAN.

Requires:  pip install pykan
"""

import torch
from kan import KAN, create_dataset

f = lambda x: torch.exp(torch.sin(torch.pi * x[:, [0]]) + x[:, [1]] ** 2)
dataset = create_dataset(f, n_var=2, train_num=1000, test_num=1000)

model = KAN(width=[2, 5, 1], grid=5, k=3, seed=0)
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.0)

mse = torch.mean(
    (model(dataset["test_input"]) - dataset["test_label"]) ** 2
).item()
print(f"test MSE = {mse:.3e}")
