"""
MLP on MNIST in PyTorch.

Two hidden layers (256, 128), ReLU, Adam, mini-batch SGD. The
explicit-training-loop idiom from the deep learning book.

Requires:  pip install torch torchvision
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
tf  = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
tr  = datasets.MNIST("~/torch-data", train=True,  download=True, transform=tf)
te  = datasets.MNIST("~/torch-data", train=False, download=True, transform=tf)
tl  = DataLoader(tr, batch_size=128, shuffle=True)
vl  = DataLoader(te, batch_size=512)

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
).to(device)
opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for X, y in tl:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        loss(model(X), y).backward()
        opt.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in vl:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    print(f"epoch {epoch:2d}  test acc = {correct / total:.4f}")
