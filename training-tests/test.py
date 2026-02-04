import torch
import torch.nn as nn

torch.manual_seed(0)

# Simple 1-hidden-layer network
model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 1))

x = torch.randn(4, 3)  # batch of 4 samples, 3 features each
y = torch.randn(4, 1)  # target

loss_fn = nn.MSELoss()

# Forward pass
y_hat = model(x)
loss = loss_fn(y_hat, y)

# Backprop (computes gradients)
loss.backward()

# Inspect gradients
for name, p in model.named_parameters():
    print(name, "grad shape:", p.grad.shape, "grad norm:", p.grad.norm().item())
