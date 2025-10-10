import torch
import matplotlib.pyplot as plt

# assume we know dt and tau
dt = 0.1
tau = 1.0

# simulate z for some number of timesteps
timesteps = 100 

# pick x_target(0) and W_target (so we know what the right answer is)
x0_target = torch.tensor(6)
W_target = torch.tensor(7)

# simulate x for 100 timesteps
x_target = [x0_target]
for t in range(timesteps - 1):
    x_next = x_target[-1] * (1 - ((dt * W_target) / tau))
    x_target.append(x_next)

# converts to tensor
x_target = torch.stack(x_target)

# random initial values for x(0) and W
x0 = torch.tensor(1.0, requires_grad=True)
W = torch.tensor(0.5, requires_grad=True) 

optimizer = torch.optim.SGD([x0, W], lr=0.05)
losses = []

# training
for epoch in range(500):
    optimizer.zero_grad()

    x_pred = [x0]

    for t in range(timesteps - 1):
        x_next = x_pred[-1] * (1 - ((dt * W) / tau))
        x_pred.append(x_next)
    
    x_pred = torch.stack(x_pred)

    loss = torch.sum((x_pred - x_target) ** 2)

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 25 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}, x0: {x0.item()}, W: {W.item()}")