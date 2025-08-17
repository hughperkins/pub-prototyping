---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np

def sample_base_x():
    return np.random.randn()

def sample_data_x():
    if n % 3 == 0:
        v = np.random.randint(1000, 1400) / 1000
    else:
        v = np.random.randint(-3000, -2500) / 1000
    return v
```

```python
# Plot base and data distributions
import matplotlib.pyplot as plt

X = np.arange(-5, 5.05, 0.05)
BASE = (1 / np.sqrt(2 * np.pi)) * np.exp(-X**2 / 2)
plt.plot(X, BASE, label='base distribution')


def data_distribution(x):
    cond1 = (1.0 <= x) & (x < 1.4)  # First uniform (1.0 ≤ x < 1.4)
    cond2 = (-3.0 <= x) & (x < -2.5) # Second uniform (-3.0 ≤ x < -2.5)
    return np.where(cond1, 5/6, np.where(cond2, 4/3, 0))

DATA = data_distribution(X)
plt.plot(X, DATA, label='data distribution')

plt.legend()
plt.xlabel('x')
plt.ylabel('probability density')
plt.show()

```

```python
import matplotlib.pyplot as plt
import numpy as np
import gstaichi as ti
import torch

ti.init(arch=ti.cpu)

M = 300
N = 300
# num_timesteps = 10
num_particles = 32

X_MIN = -5
X_MAX = 5

particles_base = ti.ndarray(ti.f32, (num_particles,))
particles_data = ti.ndarray(ti.f32, (num_particles,))

@ti.kernel
def fill(a: ti.types.NDArray[ti.math.vec2, 2]) -> None:
    for i, j in ti.ndrange(M, N):
        a[i, j] = ti.math.vec2(float(i) / M, float(j) / N)

@ti.kernel
def populate_base(particles: ti.types.NDArray[ti.f32, 1]) -> None:
    for i in ti.ndrange(num_particles):
        particles[i] = ti.randn()

@ti.kernel
def move_particles(
       a: ti.types.NDArray[ti.math.vec2, 2],
    particles_base: ti.types.NDArray[ti.f32, 1],
    particles_data: ti.types.NDArray[ti.f32, 1]
) -> None:
    for i in ti.ndrange(num_particles):
        for j in range(M):
            t = float(j) / M
            pos = particles_base[i] * (1 - t) + t * particles_data[i]
            pos_axes = (pos - X_MIN) / (X_MAX - X_MIN)
            i_idx = int(pos_axes * N)
            a[i_idx, j] = ti.math.vec2(1.0, 1.0)

a = ti.ndarray(ti.math.vec2, (M, N))
populate_base(particles_base)
print('particles_base', particles_base.to_numpy())
print('a[50, 50]', a[50, 50])

data_l = []
while len(data_l) < num_particles:
    n = len(data_l)
    v = sample_data_x()
    data_l.append(v)
        
particles_data.from_numpy(np.array(data_l, dtype=np.float32))
print('particles_data', particles_data.to_numpy())

move_particles(a, particles_base, particles_data)

# plt.imshow(a.to_numpy()[:, :, 1].transpose(), cmap='gray')

particles_base_np = particles_base.to_numpy()
particles_data_np = particles_data.to_numpy()

for i in range(num_particles):
    plt.plot([0, 1], [particles_base_np[i], particles_data_np[i]], color='grey')
plt.xlabel('tau')
plt.ylabel('x')
plt.show()
```

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, I, H, O):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(I, H))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(H, H))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(H, O))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.mlp(x) * 5
        return x

```

```python
from torch import optim
import random

particles_data_np = particles_data.to_numpy()

def sample_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs_l, targets_l = [], []
    for b in range(batch_size):
        tau = np.random.rand()
        x0 = np.random.randn()
        # print('x0', x0)
        x1 = np.random.choice(particles_data_np)
        x = (1 - tau) * x0 + tau * x1
        dx_dtau = x1 - x0
        inputs_l.append((1, tau, x))
        targets_l.append(dx_dtau)
    inputs = torch.Tensor(inputs_l)
    targets = torch.Tensor(targets_l)
    return inputs, targets

batch_size = 32
hidden_size = 16

mlp = MLP(3, hidden_size, 1)  # inputs: bias, tau, x0; outputs: 
mse_loss = nn.MSELoss()

opt = optim.Adam(mlp.parameters(), lr=0.001)
mlp.train()
losses_l = []
for it in range(10000):
    inputs, targets = sample_batch(batch_size)
    outputs = mlp(inputs)
    opt.zero_grad()
    targets = targets.unsqueeze(-1)
    loss = mse_loss(outputs, targets)
    if loss.item() > 100:
        print(inputs)
        print(targets)
        asdfad
    losses_l.append(loss.item())
    loss.backward()
    opt.step()

plt.plot(losses_l)
plt.show()

mlp.eval()
```

```python
mlp.eval()
dtau = 0.02

num_lines = 10
for x0 in np.arange(-5, 5, 10 / num_lines):
    x_l = []
    tau_l = []
    x = x0
    for tau in np.arange(0, 1 + dtau, dtau):
        inputs = torch.Tensor([[1, tau, x]])
        with torch.no_grad():
            v = mlp(inputs)
        x += dtau * v
        x_l.append(x.item())
        tau_l.append(tau)
    plt.plot(tau_l, x_l)
plt.xlabel('tau')
plt.ylabel('x')
plt.show()
```

```python
mlp.eval()
dtau = 0.01

num_lines = 20
for _ in range(num_lines):
    x = np.random.randn()
    x_l = []
    tau_l = []
    for tau in np.arange(0, 1 + dtau, dtau):
        inputs = torch.Tensor([[1, tau, x]])
        with torch.no_grad():
            v = mlp(inputs)
        x += dtau * v
        x_l.append(x.item())
        tau_l.append(tau)
    plt.plot(tau_l, x_l)
plt.xlabel('tau')
plt.ylabel('x')
plt.show()
```

```python
for tau in np.arange(0, 1.2, 0.2):
    x_l = []
    v_l = []
    for x in np.arange(-5, 5, 0.1):
        with torch.no_grad():
            v = mlp(torch.Tensor([[1, tau, x]]))
        x_l.append(x)
        v_l.append(v[0].item())
    
    plt.plot(x_l, v_l, label=f"tau={tau}")
example_inputs, example_targets = sample_batch(16)
plt.scatter(example_inputs[:, 2], example_targets, label='sample_targets')
with torch.no_grad():
    example_outputs = mlp(torch.Tensor(example_inputs))
plt.scatter(example_inputs[:, 2], example_outputs, label='sample_outputs')
plt.legend()
plt.show()
```
