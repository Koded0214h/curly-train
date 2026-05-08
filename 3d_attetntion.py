import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from model import TransformerLanguageModel


# -------------------------
# LOAD MODEL
# -------------------------

ckpt = torch.load("model_step_2700.pt", map_location="cpu")

model = TransformerLanguageModel(
    vocab_size=ckpt["model_state"]["token_embedding.weight"].shape[0],
    embed_size=128,
    block_size=64,
    num_layers=2,
    num_heads=4
)

model.load_state_dict(ckpt["model_state"])
model.eval()

# -------------------------
# SAMPLE INPUT (replace with real encoding later)
# -------------------------

idx = torch.randint(0, 50, (1, 16))

# -------------------------
# CAPTURE ATTENTION (HOOK)
# -------------------------

att_maps = []

def hook(module, input, output):
    att_maps.append(output.detach())

# attach hook to first block attention
model.blocks[0].sa.register_forward_hook(hook)

with torch.no_grad():
    model(idx)

# -------------------------
# EXTRACT ATTENTION
# -------------------------

att = att_maps[0][0]  # (T, T)

T = att.shape[0]

x, y, z = [], [], []

for i in range(T):
    for j in range(T):
        x.append(i)
        y.append(j)
        z.append(att[i, j].item())

x = np.array(x)
y = np.array(y)
z = np.array(z)

# -------------------------
# NORMALIZE FOR COLOR GRADIENT
# -------------------------

colors = (z - z.min()) / (z.max() - z.min() + 1e-8)

# -------------------------
# 3D PLOT
# -------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    x,
    y,
    z,
    c=colors,
    cmap='viridis',
    s=20
)

ax.set_title("3D Self-Attention Landscape")
ax.set_xlabel("Query Token Position")
ax.set_ylabel("Key Token Position")
ax.set_zlabel("Attention Strength")

plt.colorbar(sc)

plt.show()