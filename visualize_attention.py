import torch
import matplotlib.pyplot as plt
from model import TransformerLanguageModel

# -------------------------
# LOAD MODEL (assumes you saved checkpoint)
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
# SAMPLE INPUT
# -------------------------

text = "Ayanokoji is always"

chars = sorted(list(set(text)))  # placeholder tokenizer assumption

# NOTE: ideally reuse your tokenizer here

# fake demo tensor (replace with real encode())
idx = torch.randint(0, 50, (1, 10))

# -------------------------
# CAPTURE ATTENTION
# -------------------------

att_maps = []

def hook(module, input, output):
    att_maps.append(output.detach())

# hook first attention layer
for block in model.blocks:
    block.sa.register_forward_hook(hook)

with torch.no_grad():
    model(idx)

# -------------------------
# VISUALIZE FIRST HEAD OUTPUT
# -------------------------

att = att_maps[0][0]  # (T, T)

plt.imshow(att, cmap="viridis")
plt.title("Self-Attention Heatmap")
plt.xlabel("Key tokens")
plt.ylabel("Query tokens")

plt.colorbar()
plt.show()