import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import matplotlib.pyplot as plt

from model import TransformerLanguageModel

# -------------------------
# LOAD DATA
# -------------------------

text_path = Path("processed/dialogue.txt")

with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# TOKENIZER
# -------------------------

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(tokens):
    return "".join([itos[t] for t in tokens])

data = torch.tensor(encode(text), dtype=torch.long)

# -------------------------
# SPLIT
# -------------------------

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# -------------------------
# BATCHING
# -------------------------

block_size = 64
batch_size = 32

def get_batch(split):

    data_source = train_data if split == "train" else val_data

    ix = torch.randint(len(data_source) - block_size, (batch_size,))

    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])

    return x, y

# -------------------------
# MODEL (Transformer)
# -------------------------

model = TransformerLanguageModel(
    vocab_size=vocab_size,
    embed_size=128,
    block_size=block_size,
    num_layers=2,
    num_heads=4
)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# -------------------------
# TRAINING
# -------------------------

max_iters = 3000
eval_interval = 300

loss_history = []

for iter in range(max_iters):

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    if iter % eval_interval == 0:

        print(f"step {iter}: loss {loss.item():.4f}")

        loss_history.append(loss.item())

# -------------------------
# GENERATION
# -------------------------

context = torch.zeros((1, 1), dtype=torch.long)

generated = model.generate(context, max_new_tokens=500)

print("\n--- GENERATED TEXT ---\n")
print(decode(generated[0].tolist()))

# -------------------------
# VISUALIZATION (LOSS CURVE)
# -------------------------

plt.figure()

plt.plot(
    [i * eval_interval for i in range(len(loss_history))],
    loss_history
)

plt.title("Training Loss Curve")
plt.xlabel("Steps")
plt.ylabel("Loss")

plt.show()