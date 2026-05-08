import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from model import TransformerLanguageModel

# -------------------------
# LOAD DATA
# -------------------------

text_path = Path("processed/dialogue.txt")

with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# TOKENIZER (char-level)
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
# TRAIN / VAL SPLIT
# -------------------------

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# -------------------------
# HYPERPARAMETERS
# -------------------------

block_size = 64
batch_size = 32
max_iters = 3000
eval_interval = 300

# -------------------------
# BATCHING
# -------------------------

def get_batch(split):

    data_source = train_data if split == "train" else val_data

    ix = torch.randint(len(data_source) - block_size, (batch_size,))

    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])

    return x, y

# -------------------------
# MODEL
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
# LOGGING SETUP
# -------------------------

log_file = Path("output.log")

with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"=== TRAINING STARTED: {datetime.now()} ===\n\n")


def log(text):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# -------------------------
# TRAINING LOOP
# -------------------------

loss_history = []

for iter in range(max_iters):

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    if iter % eval_interval == 0:

        msg = f"STEP {iter} | LOSS {loss.item():.4f}"

        print(msg)

        log(msg)
        log("----")

        loss_history.append(loss.item())

        # -------------------------
        # GENERATION SAMPLE
        # -------------------------

        context = torch.zeros((1, 1), dtype=torch.long)

        generated = model.generate(context, max_new_tokens=200)

        sample = decode(generated[0].tolist())

        log("SAMPLE:")
        log(sample)
        log("====\n")

        # -------------------------
        # CHECKPOINT SAVE
        # -------------------------

        ckpt_path = f"model_step_{iter}.pt"

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "iter": iter
        }, ckpt_path)

        log(f"CHECKPOINT SAVED: {ckpt_path}")
        log("====\n")

# -------------------------
# FINAL GENERATION
# -------------------------

context = torch.zeros((1, 1), dtype=torch.long)

generated = model.generate(context, max_new_tokens=500)

final_text = decode(generated[0].tolist())

print("\n--- FINAL GENERATED TEXT ---\n")
print(final_text)

log("FINAL SAMPLE:")
log(final_text)
log("====")

# -------------------------
# LOSS VISUALIZATION
# -------------------------

plt.plot(
    [i * eval_interval for i in range(len(loss_history))],
    loss_history
)

plt.title("Training Loss Curve")
plt.xlabel("Steps")
plt.ylabel("Loss")

plt.show()