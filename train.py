import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import tokenizer as tok
from model import TransformerLanguageModel

# -------------------------
# DEVICE SETUP
# -------------------------

print("CUDA available:", torch.cuda.is_available())
print("MPS available (Mac GPU):", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# optional speed tweak
try:
    torch.set_float32_matmul_precision("high")
except:
    pass

# -------------------------
# TOKENIZER
# -------------------------

tokenizer = tok.build_tokenizer(target_vocab_size=2000)

# -------------------------
# LOAD DATA
# -------------------------

text_path = Path("processed/dialogue.txt")

with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

# -------------------------
# ENCODE DATA (KEEP ON CPU)
# -------------------------

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# -------------------------
# HYPERPARAMETERS
# -------------------------

block_size = 128
batch_size = 32 if device.type != "cpu" else 16

max_iters = 3000
eval_interval = 300

embed_size = 256
num_layers = 4
num_heads = 8

temperature = 0.8
top_k = 50

# -------------------------
# BATCHING (FIXED)
# -------------------------

def get_batch(split):
    data_source = train_data if split == "train" else val_data

    ix = torch.randint(len(data_source) - block_size, (batch_size,))

    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

# -------------------------
# MODEL
# -------------------------

model = TransformerLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embed_size=embed_size,
    block_size=block_size,
    num_layers=num_layers,
    num_heads=num_heads
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# -------------------------
# LOGGING
# -------------------------

log_file = Path("output.log")

with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"=== TRAINING STARTED: {datetime.now()} ===\n\n")

def log(msg):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# -------------------------
# TRAINING LOOP
# -------------------------

loss_history = []

for step in range(max_iters):

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    # -------------------------
    # EVAL + LOGGING
    # -------------------------

    if step % eval_interval == 0:

        msg = f"STEP {step} | LOSS {loss.item():.4f}"

        print(msg)
        log(msg)
        log("----")

        loss_history.append(loss.item())

        # -------------------------
        # GENERATION SAMPLE
        # -------------------------

        context = torch.zeros((1, 1), dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(
                context,
                max_new_tokens=200,
                temperature=temperature,
                top_k=top_k,
            )

        sample = tokenizer.decode(generated[0].tolist())

        log("SAMPLE:")
        log(sample)
        log("====\n")

        # -------------------------
        # CHECKPOINT
        # -------------------------

        ckpt_path = f"model_step_{step}.pt"

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step
        }, ckpt_path)

        log(f"CHECKPOINT SAVED: {ckpt_path}")
        log("====\n")

# -------------------------
# FINAL GENERATION
# -------------------------

context = torch.zeros((1, 1), dtype=torch.long, device=device)

with torch.no_grad():
    generated = model.generate(
        context,
        max_new_tokens=500,
        temperature=temperature,
        top_k=top_k,
    )

final_text = tokenizer.decode(generated[0].tolist())

print("\n--- FINAL GENERATED TEXT ---\n")
print(final_text)

log("FINAL SAMPLE:")
log(final_text)
log("====")

# # -------------------------
# # LOSS CURVE
# # -------------------------

# plt.plot(
#     [i * eval_interval for i in range(len(loss_history))],
#     loss_history
# )

# plt.title("Training Loss Curve")
# plt.xlabel("Steps")
# plt.ylabel("Loss")

# plt.show()