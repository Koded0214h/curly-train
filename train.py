import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

import tokenizer as tok
from model import TransformerLanguageModel

ROOT = Path(__file__).resolve().parent

# ── device ──────────────────────────────────────────────────────────────────

print("CUDA available:", torch.cuda.is_available())
print("MPS available (Mac GPU):", torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ── tokenizer ────────────────────────────────────────────────────────────────

tokenizer = tok.build_tokenizer(target_vocab_size=2000)

# ── data ─────────────────────────────────────────────────────────────────────

text_path = ROOT / "processed" / "dialogue.txt"
with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ── hyperparameters ───────────────────────────────────────────────────────────

block_size   = 128
batch_size   = 32 if device.type != "cpu" else 16
max_iters    = 3000
eval_interval = 300
eval_iters   = 64       # mini-batches averaged for stable val loss

embed_size  = 256
num_layers  = 4
num_heads   = 8
dropout     = 0.1

lr          = 3e-4
max_grad_norm = 1.0

temperature = 0.8
top_k       = 50

# ── batching ──────────────────────────────────────────────────────────────────

def get_batch(split):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i : i + block_size] for i in ix])
    y = torch.stack([src[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ── model ────────────────────────────────────────────────────────────────────

model = TransformerLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embed_size=embed_size,
    block_size=block_size,
    num_layers=num_layers,
    num_heads=num_heads,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# cosine LR decay: warm up for 100 steps, decay to 10% of peak
def lr_lambda(step):
    if step < 100:
        return step / 100
    progress = (step - 100) / max(max_iters - 100, 1)
    return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ── logging ───────────────────────────────────────────────────────────────────

log_file = ROOT / "output.log"
with open(log_file, "w", encoding="utf-8") as f:
    f.write(f"=== TRAINING STARTED: {datetime.now()} ===\n\n")

def log(msg):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ── eval helper ───────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss():
    model.eval()
    results = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        results[split] = losses.mean().item()
    model.train()
    return results

# ── training loop ─────────────────────────────────────────────────────────────

loss_history = []

for step in range(max_iters):

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()

    if step % eval_interval == 0:
        losses = estimate_loss()
        current_lr = scheduler.get_last_lr()[0]
        msg = (
            f"STEP {step:4d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f} | "
            f"lr {current_lr:.2e}"
        )
        print(msg)
        log(msg)
        log("----")

        loss_history.append((step, losses["train"], losses["val"]))

        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=200, temperature=temperature, top_k=top_k)
        sample = tokenizer.decode(generated[0].tolist())
        log("SAMPLE:")
        log(sample)
        log("====\n")

        ckpt_path = ROOT / f"model_step_{step}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "step": step,
            "val_loss": losses["val"],
        }, ckpt_path)
        log(f"CHECKPOINT SAVED: {ckpt_path.name}")
        log("====\n")

# ── final generation ──────────────────────────────────────────────────────────

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    generated = model.generate(context, max_new_tokens=500, temperature=temperature, top_k=top_k)

final_text = tokenizer.decode(generated[0].tolist())
print("\n--- FINAL GENERATED TEXT ---\n")
print(final_text)
log("FINAL SAMPLE:")
log(final_text)
log("====")

# ── loss curve ────────────────────────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt
    steps      = [r[0] for r in loss_history]
    train_vals = [r[1] for r in loss_history]
    val_vals   = [r[2] for r in loss_history]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, train_vals, label="train")
    plt.plot(steps, val_vals,   label="val", linestyle="--")
    plt.title("Training / Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "loss_curve.png", dpi=120)
    plt.show()
except Exception:
    pass
