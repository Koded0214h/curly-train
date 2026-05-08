import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

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

# Encode full dataset
data = torch.tensor(encode(text), dtype=torch.long)

# -------------------------
# TRAIN / VAL SPLIT
# -------------------------

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# -------------------------
# BATCHING
# -------------------------

block_size = 8
batch_size = 32

def get_batch(split):

    data_source = train_data if split == "train" else val_data

    ix = torch.randint(len(data_source) - block_size, (batch_size,))

    x = torch.stack([data_source[i:i+block_size] for i in ix])

    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])

    return x, y

# -------------------------
# BIGRAM MODEL
# -------------------------

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # token embeddings
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            vocab_size
        )

    def forward(self, idx, targets=None):

        # idx shape: (B, T)
        logits = self.token_embedding_table(idx)

        # logits shape:
        # (B, T, vocab_size)

        if targets is None:
            loss = None

        else:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, loss = self(idx)

            # focus only on last token
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat((idx, next_idx), dim=1)

        return idx

# -------------------------
# CREATE MODEL
# -------------------------

model = BigramLanguageModel(vocab_size)

# optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-2
)

# -------------------------
# TRAINING LOOP
# -------------------------

max_iters = 3000
eval_interval = 300

for iter in range(max_iters):

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    if iter % eval_interval == 0:
        print(f"step {iter}: loss {loss.item():.4f}")

# -------------------------
# GENERATE TEXT
# -------------------------

context = torch.zeros((1, 1), dtype=torch.long)

generated = model.generate(
    context,
    max_new_tokens=500
)

print("\n--- GENERATED TEXT ---\n")

print(decode(generated[0].tolist()))