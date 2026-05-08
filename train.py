import torch
from pathlib import Path

# -------------------------
# LOAD DATA
# -------------------------

text_path = Path("processed/dialogue.txt")

with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

print("Dataset length:", len(text))

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

# Encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

print("Tensor shape:", data.shape)

# -------------------------
# TRAIN / VALID SPLIT
# -------------------------

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print("Train size:", len(train_data))
print("Val size:", len(val_data))

# -------------------------
# CONTEXT WINDOW
# -------------------------

block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size + 1]

print("\nINPUT:")
print(x)

print("\nTARGET:")
print(y)

print("\nTraining examples:\n")

for t in range(block_size):

    context = x[:t+1]
    target = y[t]

    print(
        f"When input is {context.tolist()} "
        f"the target is {target}"
    )