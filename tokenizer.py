from pathlib import Path

# Load cleaned text
text_path = Path("processed/dialogue.txt")

with open(text_path, "r", encoding="utf-8") as f:
    text = f.read()

print("Dataset length:", len(text))

# Get unique characters
chars = sorted(list(set(text)))

# Vocabulary size
vocab_size = len(chars)

print("Vocabulary size:", vocab_size)
print(chars)

# Character → Integer
stoi = {ch: i for i, ch in enumerate(chars)}

# Integer → Character
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder function
def encode(s):
    return [stoi[c] for c in s]

# Decoder function
def decode(tokens):
    return "".join([itos[t] for t in tokens])

# Test
sample = "Ayanokoji"

encoded = encode(sample)
decoded = decode(encoded)

print("\nSample:")
print(sample)

print("\nEncoded:")
print(encoded)

print("\nDecoded:")
print(decoded)