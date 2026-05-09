import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, embed_size, head_size, block_size, dropout=0.1):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_drop(wei)

        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = embed_size // num_heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_size, head_size, block_size, dropout=dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_size, embed_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.drop(self.proj(out))
