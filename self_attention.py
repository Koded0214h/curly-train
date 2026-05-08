import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    """
    A single self-attention head (the core building block of transformers)
    """

    def __init__(self, embed_size, head_size, block_size):
        super().__init__()

        self.head_size = head_size
        self.embed_size = embed_size
        self.block_size = block_size

        # projections
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

        # causal mask (prevents looking into future tokens)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        """
        x shape: (B, T, C)
        """

        B, T, C = x.shape

        # project inputs into key/query/value spaces
        k = self.key(x)   # (B, T, H)
        q = self.query(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)

        # compute attention scores
        wei = q @ k.transpose(-2, -1)  # (B, T, T)

        # scale down for stability
        wei = wei / (self.head_size ** 0.5)

        # apply causal mask (no future tokens allowed)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # softmax → probabilities
        wei = F.softmax(wei, dim=-1)

        # weighted aggregation of values
        out = wei @ v  # (B, T, H)

        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads working in parallel
    """

    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()

        head_size = embed_size // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_size, head_size, block_size)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)

        return self.proj(out)