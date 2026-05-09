import torch
import torch.nn as nn
import torch.nn.functional as F

from self_attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Simple MLP block (post-attention processing)
    """

    def __init__(self, embed_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block:
    Attention + FeedForward + Residual connections
    """

    def __init__(self, embed_size, num_heads, block_size):
        super().__init__()

        self.sa = MultiHeadAttention(embed_size, num_heads, block_size)
        self.ff = FeedForward(embed_size)

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):

        # attention + residual
        x = x + self.sa(self.ln1(x))

        # feedforward + residual
        x = x + self.ff(self.ln2(x))

        return x


class TransformerLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size, block_size, num_layers, num_heads):
        super().__init__()

        self.block_size = block_size

        # token + position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)

        # transformer blocks
        self.blocks = nn.Sequential(*[
            Block(embed_size, num_heads, block_size)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_size)

        # output head (logits over vocab)
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        token_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )

        x = token_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                topk_values, _ = torch.topk(logits, top_k)
                cutoff = topk_values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < cutoff,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_idx), dim=1)

        return idx
