import torch
import torch.nn as nn
import torch.nn.functional as F

from self_attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_size, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(embed_size, num_heads, block_size, dropout=dropout)
        self.ff = FeedForward(embed_size, dropout=dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.sa(self.ln1(x)))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(embed_size, num_heads, block_size, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.drop(token_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    def _sample_next(self, logits, temperature, top_k):
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            topk_vals, _ = torch.topk(logits, k)
            logits = logits.masked_fill(logits < topk_vals[:, -1:], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            next_idx = self._sample_next(logits, temperature, top_k)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            next_idx = self._sample_next(logits, temperature, top_k)
            idx = torch.cat((idx, next_idx), dim=1)
            yield next_idx, idx
