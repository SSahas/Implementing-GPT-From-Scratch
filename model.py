import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional



class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.1):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads = 12,
        dim_head = 64,
        dropout = 0.1
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.inner_dim = inner_dim
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)


        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _, heads = *x.shape, self.heads
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        inner_dim = self.inner_dim
        
        q = q.view(batch_size, seq_len, heads, inner_dim // heads).permute(0, 2, 1, 3) 
        k = k.view(batch_size, seq_len, heads, inner_dim // heads).permute(0, 2, 3, 1)
        v = v.view(batch_size, seq_len, heads, inner_dim // heads).permute(0, 2, 1, 3)
        
        q = q * self.scale

        similarities = q @ k

        mask_value = -torch.finfo(similarities.dtype).max


        i, j = similarities.shape[-2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
        similarities = similarities.masked_fill(causal_mask, mask_value)


        attn = similarities.softmax(dim = -1)
        attn = self.dropout(attn)


        out = attn @ v


        out = out.permute(0, 2, 1, 3)
        out = out.reshape(batch_size, seq_len, inner_dim)
        

        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_heads: int):
        super().__init__()
        dim_head = n_embd // n_heads
        self.attention = AttentionBlock(dim=n_embd, heads=n_heads, dim_head=dim_head)
        self.feed_forward = FeedForward(n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class DecoderOnlyModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding_table = nn.Embedding(config['block_size'], config['n_embd'])
        self.decoder_stack = nn.Sequential(*[Block(config['n_embd'], config['n_heads']) for _ in range(config['n_layers'])])
        self.final_layer_norm = nn.LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

    def forward(self, batch: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = batch.shape
        batch_embeds = self.shared(batch)
        pos_emb = self.position_embedding_table(torch.arange(T, device=batch.device))
        x = batch_embeds + pos_emb
        x = self.decoder_stack(x)
        x = self.final_layer_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, do_sample: bool = True, top_k: int = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

