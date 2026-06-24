---
layout: post
title: "Understanding Transformers: The Architecture Behind Modern AI"
date: 2024-03-15
description: A deep dive into the transformer architecture — the building block of GPT, BERT, and virtually every modern AI system. We'll implement multi-head attention from scratch.
tags: [Transformers, Attention, LLMs, PyTorch]
categories: deep-learning
giscus_comments: false
related_posts: false
---

# Understanding Transformers: The Architecture Behind Modern AI

The transformer architecture, introduced in the landmark 2017 paper _"Attention Is All You Need"_ by Vaswani et al., has become the foundational building block of modern AI. From GPT-4 to BERT, from DALL-E to Whisper — they all share the same core mechanism: **self-attention**.

In this post, I'll demystify how transformers work, build intuition for attention, and implement a minimal transformer from scratch in PyTorch.

## The Core Idea: Attention

Before transformers, sequence models (RNNs, LSTMs) processed tokens sequentially — each step depended on the previous one. This created a bottleneck: long-range dependencies were hard to learn, and parallelization was impossible.

Transformers solve this with **attention**: every token can directly attend to every other token in a single pass.

The attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

Where:

- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I contain?"
- **V** (Value): "What do I actually pass forward?"

## Implementing Multi-Head Attention in PyTorch

Let's build it from scratch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Projection matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, heads, seq_len, d_k)
        Returns: (batch, heads, seq_len, d_k)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V), attn_weights

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project and reshape to multiple heads
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(attn_output)
```

## The Transformer Block

Each transformer block stacks:

1. **Multi-head self-attention**
2. **Add & Norm** (residual connection + layer norm)
3. **Feed-forward network** (two linear layers with ReLU)
4. **Add & Norm** again

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Self-attention with residual
        x = self.norm1(x + self.attention(x, mask))
        # Feed-forward with residual
        x = self.norm2(x + self.feed_forward(x))
        return x
```

## Why Multi-Head Attention?

Multiple heads allow the model to attend to different aspects simultaneously:

- One head might capture **syntactic relationships** (subject-verb agreement)
- Another might capture **semantic similarity** (synonyms)
- Yet another might capture **positional patterns** (nearby words)

Empirically, we see that different heads specialize in remarkably interpretable patterns.

## Positional Encoding

Since attention has no inherent sense of order, we inject position information using sinusoidal encodings:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 8192):
        super().__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

## Scaling to LLMs

Modern LLMs like GPT-4 stack 96+ transformer blocks with:

- **Flash Attention** for memory-efficient attention computation
- **Rotary Position Embeddings (RoPE)** for better length generalization
- **Grouped Query Attention (GQA)** for faster inference
- **SwiGLU** activation functions in the FFN
- **RMS Norm** instead of Layer Norm

These modifications dramatically improve training efficiency and model quality at scale.

## Key Takeaways

1. **Attention is the core primitive**: everything else (position encoding, FFN, norms) serves to make attention work well at scale.

2. **Parallelism wins**: unlike RNNs, all attention computations can happen simultaneously — this is what made scaling to trillion-token datasets possible.

3. **Residual streams matter**: the residual connections create a "highway" for gradients and information to flow through deep networks.

4. **Scale is all you need (mostly)**: the core architecture from 2017 hasn't changed dramatically — we've mainly learned that more data, more compute, and longer training produces emergent capabilities.

Understanding transformers is the foundational knowledge every modern ML engineer needs. In the next post, I'll cover **Flash Attention** — the algorithmic breakthrough that made training GPT-4 scale possible.

---

_Questions or thoughts? Reach out on [Twitter](https://twitter.com/saisridhartarra) or [LinkedIn](https://linkedin.com/in/saisridhartarra)._
