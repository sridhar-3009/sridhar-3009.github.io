---
layout: post
title: "KV Cache Compression: How LLMs Handle Million-Token Contexts Without Running Out of Memory"
date: 2026-05-02
description: Every token you generate costs memory — permanently. At 1M token contexts, the KV cache alone can consume 100GB+ of VRAM. Here's how modern LLMs compress it without losing what matters.
tags: [KV Cache, Attention, LLMs, Memory Optimization, GQA, Quantization, Long Context, PyTorch]
categories: deep-learning
giscus_comments: false
related_posts: false
---

# KV Cache Compression: How LLMs Handle Million-Token Contexts Without Running Out of Memory

Gemini 1.5 has a 1M token context window. Claude 3.5 handles 200K. GPT-4 Turbo runs 128K. These numbers sound like marketing — until you realize what they actually require in memory.

Every token the model processes stores two vectors in GPU memory: a **key** and a **value**, for every attention head, in every layer. These stay resident in VRAM for the entire generation session. With a 70B model running at 128K context, the KV cache alone consumes **over 100GB of VRAM** — more than most GPU clusters have available.

Long context isn't free. It's a memory engineering problem. The techniques that solve it — GQA, quantization, eviction policies, sliding windows — are what separate a research demo from something you can actually deploy.

This post builds from first principles: what the KV cache is and why it matters, then every major compression technique with the intuition before the code.

---

## What Is the KV Cache, and Why Does It Exist?

To understand the cache, you need to understand the cost without it.

When a transformer generates text, it works **one token at a time**. To generate token number 500, it needs to run attention over all 499 previous tokens — computing how much each prior token should influence the current one. That requires the **key** and **value** vectors for all 499 previous tokens.

Here's the problem: those key and value vectors were already computed when those tokens were first processed. Without a cache, you'd recompute them from scratch every single step. To generate a 1000-token response, you'd compute the keys and values for token 1 a thousand times, for token 2 a total of 999 times, and so on — a staggering amount of redundant work.

The **KV cache** solves this by storing the key and value vectors as they're computed, and reusing them on every subsequent step. It's the difference between O(n²) recomputation and O(n) incremental updates. Without it, even a 4K-token generation would be unbearably slow.

Think of it like a notepad. Every time you read a new sentence in a book, instead of re-reading every previous sentence to remember what happened, you jot down the key facts. The notepad grows with every page — but you never have to go back to re-read.

Here's a minimal implementation to make it concrete:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionWithKVCache(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache: dict | None = None):
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # If cache exists, prepend previous K/V to current ones
        if kv_cache is not None:
            if "K" in kv_cache:
                K = torch.cat([kv_cache["K"], K], dim=2)
                V = torch.cat([kv_cache["V"], V], dim=2)
            kv_cache["K"] = K
            kv_cache["V"] = V

        scale = self.d_k ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, V)

        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.d_k)
        return self.W_o(out), kv_cache
```

The key line is `K = torch.cat([kv_cache["K"], K], dim=2)` — we're prepending everything we computed before. The cache just grows, one token at a time, for the entire session.

---

## The Memory Problem at Scale

Here's where things get alarming. Each cached token stores:

```
2 vectors (K and V) × num_layers × num_heads × head_dim × bytes_per_float
```

For **Llama 3 70B** in fp16 (2 bytes per float):

```
2 × 80 layers × 64 heads × 128 head_dim × 2 bytes = ~2.5 MB per token
```

That's 2.5 megabytes for a **single token**. Now scale it:

- At 4K tokens (a short conversation): **10 GB**
- At 32K tokens (a long document): **80 GB**
- At 128K tokens (a book): **320 GB**

The model weights themselves are only ~140GB. At 128K context, the **KV cache is bigger than the model**. This is why long context is hard — it's not an architecture problem, it's a memory engineering problem.

Every technique in this post is an attack on that 2.5 MB/token number from a different angle.

---

## Technique 1: Multi-Query Attention (MQA) — One Key/Value for All

The most aggressive fix is also the most obvious: instead of storing separate key and value vectors for every attention head, store **just one shared pair** that all heads use.

In standard multi-head attention (MHA), if you have 64 heads, you have 64 independent K/V projections. Each head learns to attend differently — one head might focus on syntax, another on entity references, another on recency. That diversity is valuable, but it's expensive: 64 heads means 64× the cache storage.

**MQA says**: all heads share one K/V. They still each have their own queries (so they can ask different questions), but they all look up the same keys and values. Like a classroom where every student asks different questions, but they're all reading from the same textbook.

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)   # full — one per head
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)  # single shared key
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)  # single shared value
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache: dict | None = None):
        B, T, _ = x.shape

        # Full queries: each head asks its own question
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Single K and V: shape (B, 1, T, d_k)
        K = self.W_k(x).unsqueeze(1)
        V = self.W_v(x).unsqueeze(1)

        if kv_cache is not None:
            if "K" in kv_cache:
                K = torch.cat([kv_cache["K"], K], dim=2)
                V = torch.cat([kv_cache["V"], V], dim=2)
            kv_cache["K"] = K
            kv_cache["V"] = V

        # Broadcast the single K/V across all query heads
        K = K.expand(-1, self.num_heads, -1, -1)
        V = V.expand(-1, self.num_heads, -1, -1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        out = torch.matmul(F.softmax(scores, dim=-1), V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out), kv_cache
```

Notice `W_k` and `W_v` project to `d_k` (one head's worth) instead of `d_model` (all heads' worth). The `.expand()` call broadcasts that single K/V to all heads without actually copying memory — efficient but conceptually it's one shared lookup.

**Memory reduction**: 64× on the KV cache for a 64-head model. From 320GB to 5GB at 128K context for Llama 70B. Dramatic.

**The cost**: measurable quality degradation. When all heads share the same keys and values, they lose their ability to specialize in different types of information. A head that wants to focus on syntactic structure and a head that wants to focus on entity coreference are now forced to look at the same representation — they can ask different questions, but they're looking at the same answers. Some tasks suffer meaningfully.

Used by: **PaLM**, **Falcon**, **StarCoder**. Mostly superseded by GQA.

---

## Technique 2: Grouped Query Attention (GQA) — The Goldilocks Solution

MQA went too far. Standard MHA is too expensive. **GQA** finds the middle ground: instead of one K/V group or 64 K/V groups, use somewhere in between — say, 8 groups for a 64-head model.

Each group of 8 query heads shares one K/V pair. You get 8× memory reduction (vs MHA's baseline) while giving each group of heads some independence. The heads within a group share a lookup table, but different groups have different lookup tables.

Imagine 64 students split into 8 study groups of 8. Each group shares one set of notes. Students within a group see the same notes, but groups can take different notes. It's not as rich as everyone having private notes, but it's much better than the entire class sharing one set.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads  # heads per group
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache: dict | None = None):
        B, T, _ = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            if "K" in kv_cache:
                K = torch.cat([kv_cache["K"], K], dim=2)
                V = torch.cat([kv_cache["V"], V], dim=2)
            kv_cache["K"] = K
            kv_cache["V"] = V

        # Expand each KV group to cover its assigned query heads
        # (B, 8 groups, seq, d_k) → (B, 64 heads, seq, d_k)
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        out = torch.matmul(F.softmax(scores, dim=-1), V)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out), kv_cache
```

The critical line is `repeat_interleave` — it takes group 1's K/V and assigns it to heads 1–8, group 2's to heads 9–16, and so on. Only 8 K/V pairs get stored in the cache, not 64.

**Real-world impact (Llama 3 70B, 128K context):**

| Config | KV Heads | Cache Size |
|---|---|---|
| Standard MHA | 64 | 320 GB |
| GQA (8 KV heads) | 8 | **40 GB** |
| MQA | 1 | 5 GB |

40GB is actually manageable on a pair of A100s. Quality versus MHA: barely distinguishable on standard benchmarks. This is why **GQA is now the default** in every modern open-source model — Llama 2/3, Mistral, Gemma, Falcon 180B, Qwen, Phi. If you're building a transformer today, GQA is the starting point, not MHA.

---

## Technique 3: KV Cache Quantization — Shrink Each Number

GQA reduces the **number** of vectors stored. Quantization reduces the **size of each number** in those vectors.

By default, floating point numbers in a transformer are stored in fp16 — 16 bits, or 2 bytes each. The insight behind quantization is that you don't need full 16-bit precision for every cached value. Most of the information is preserved with 8 bits (int8) or even 4 bits (int4). The numbers get slightly wrong, but not wrong enough to matter for generation quality in most cases.

The process works like this: find the largest absolute value in a vector (the scale), divide everything by that scale to fit it into the smaller integer range, store the integer and the scale separately, then multiply back (dequantize) when you need the actual values. It's lossy compression — like JPEG for your attention vectors.

```python
class QuantizedKVCache:
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.cache_k: list[torch.Tensor] = []
        self.cache_v: list[torch.Tensor] = []
        self.scales_k: list[torch.Tensor] = []
        self.scales_v: list[torch.Tensor] = []

    def _quantize(self, x: torch.Tensor):
        # Find the max absolute value per token vector — this becomes our scale
        max_val = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = max_val / (2 ** (self.bits - 1) - 1)

        if self.bits == 8:
            # Compress float range into -128..127
            x_q = (x / scale).round().clamp(-128, 127).to(torch.int8)
        else:
            # 4-bit: compress into -8..7
            x_q = (x / scale).round().clamp(-8, 7).to(torch.int8)

        return x_q, scale  # store both the compressed value and how to undo it

    def _dequantize(self, x_q: torch.Tensor, scale: torch.Tensor):
        return x_q.float() * scale  # undo compression

    def append(self, K: torch.Tensor, V: torch.Tensor):
        K_q, scale_k = self._quantize(K)
        V_q, scale_v = self._quantize(V)
        self.cache_k.append(K_q)
        self.cache_v.append(V_q)
        self.scales_k.append(scale_k)
        self.scales_v.append(scale_v)

    def get(self):
        # Dequantize everything before attention computation
        K = torch.cat([self._dequantize(k, s) for k, s in zip(self.cache_k, self.scales_k)], dim=2)
        V = torch.cat([self._dequantize(v, s) for v, s in zip(self.cache_v, self.scales_v)], dim=2)
        return K, V
```

Every token gets quantized as it enters the cache and dequantized just before the attention computation. The GPU stores the compressed integers; the scale factor is tiny (one number per token vector). Net effect: half the memory for int8, quarter for int4.

**Memory savings vs fp16:**

| Precision | Bytes/value | Reduction |
|---|---|---|
| fp16 (baseline) | 2 | 1× |
| int8 | 1 | 2× |
| int4 | 0.5 | 4× |
| int2 | 0.25 | 8× |

The 2024 **KVQuant** paper pushed this further with non-uniform quantization — instead of a uniform grid of integer values, it calibrates the grid shape per-channel to match the actual distribution of key/value vectors. Result: fp16-level perplexity at 2-bit precision. That's a 8× memory reduction with essentially zero quality loss.

Used by: **vLLM** (int8 KV), **llama.cpp**, **TGI**, every major serving framework.

---

## Technique 4: Token Eviction — H2O

The previous techniques compress how we store tokens. This one asks a different question: **do we need to store all tokens at all?**

The answer is no — and the reason is one of the most striking empirical findings in LLM research. When you visualize attention weights across a long generation, you find that attention is wildly **non-uniform**. A tiny fraction of tokens — around 5% — receive roughly 80% of all attention mass. The rest barely get noticed.

These frequently attended tokens are called **heavy hitters**. They tend to be things like: key facts introduced early in the prompt, entities that the model keeps referencing, structural tokens like paragraph breaks or list markers. The model keeps coming back to them.

The remaining 95% of tokens get attended to occasionally or never. Evicting them would barely change the model's behavior — but it would dramatically shrink the cache.

**H2O** (Heavy-Hitter Oracle) implements this idea: track cumulative attention scores for every cached token. When the cache exceeds a budget, evict the tokens with the lowest cumulative scores. Always keep the most recent token regardless (since it's needed for immediate context).

```python
class H2OKVCache:
    def __init__(self, max_cache_size: int, num_heads: int, d_k: int):
        self.max_cache_size = max_cache_size
        self.num_heads = num_heads
        self.d_k = d_k
        self.K = None
        self.V = None
        # Running tally: how much total attention has each token received?
        self.attention_scores = None

    def update(self, K_new, V_new, attn_weights):
        """
        Called each generation step.
        attn_weights: how much the new token attended to each cached token.
        We use this to update each cached token's importance score.
        """
        if self.K is None:
            self.K = K_new
            self.V = V_new
            self.attention_scores = torch.zeros(K_new.shape[0], self.num_heads, 1, device=K_new.device)
            return

        # Add this step's attention to the running total for each token
        self.attention_scores += attn_weights.sum(dim=2, keepdim=True).transpose(2, 3)

        # Append new token to cache
        self.K = torch.cat([self.K, K_new], dim=2)
        self.V = torch.cat([self.V, V_new], dim=2)
        new_score = torch.zeros(K_new.shape[0], self.num_heads, 1, device=K_new.device)
        self.attention_scores = torch.cat([self.attention_scores, new_score], dim=2)

        # Over budget? Evict the least important tokens
        if self.K.shape[2] > self.max_cache_size:
            self._evict()

    def _evict(self):
        seq_len = self.K.shape[2]
        keep = self.max_cache_size
        recent_idx = seq_len - 1

        # Average importance across all heads
        importance = self.attention_scores.mean(dim=1).squeeze(1)  # (batch, seq)

        # Protect the most recent token — always needed
        scores_no_recent = importance.clone()
        scores_no_recent[:, recent_idx] = float('-inf')

        # Take the (keep-1) highest-importance historical tokens, plus the recent one
        _, top_indices = scores_no_recent.topk(keep - 1, dim=-1)
        keep_indices = torch.cat([
            top_indices,
            torch.tensor([[recent_idx]], device=top_indices.device).expand(top_indices.shape[0], -1)
        ], dim=-1)
        keep_indices, _ = keep_indices.sort(dim=-1)

        # Discard the evicted tokens from K, V, and score trackers
        idx = keep_indices.unsqueeze(1).unsqueeze(-1).expand(-1, self.num_heads, -1, self.d_k)
        self.K = torch.gather(self.K, 2, idx)
        self.V = torch.gather(self.V, 2, idx)
        self.attention_scores = torch.gather(
            self.attention_scores, 2,
            keep_indices.unsqueeze(1).expand(-1, self.num_heads, -1)
        )
```

The `_evict` method is the core: sort tokens by cumulative attention received, keep the top-K, drop the rest. The recently added token is always protected from eviction because it hasn't had time to accumulate attention scores yet.

**H2O results**: 20× cache compression with less than 1% accuracy degradation on most benchmarks. Set a cache budget of 5% of your sequence length and you barely notice the difference on most tasks.

Why? Because the model was already ignoring 95% of the tokens. Evicting them just makes the neglect explicit.

---

## Technique 5: StreamingLLM — Infinite Context at Fixed Memory

H2O evicts the least-attended tokens. But researchers at MIT made a more fundamental observation: **not all tokens are equal from the moment they're created**.

When you visualize attention patterns across thousands of generation steps, two patterns appear consistently:

**Pattern 1 — Locality**: most attention concentrates on the last few hundred tokens. The model cares most about what it just said. This is intuitive — you need your immediate context to write coherently.

**Pattern 2 — Attention sinks**: the very first tokens in any sequence (positions 0, 1, 2, 3) receive surprisingly high attention throughout the entire generation — even when their content is completely irrelevant to the current generation step. A document about quantum physics will still have heavy attention on its first four tokens, even 50,000 tokens later.

This second pattern is counterintuitive and strange. Why would a model keep attending to the first few words of a document it's now thousands of tokens past?

The answer is a training artifact. During training, the softmax over attention scores must sum to 1. When a token has nothing relevant to attend to, the probability mass has to go somewhere. The model learned to dump it onto the first few tokens as a kind of "soft null" — they absorb the leftover probability. They're not semantically important, they're structurally necessary as garbage collectors for the softmax.

**StreamingLLM** uses both observations to build a fixed-size cache for infinite-length generation: always keep the first 4 tokens (the sinks) plus a rolling window of the most recent N tokens. Everything between the sink tokens and the window gets dropped permanently.

```python
class StreamingKVCache:
    def __init__(self, sink_size: int = 4, window_size: int = 1020):
        """
        Total cache = sink_size + window_size (constant, regardless of sequence length).
        The sink tokens anchor the softmax. The window provides local context.
        Everything older than the window is evicted.
        """
        self.sink_size = sink_size
        self.window_size = window_size
        self.sink_K = self.sink_V = None
        self.window_K = self.window_V = None

    def update(self, K_new: torch.Tensor, V_new: torch.Tensor):
        # Phase 1: fill the sink buffer with the first few tokens
        if self.sink_K is None or self.sink_K.shape[2] < self.sink_size:
            self.sink_K = torch.cat([self.sink_K, K_new], dim=2) if self.sink_K is not None else K_new
            self.sink_V = torch.cat([self.sink_V, V_new], dim=2) if self.sink_V is not None else V_new
            return self.get()

        # Phase 2: everything after the sink goes into the rolling window
        if self.window_K is None:
            self.window_K = K_new
            self.window_V = V_new
        else:
            self.window_K = torch.cat([self.window_K, K_new], dim=2)
            self.window_V = torch.cat([self.window_V, V_new], dim=2)

            # Window full — drop the oldest token (slide forward)
            if self.window_K.shape[2] > self.window_size:
                self.window_K = self.window_K[:, :, 1:, :]
                self.window_V = self.window_V[:, :, 1:, :]

        return self.get()

    def get(self):
        if self.window_K is None:
            return self.sink_K, self.sink_V
        return (
            torch.cat([self.sink_K, self.window_K], dim=2),
            torch.cat([self.sink_V, self.window_V], dim=2),
        )
```

The `window_K[:, :, 1:, :]` line is the sliding — every new token pushes the oldest window token off the edge. But the sink tokens never move.

**What happens without the sinks?** If you just use a sliding window with no sink tokens, the model collapses — perplexity spikes dramatically after the context fills. The softmax has nowhere to send its garbage probability mass and the distribution breaks down. The 4 sink tokens are tiny in cost but critical in function.

**StreamingLLM result**: Llama 2 ran on 4-million-token documents using a 1024-token cache. 22× memory reduction. Constant memory usage no matter how long the generation runs.

Used by: **Mistral** (sliding window attention), **Gemma**, **Longformer**.

---

## Technique 6: SnapKV — Compress Before You Generate

All the techniques above deal with the cache during generation — either evicting tokens as you go or using a fixed window. **SnapKV** takes a step back and asks: can we compress the cache *before* generation even starts?

When an LLM processes a long prompt (say, 50,000 tokens of context before generating an answer), it runs what's called a **prefill** step — a full forward pass over the entire prompt in parallel. During this prefill, every token attends to every other token, producing a full attention matrix.

The SnapKV insight: that prefill attention matrix tells you which tokens are important before generation begins. Tokens that many other tokens attend to during prefill are likely to stay important during decoding. You can use the prefill attention to pre-select which tokens to keep in the cache, and discard the rest before decoding starts.

```python
def snapkv_compress(K, V, attn_weights, compression_ratio=0.3):
    """
    Run this after prefill, before generation starts.

    We look at how much attention each token received from others during prefill.
    High-attention tokens are "important" — keep them.
    Low-attention tokens are likely padding or filler — discard them.

    compression_ratio = 0.3 means keep only the top 30% of tokens.
    """
    B, H, S, d_k = K.shape
    keep_count = max(1, int(S * compression_ratio))

    # Column sums of attention matrix: how much did each token receive?
    # Shape: (B, H, S) — one score per token per head
    importance = attn_weights.sum(dim=2)

    # Average importance across all heads to get a single score per token
    avg_importance = importance.mean(dim=1)  # (B, S)

    # Pick the top-k most attended-to tokens
    _, top_indices = avg_importance.topk(keep_count, dim=-1)
    top_indices, _ = top_indices.sort(dim=-1)  # preserve original order

    # Build the compressed cache — only these tokens enter generation
    idx = top_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, -1, d_k)
    K_compressed = torch.gather(K, 2, idx)
    V_compressed = torch.gather(V, 2, idx)

    return K_compressed, V_compressed
```

The key idea is `attn_weights.sum(dim=2)` — summing across the "who is attending" dimension to get "how much is each token being attended to." High score = many tokens find this one important = keep it. Low score = nobody's looking at this = safe to discard.

**SnapKV results**: 3.6× cache reduction, 3.8× speedup on generation, less than 1% accuracy loss on LongBench. Particularly strong for RAG-style tasks where the prompt is mostly retrieved context — most chunks aren't relevant to the answer, and SnapKV confidently discards them based on prefill attention.

The difference from H2O: SnapKV makes one compression decision upfront (at prefill time) and never changes it. H2O continuously evicts during generation. SnapKV is simpler and faster; H2O adapts dynamically as the generation unfolds.

---

## The Full Picture: Stacking Techniques

These aren't competing approaches — they're layers. Production systems combine them:

```
GQA              → reduces number of K/V vectors stored
  + INT8 quant   → halves bytes per stored value
  + H2O eviction → further reduces token count dynamically
  + Sliding window → caps total cache at constant size
```

Here's what that stacking does for **Llama 3 70B at 128K context**:

| What's applied | Cache Size |
|---|---|
| Baseline MHA, fp16 | 320 GB |
| + GQA (8 KV heads) | 40 GB |
| + INT8 quantization | 20 GB |
| + H2O (20% budget) | **4 GB** |

From 320 GB to 4 GB. Same model. Same outputs (approximately). Now fits on a single A100 with room for the model weights too.

---

## Why This Also Matters for Speed

Smaller cache isn't just about fitting in memory — it directly affects **how fast the model generates tokens**.

Every time the model generates a new token, it reads the entire KV cache from GPU high-bandwidth memory (HBM). The speed of that read is bounded by HBM bandwidth — for an A100, that's about 2 TB/s. At 128K context with GQA 70B:

- Cache size: ~40 GB
- Time to read: 40 GB ÷ 2 TB/s = **20ms per token** → ceiling of 50 tokens/second

Apply H2O + quantization to bring cache to 4 GB:

- Time to read: 4 GB ÷ 2 TB/s = **2ms per token** → ceiling of 500 tokens/second

A **10× throughput gain** from the same GPU, same model, same weights. This is why inference frameworks like vLLM, TensorRT-LLM, and SGLang have invested so heavily in KV cache engineering — it's not just about memory, it's about the money you spend on compute per generated token.

---

## Key Takeaways

1. **KV cache is the dominant memory cost at long context** — at 128K tokens, the cache is bigger than the model weights. This is the problem every technique here is solving.

2. **GQA is the new baseline** — 8× memory reduction, negligible quality loss. Every serious model uses it now. MHA is legacy.

3. **Quantization stacks cleanly on top of GQA** — INT8 gives you 2× more, INT4 gives 4× more, 2-bit is viable with per-channel calibration.

4. **Most tokens don't matter** — 80% of attention concentrates on 5% of tokens. Eviction strategies (H2O, SnapKV) exploit this to compress aggressively without hurting quality.

5. **Attention sinks are a real phenomenon** — the first few tokens in any sequence must always be kept. Removing them breaks generation even when their content is irrelevant. This is a training artifact, not a design choice.

6. **Cache size = latency** — HBM bandwidth is the bottleneck during decoding. Smaller cache means faster reads means more tokens per second. Memory compression and speed optimization are the same problem.

---

## Further Reading

### KV Cache Fundamentals

- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)** — Kwon et al., UC Berkeley (2023)
  The vLLM paper. Introduces virtual memory paging for KV cache — eliminates fragmentation, enables dynamic batching across requests. The production standard for KV cache management.

- **[Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150)** — Shazeer, Google (2019)
  The original MQA paper. Shows that sharing KV heads barely hurts quality while massively reducing memory and improving decode speed. The idea GQA refined.

- **[GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)** — Ainslie et al., Google (2023)
  GQA paper. Introduces the grouped approach and shows how to convert existing MHA checkpoints via uptraining. The technique behind Llama 2, Mistral, and every modern open-source model.

### Quantization

- **[KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079)** — Hooper et al., UC Berkeley (2024)
  Non-uniform quantization calibrated per-channel. Achieves fp16 perplexity at 2-bit precision. The paper that proved INT2 KV cache is viable.

- **[KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)** — Liu et al. (2024)
  Keys and values have different numerical distributions — KIVI quantizes them differently. Clean result: 2-bit KV cache with almost no accuracy loss, no fine-tuning required.

### Eviction & Compression

- **[H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)** — Zhang et al., Rice University (2023)
  The paper that identified the heavy-hitter phenomenon. 20× cache compression with less than 1% degradation. Empirically shows that 80% of attention concentrates on 5% of tokens.

- **[SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469)** — Li et al. (2024)
  Prefill-time compression. Uses the attention matrix from prompt processing to predict which tokens will matter during decoding. 3.6× compression, 3.8× speedup.

- **[Efficient Streaming Language Models with Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453)** — Xiao et al., MIT (2023)
  Discovers and explains attention sinks. Shows that 4 sink tokens + rolling window enables infinite-length generation at constant memory. Clean, surprising, and practically useful.

### Systems

- **[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)** — Dao (2023)
  IO-aware attention that tiles computation to avoid materializing the full O(n²) attention matrix. The algorithm that makes long-context attention practical from a compute perspective — complements KV cache compression from the memory side.

- **[Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)** — Agrawal et al. (2023)
  Production serving paper on pipelining prefill and decode efficiently. The operational reality of running bounded KV cache systems at scale.

---

*The context window arms race isn't about model capability — it's about memory engineering. Every extra token you can fit in context at the same VRAM budget is a capability gain. The teams winning on long context aren't necessarily building smarter models. They're building smarter caches.*
