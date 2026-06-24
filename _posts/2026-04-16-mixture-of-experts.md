---
layout: post
title: "Mixture of Experts: How a 141B Model Runs Like a 39B"
date: 2026-04-16
description: GPT-4, Mixtral, and DeepSeek aren't the dense models most people imagine. They're sparse — activating only a fraction of their parameters per token. Here's the architecture that makes that possible, and why it changes everything about how we scale AI.
tags: [Mixture of Experts, MoE, Mixtral, DeepSeek, LLMs, Architecture, PyTorch]
categories: deep-learning
giscus_comments: false
related_posts: false
---

# Mixture of Experts: How a 141B Model Runs Like a 39B

When Mistral released Mixtral 8x7B in December 2023, people were confused. The model card said 46.7B total parameters but only 12.9B active per token. How does a 46B model run at the speed of a 13B one?

When DeepSeek-V3 launched in late 2024 with 671B total parameters but only 37B active, the confusion deepened. Numbers that large shouldn't fit on any reasonable hardware — but they did.

The answer is **Mixture of Experts** (MoE): a decades-old idea from the early 1990s that became the dominant architecture for frontier LLMs once engineers figured out how to make it work at scale.

The core insight is almost offensively simple: **you don't need to run every part of the network on every input.** Different tokens need different computations. A model smart enough to route each token to the relevant specialists — and ignore the rest — gets the knowledge of a huge model at the cost of a small one.

This post goes deep on how that works: the router, the expert layers, the training tricks, and the engineering tradeoffs that make production MoE systems possible.

---

## The Problem With Dense Models

In a standard transformer, every token passes through every parameter on every forward pass. A 70B dense model activates all 70B parameters for each token — a word in a poem, a number in an equation, a variable name in code. The model applies the full machinery regardless of whether it's relevant.

This is wasteful. A token in a Python function doesn't need the same computations as a token in a French poem. The model's knowledge of French grammar is entirely irrelevant when parsing `def forward(self, x):`.

Dense scaling has another problem: **compute scales with parameters**. Double the parameters, roughly double the FLOPs per token. At some point, training cost becomes the bottleneck — not capability.

MoE breaks this link between total parameters and per-token compute.

---

## The Architecture: Experts Replace the FFN

In a standard transformer block, the feed-forward network (FFN) is the largest sub-component — typically 4× the model dimension, applied identically to every token.

MoE replaces this single FFN with `N` independent FFNs — the **experts** — plus a lightweight **router** that selects which experts handle each token:

```
Standard Transformer Block:
  x → Self-Attention → Add & Norm → FFN → Add & Norm → output

MoE Transformer Block:
  x → Self-Attention → Add & Norm → Router → Top-K Experts → Add & Norm → output
```

The self-attention layers stay dense — every token attends to every other token. Only the FFN is replaced. This is the key design choice: attention handles context and relationships; experts handle token-level transformation.

---

## The Router: Making the Decision

The router is a small linear layer that maps each token's hidden state to a probability distribution over experts:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoERouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # A single linear layer: hidden_dim → num_experts logits
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x):
        """
        x: (batch_size * seq_len, hidden_dim)
        Returns:
            top_k_indices:  (batch_size * seq_len, top_k)  — which experts to use
            top_k_weights:  (batch_size * seq_len, top_k)  — how much to weight each
        """
        # Compute logits over all experts
        logits = self.gate(x)                          # (tokens, num_experts)

        # Softmax to get routing probabilities
        probs = F.softmax(logits, dim=-1)              # (tokens, num_experts)

        # Select top-K experts per token
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Renormalize so selected weights sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return top_k_indices, top_k_weights
```

For each token, the router outputs two things: **which K experts to use**, and **how much weight to give each one**. The final output is a weighted sum of those K experts' outputs.

Typical values: `num_experts = 8`, `top_k = 2`. So each token uses exactly 2 of 8 experts — activating 25% of the expert parameters.

---

## The Expert: Just an FFN

Each expert is a standard feed-forward network — nothing exotic:

```python
class Expert(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.SiLU(),               # SwiGLU variant used in Mixtral
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)
```

The size of each expert's FFN is typically the same as the FFN in a comparable dense model. So Mixtral 8x7B has 8 experts, each the size of a 7B model's FFN — but only 2 are active per token. The "8x7B" name reflects this: 8 experts, each ~7B FFN scale.

---

## Putting It Together: The MoE Layer

```python
class MoELayer(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = MoERouter(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        B, S, H = x.shape

        # Flatten to (batch*seq, hidden) — router operates per token
        x_flat = x.view(B * S, H)

        # Get routing decisions
        expert_indices, expert_weights = self.router(x_flat)
        # expert_indices:  (B*S, top_k)
        # expert_weights:  (B*S, top_k)

        # Collect expert outputs
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            # Which tokens go to this slot's expert?
            indices = expert_indices[:, k]      # (B*S,) — expert index per token
            weights = expert_weights[:, k]      # (B*S,) — weight per token

            # Process each expert's assigned tokens
            for expert_id in range(self.num_experts):
                token_mask = (indices == expert_id)
                if not token_mask.any():
                    continue

                expert_input = x_flat[token_mask]
                expert_output = self.experts[expert_id](expert_input)

                # Weighted accumulation
                output[token_mask] += weights[token_mask].unsqueeze(-1) * expert_output

        return output.view(B, S, H)
```

This is the naive implementation — readable but slow. Production systems batch all tokens destined for the same expert together (a single large matmul instead of a loop), which is why MoE requires specialized CUDA kernels in practice.

---

## The Load Balancing Problem

Here's the failure mode that makes MoE training notoriously tricky: **expert collapse**.

Without any intervention, the router quickly learns to always route tokens to the same 1–2 experts. Those experts see more gradients, get better faster, and attract even more tokens. The remaining experts starve — they're never trained because tokens never reach them. You end up with a 141B model that effectively uses 10B parameters.

The standard fix is an **auxiliary load balancing loss** that penalizes unequal expert utilization:

```python
def load_balancing_loss(router_probs, expert_indices, num_experts):
    """
    Encourages uniform distribution of tokens across experts.

    router_probs:   (num_tokens, num_experts) — softmax output of router
    expert_indices: (num_tokens, top_k)       — selected expert indices
    """
    num_tokens = router_probs.shape[0]

    # Fraction of tokens routed to each expert
    # (how often each expert is selected)
    one_hot = F.one_hot(expert_indices, num_classes=num_experts).float()
    tokens_per_expert = one_hot.sum(dim=1).mean(dim=0)  # (num_experts,)
    f = tokens_per_expert / tokens_per_expert.sum()

    # Average router probability assigned to each expert
    # (how confident the router is about each expert)
    p = router_probs.mean(dim=0)  # (num_experts,)

    # Loss = num_experts * sum(f_i * p_i)
    # Minimized when both f and p are uniform (1/num_experts each)
    loss = num_experts * (f * p).sum()
    return loss
```

This loss is added to the main language modeling loss with a small coefficient (typically `α = 0.01`). It gently pushes the router toward distributing tokens evenly without overwhelming the task objective.

---

## Expert Capacity: The Traffic Problem

Even with load balancing, real deployments face a harder constraint: **expert capacity**.

In distributed training, each expert lives on a specific device. If 1000 tokens all want expert #3 and only 100 want expert #7, expert #3's device is overwhelmed while #7's sits idle. You can't dynamically resize — the devices are fixed.

The solution is a **capacity factor**: set a maximum number of tokens each expert can process per forward pass. Tokens that exceed an expert's capacity are **dropped** — they don't get processed by any expert.

```python
def route_with_capacity(router_probs, top_k, capacity_factor=1.25):
    """
    Routes tokens to experts with a hard capacity ceiling.
    Excess tokens are dropped (their expert output is zeroed out).
    """
    num_tokens, num_experts = router_probs.shape
    # Max tokens any single expert will process
    capacity = int(capacity_factor * num_tokens * top_k / num_experts)

    top_k_weights, top_k_indices = torch.topk(router_probs, top_k, dim=-1)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    # Track how many tokens each expert has accepted
    expert_counts = torch.zeros(num_experts, dtype=torch.long)
    dispatch_mask = torch.zeros(num_tokens, top_k, dtype=torch.bool)

    for token_idx in range(num_tokens):
        for k in range(top_k):
            expert_id = top_k_indices[token_idx, k].item()
            if expert_counts[expert_id] < capacity:
                dispatch_mask[token_idx, k] = True
                expert_counts[expert_id] += 1
            # else: token is dropped for this expert slot

    return top_k_indices, top_k_weights, dispatch_mask
```

Token dropping introduces a tricky training dynamic: the model must learn to be robust to occasionally missing expert outputs. In practice, capacity factor ~1.25 means ~2–5% of tokens get dropped — small enough that the model learns to tolerate it.

---

## How Mixtral Does It

Mixtral 8x7B uses 8 experts per MoE layer with top-2 routing. Every other transformer layer is an MoE layer (the rest are standard dense attention + FFN). The architecture:

| Property                    | Value                      |
| --------------------------- | -------------------------- |
| Total parameters            | 46.7B                      |
| Active parameters per token | 12.9B                      |
| Experts per MoE layer       | 8                          |
| Active experts per token    | 2                          |
| Hidden dimension            | 4096                       |
| Expert FFN dimension        | 14336                      |
| Layers                      | 32 (alternating dense/MoE) |

The critical insight: **attention layers are dense** (all 12.9B active parameters), but the expert FFNs represent the bulk of "knowledge storage" — and only 2/8 = 25% of that storage is accessed per token.

Mixtral's routing is **token-level, not sequence-level**: the same word in different contexts can be routed to completely different experts. The router makes a fresh decision every token, every layer.

---

## How DeepSeek Pushes It Further

DeepSeek-V3 (671B total / 37B active) introduces two innovations beyond standard MoE:

**1. Fine-Grained Expert Segmentation**

Instead of 8 large experts, DeepSeek uses 256 small experts per layer with top-8 routing. Each expert is ~1/32 the size of a Mixtral expert. This gives the router much finer-grained control — you can build any "combination" of specializations rather than picking from 8 coarse buckets.

**2. Shared Experts**

DeepSeek designates a small number of experts as **always-active shared experts**. Every token passes through these, regardless of routing. The shared experts handle common, general-purpose transformations; the routed experts handle token-specific specialization.

```
DeepSeek MoE Layer:
  token → [Shared Expert 1] ──────────────────────────────┐
         → [Shared Expert 2] ──────────────────────────────┤
         → Router → Top-K from 256 routed experts ─────────┴→ sum → output
```

This hybrid approach lets the model separate "what every token needs" from "what this specific token needs" — a cleaner decomposition than pure routing.

---

## Why GPT-4 Is (Probably) MoE

OpenAI has never confirmed GPT-4's architecture. But the evidence is strong:

- Sam Altman confirmed GPT-4 cost ~$100M to train — consistent with MoE (you can train more total parameters for the same FLOP budget)
- Multiple credible leaks from 2023 describe GPT-4 as a ~1.8T parameter MoE with 16 experts, top-2 routing
- GPT-4's inference latency and throughput numbers are inconsistent with a single dense model at that capability level
- OpenAI's research trajectory — from GPT-3 (dense) through the sparse attention work — pointed toward MoE

The economics make it inevitable: a 1.8T MoE with 2/16 active experts costs the same to run as a ~200B dense model, but has the knowledge capacity of a 1.8T one. At frontier scale, there's no competitive reason to use dense.

---

## The Real Cost: Communication Overhead

MoE sounds like a free lunch. It isn't. The hidden cost is **all-to-all communication**.

In distributed training, model parameters are split across devices (tensor parallelism) or layers (pipeline parallelism). With MoE, you add a third dimension: **expert parallelism** — each expert lives on a different device.

This creates a routing problem at the hardware level: after the router decides which expert each token goes to, you have to physically ship that token's activations to the device that holds the relevant expert. After the expert processes it, the results ship back.

This **all-to-all collective** — where every device sends data to every other device — is expensive on current hardware. The NVLink bandwidth between GPUs is fast, but not free. At the scale of 256 experts across 32 nodes, this communication can dominate training time.

DeepSeek's engineering contribution (described in their training paper) is a custom all-to-all kernel that overlaps communication with computation — hiding the latency by starting the next layer's attention while the current layer's expert results are still in transit. This is where the real engineering work happens in production MoE systems.

---

## Dense vs. Sparse: When to Use Which

|                           | Dense                          | MoE                             |
| ------------------------- | ------------------------------ | ------------------------------- |
| Training FLOPs per token  | High                           | Low                             |
| Inference FLOPs per token | High                           | Low                             |
| Memory (weights)          | Low                            | High                            |
| Training stability        | High                           | Moderate (needs load balancing) |
| Serving complexity        | Simple                         | Complex (expert parallelism)    |
| Good for                  | Smaller models, simple serving | Frontier scale, research        |
| Bad for                   | Frontier scale (too expensive) | Edge/mobile deployment          |

The crossover point is roughly 30–70B parameters: below that, dense is simpler and sufficient; above that, MoE becomes the only economically viable path to higher capability.

---

## Key Takeaways

1. **MoE breaks the compute-parameter link** — you can have 10× more parameters for the same per-token FLOPs, because only a fraction of experts activate per token.

2. **The router is a learned function** — it figures out which tokens need which specializations entirely from gradient descent, with no human-defined expert roles.

3. **Load balancing is non-negotiable** — without the auxiliary loss, expert collapse happens within the first few thousand steps.

4. **Expert capacity introduces token dropping** — a necessary engineering compromise for distributed training that the model learns to tolerate.

5. **DeepSeek's fine-grained + shared expert design is the current state of the art** — more routing flexibility, cleaner separation of general vs. specialized computation.

6. **The real cost is communication, not compute** — building production MoE systems is a distributed systems problem as much as an ML problem.

---

## Further Reading

### Foundational Papers

- **[Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)** — Shazeer et al., Google Brain (2017)
  The paper that revived MoE for deep learning. Introduced the top-K gating mechanism and the load balancing auxiliary loss. Everything since builds on this.

- **[Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)** — Fedus et al., Google (2021)
  Simplified MoE to top-1 routing, showed you could scale to 1T+ parameters. First clear demonstration that MoE could match dense models at trillion-parameter scale.

- **[GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)** — Du et al., Google (2021)
  Trained a 1.2T MoE model using 1/3 the energy of GPT-3 training. Landmark result on MoE efficiency vs. dense models.

### Modern MoE Architectures

- **[Mixtral of Experts](https://arxiv.org/abs/2401.04088)** — Jiang et al., Mistral AI (2024)
  The architecture paper behind Mixtral 8x7B and 8x22B. Clean, readable description of token-level routing with top-2 selection. The open-source MoE baseline.

- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)** — DeepSeek AI (2024)
  671B parameters, 37B active, trained for ~$5.5M. Describes fine-grained expert segmentation, shared experts, and the custom all-to-all communication kernel. State of the art in MoE efficiency.

- **[DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)** — DeepSeek AI (2024)
  Predecessor to V3. Introduces the DeepSeekMoE architecture — the fine-grained + shared expert design — and validates it at scale.

### Training & Efficiency

- **[ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)** — Zoph et al., Google (2022)
  A detailed investigation into what makes MoE training unstable and how to fix it. Router z-loss, capacity factor tuning, and expert dropout. Essential reading before training your own MoE.

- **[Towards Understanding Mixture of Experts in Deep Learning](https://arxiv.org/abs/2208.02813)** — Chen et al. (2022)
  Theoretical analysis of why MoE works: shows that sparse routing implicitly implements a form of conditional computation that dense models can't replicate efficiently.

- **[Expert Choice Routing: Better MoE with Expert-Selected Tokens](https://arxiv.org/abs/2202.09368)** — Zhou et al., Google (2022)
  Inverts the routing: instead of tokens choosing experts, experts choose tokens. Eliminates dropped tokens, achieves perfect load balance. An elegant alternative to the standard approach.

### Interpretability

- **[Do Sparse Language Models Generalize Well?](https://arxiv.org/abs/2310.01786)** — Research into whether different experts actually specialize for different topics, syntactic structures, or languages. The empirical answer: yes, but it's messier than you'd hope. Some experts specialize cleanly; others are general-purpose.

---

_MoE is a bet on a simple idea: not every token needs the same computation. The remarkable thing is how much that bet pays off. A model that thinks selectively — routing each token to relevant specialists, ignoring the rest — turns out to be both more capable and cheaper to run than one that applies maximum force to everything. Intelligence, apparently, is less about brute force and more about knowing when not to think._
