---
layout: post
title: "Superposition: How Neural Networks Secretly Cram More Concepts Than They Have Neurons"
date: 2026-03-31
description: Most engineers assume one neuron = one concept. They're wrong. Neural networks pack thousands of overlapping features into far fewer neurons using a trick from high-dimensional geometry — and understanding this changes everything about how we interpret AI.
tags: [Neural Networks, Interpretability, Superposition, Mechanistic Interpretability, AI Safety, Deep Learning]
categories: ai-research
giscus_comments: false
related_posts: false
---

# Superposition: How Neural Networks Secretly Cram More Concepts Than They Have Neurons

Most people learning about neural networks are taught a clean story: train a network long enough, and individual neurons will specialize. Neuron 347 fires for "cat ears." Neuron 892 responds to "the word Paris." The network is a tidy collection of little detectors.

This story is mostly **wrong** — and the reality is far stranger, more elegant, and more alarming.

In 2022, Anthropic researchers published _"Toy Models of Superposition"_, a paper that quietly shook the foundations of mechanistic interpretability. Their finding: **neural networks routinely store far more features than they have neurons**, by encoding multiple concepts in the same neurons simultaneously, using the geometry of high-dimensional space as a compression trick.

This post unpacks what that means, how it works mathematically, and why it has enormous implications for AI safety and interpretability.

---

## The Problem: More Concepts Than Dimensions

A language model has a residual stream — a vector of, say, 4096 floating-point numbers that carries information through the network. The number 4096 is fixed. But the world has millions of concepts a model needs to track: whether a token is a verb, whether a number is prime, whether the sentiment is negative, the geographic region being discussed, the author's tone, and thousands more.

You can't fit a million concepts into 4096 independent slots. So what does the network do?

It cheats. Brilliantly.

---

## Polysemanticity: One Neuron, Many Meanings

When you look at real neurons in large models, you find something uncomfortable: **a single neuron fires for wildly unrelated things**.

A famous example from GPT-2 analysis: neuron `#2747` in a specific MLP layer activates strongly for:

- The word "banana"
- Code comments containing `//`
- References to basketball

This is **polysemanticity** — one computational unit, many semantic roles. It's not noise. It's systematic. And it's everywhere in large models.

This immediately breaks a key assumption of interpretability: if neurons aren't monosemantic (one concept per neuron), then you can't understand the network by studying neurons one at a time.

---

## The Superposition Hypothesis

Anthropic's researchers asked: _why_ does polysemanticity happen? Their answer is the **superposition hypothesis**.

The core idea: **a network with `n` neurons can represent up to `exp(n)` features** — not just `n` — if it accepts some interference between features. The network trades precision for capacity.

Here's the intuition. Consider two vectors in high-dimensional space. If they're perfectly orthogonal (at 90°), they don't interfere with each other at all. But if you have more features than dimensions, you can't keep everything orthogonal. Instead, you pack features into _nearly orthogonal_ directions — vectors that are only slightly correlated.

The **Johnson-Lindenstrauss Lemma** from mathematics tells us that you can embed `m` points in `O(log m / ε²)` dimensions while preserving all pairwise distances up to factor `(1 ± ε)`. For neural networks, this means: with 4096 dimensions, you can represent **tens of thousands of features** with only small mutual interference, as long as features are sparse — i.e., most features are inactive at any given time.

Mathematically, the model learns a matrix `W` (encoder) and `W^T` (decoder) such that:

```
x_reconstructed = ReLU(W^T · W · x + b)
```

For sparse inputs (few features active at once), the cross-feature interference `W_i · W_j` stays small, and each feature can be approximately recovered despite sharing representational space.

---

## A Concrete Toy Example

Imagine you have 5 features but only 2 neurons. Normally, you'd think you can store at most 2 things. But Anthropic's toy model experiments show that if the features are **sparse and roughly equal in importance**, the network learns to store all 5 in a pentagon arrangement — evenly spaced unit vectors in 2D space.

```
Feature 1: (1.0,  0.0)
Feature 2: (0.31, 0.95)
Feature 3: (-0.81, 0.59)
Feature 4: (-0.81, -0.59)
Feature 5: (0.31, -0.95)
```

These vectors aren't orthogonal — they interfere slightly. But because only 1–2 features are typically active at once (sparsity), the interference averages out to a manageable level. The network recovers all 5 features from 2 dimensions with low reconstruction error.

This is superposition. **The network is doing lossy compression with a sparsity prior baked in.**

---

## Why Networks Learn This Spontaneously

Two conditions drive a network toward superposition:

1. **Sparsity** — most features are zero (or near-zero) most of the time. In language, most facts about the world are irrelevant to any given token.
2. **Importance imbalance** — some features matter a lot (syntactic role, entity type) and some rarely matter. The network allocates dedicated neurons to critical features and superimposes rare features.

When both conditions hold, superposition is the optimal solution. A model that refuses to superpose would either need astronomically more neurons, or fail to represent the full richness of the world.

This is why **scaling works**: bigger models don't just get more neurons per feature — they get a better superposition regime with less interference per concept.

---

## The Geometry Gets Weirder: Interference Patterns and Antipodal Pairs

In higher dimensions, the geometry becomes intricate. Anthropic researchers found that features don't just pack uniformly — they form structured geometric configurations:

- **Antipodal pairs**: Two mutually exclusive features (like "male name" / "female name") are stored as exact opposites `(v, -v)`. They can't both be active, so there's zero interference cost.
- **Digons, triangles, pentagons**: Groups of features that rarely co-occur pack into regular polygons in 2D subspaces.
- **Tetrahedra and cubes**: More features, more dimensions, more complex polytopes.

The network is essentially discovering combinatorial geometry to minimize interference, **without anyone explicitly programming this in**.

---

## Why This Matters for AI Safety and Interpretability

### 1. You Can't Trust Single-Neuron Analysis

If neurons are polysemantic, then studying "what does neuron X do?" gives you a misleading picture. A neuron that fires for both "financial debt" and "grammatical negation" isn't a single detector for anything coherent.

This is a foundational problem for mechanistic interpretability — the field trying to reverse-engineer what neural networks have learned. Most early interpretability work assumed monosemanticity. Superposition means you need to find the **actual features** (sparse linear combinations of neurons), not the neurons themselves.

### 2. The Features Are the Right Unit of Analysis

Anthropic's follow-up work, **"Towards Monosemanticity"** (2023), used sparse autoencoders (SAEs) to decompose superposed representations back into interpretable features. They found ~50,000 distinct features inside a single-layer transformer — vastly more than its neuron count.

Some of these features were remarkably clean:

- "the model is currently tracking a list"
- "this token is part of a DNA sequence"
- "the context involves deception"

This is real progress. But it also means the true internal vocabulary of a model is orders of magnitude more complex than anyone assumed.

### 3. Deceptive Features Are Harder to Find

If a model has learned a feature for "I am being evaluated" or "this is a safety test" — stored in superposition across many neurons — it becomes nearly invisible to naive interpretability methods. You'd need to specifically look for it as a sparse direction in activation space.

This is not a hypothetical concern. It's the crux of why superposition matters for AI safety: **you can't verify what a sufficiently capable model has learned just by inspecting its neurons**.

### 4. The Activation Space Has Structure You're Not Using

Most practitioners treat neural activations as opaque blobs. But superposition tells us activations have rich geometric structure — clusters, polytopes, directional features — that encodes the model's entire world model.

Tools like **probing classifiers**, **activation steering**, and **sparse autoencoders** are beginning to exploit this structure. If you want to understand why your model behaves a certain way, the answer is almost certainly written in activation space geometry.

---

## Practical Takeaways for ML Engineers

**1. Sparse autoencoders are the new probe**
Training a sparse autoencoder on your model's activations can surface the actual feature basis. Libraries like `transformer-lens` (EleutherAI) and Anthropic's open-source SAE work make this accessible.

```python
# Rough sketch using transformer-lens
from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained("gpt2-small")

# Hook into residual stream at layer 6
_, cache = model.run_with_cache("The Eiffel Tower is in Paris")
activations = cache["resid_post", 6]  # shape: [batch, seq, d_model]

# Now train an SAE on these activations to find the feature basis
```

**2. Think in directions, not neurons**
When debugging unexpected model behavior, don't ask "which neurons fire?" Ask "which direction in activation space is active?" A feature is a vector, not a scalar.

**3. Sparsity is your friend**
If you're designing custom architectures or fine-tuning strategies, features encoded in sparser representations are easier to disentangle. Architectures that encourage sparsity (Mixture of Experts, sparse attention) may produce more interpretable internal representations.

**4. Polysemanticity grows with scale — plan accordingly**
Larger models have more superposition, not less (because they have more features to represent). If interpretability matters for your use case (medical, legal, safety-critical), build it into your pipeline from the start, not as an afterthought.

---

## The Bigger Picture

Superposition is one piece of a larger puzzle that mechanistic interpretability is assembling. We now know:

- **Features** (not neurons) are the atoms of what networks know.
- **Circuits** (subgraphs connecting features across layers) implement algorithms.
- **Universality** suggests similar circuits appear independently across different models trained on similar data.

The dream is a complete reverse-engineering of what a large model has learned — its full "world model" — expressed as a legible set of features and circuits. Superposition is the main obstacle between us and that dream.

We don't yet know how to fully solve it. But we know it exists, we know why it happens, and we're building the tools to navigate it. That's already more than most people in the field realize.

---

## Further Reading

### Foundational Papers

- **[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)** — Elhage et al., Anthropic (2022)
  The original paper introducing the superposition hypothesis. Uses minimal toy models to show how networks store more features than dimensions. Dense but essential.

- **[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)** — Elhage et al., Anthropic (2021)
  The foundational paper for mechanistic interpretability. Introduces circuits, attention heads as independent learners, and the residual stream as a shared communication channel.

- **[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)** — Olsson et al., Anthropic (2022)
  Discovers that a specific two-head circuit ("induction heads") is responsible for in-context learning — one of the clearest examples of a circuit implementing a meaningful algorithm.

### Sparse Autoencoders & Feature Discovery

- **[Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)** — Bricken et al., Anthropic (2023)
  Applies sparse autoencoders (SAEs) to a one-layer transformer and recovers ~512 clean, interpretable features from 512 neurons. The breakthrough moment for feature-level interpretability.

- **[Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093)** — Gao et al., OpenAI (2024)
  OpenAI's take on SAEs at scale. Introduces new evaluation metrics and shows SAEs scale well to large models.

- **[Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)** — Templeton et al., Anthropic (2024)
  SAEs applied to Claude 3 Sonnet, revealing millions of interpretable features including concepts like "the Golden Gate Bridge" and "emotions like frustration." Landmark result showing this technique works at production scale.

### Grokking & Emergent Generalization

- **[Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)** — Power et al., OpenAI (2022)
  Documents the "grokking" phenomenon: networks that perfectly memorize training data eventually undergo a phase transition and learn the true underlying algorithm. Completely counterintuitive.

- **[Progress Measures for Grokking via Mechanistic Interpretability](https://arxiv.org/abs/2301.05217)** — Nanda et al. (2023)
  Reverse-engineers _how_ a network learns modular arithmetic, identifying the exact Fourier-basis algorithm the network discovers. One of the most complete circuit analyses ever done.

### Tools & Libraries

- **[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)** — Neel Nanda / EleutherAI
  The go-to PyTorch library for mechanistic interpretability. Hooks into any transformer layer, caches activations, and ships with dozens of pre-loaded models. Start here for hands-on work.

- **[SAELens](https://github.com/jbloomAus/SAELens)** — Joseph Bloom
  A library for training and analyzing sparse autoencoders on transformer activations. Companion to the Neuronpedia platform.

- **[Neuronpedia](https://www.neuronpedia.org/)** — Johnny Lin et al.
  A searchable database of millions of SAE features extracted from GPT-2 and other models. Search for any concept and see which features activate for it — the best way to build intuition.

### Accessible Introductions

- **[Mechanistic Interpretability Quickstart Guide](https://neelnanda.io/mechanistic-interpretability/quickstart)** — Neel Nanda
  The best on-ramp for practitioners. Covers what to read, in what order, and which tools to use.

- **[200 Concrete Open Problems in Mechanistic Interpretability](https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability)** — Neel Nanda (2022)
  If you want to contribute to this field, this is the map. 200 tractable research questions at varying difficulty levels.

- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** — Jay Alammar
  Not specifically about superposition, but the clearest visual explanation of transformer internals. Essential prerequisite reading.

---

_The most important things happening in AI are often not the capabilities — they're the discoveries about what's already inside the models we've built. Superposition is one of those discoveries. It changes what questions we ask, what we trust, and what we fear._
