---
layout: post
title: "Byte-Latent Transformers: Training AI Without a Tokenizer"
date: 2026-05-14
description: Every LLM you've ever used runs on a tokenizer that chops text into chunks before the model sees a single character. Meta's Byte-Latent Transformer throws that away entirely — and matches LLaMA 3 performance. Here's how.
tags: [BLT, Tokenization, Transformers, Meta AI, Byte Models, LLMs, Architecture]
categories: deep-learning
giscus_comments: false
related_posts: false
---

# Byte-Latent Transformers: Training AI Without a Tokenizer

Every language model you've ever used — GPT-4, Claude, Gemini, LLaMA — starts with the same preprocessing step before a single parameter is touched.

A **tokenizer** runs over your input and converts it into a sequence of integer IDs. `"Hello world"` becomes something like `[9906, 1917]`. The model never sees characters. It sees token IDs drawn from a fixed vocabulary, usually 32K to 128K entries, built offline by running BPE or SentencePiece on a large corpus.

This has been the unquestioned default for five years.

In late 2024, Meta AI published **"Byte Latent Transformer: Patches Scale Better Than Tokens"** — and questioned it.

The core claim: you can train a competitive language model directly on raw bytes, no tokenizer required, and at the same compute budget it matches or beats LLaMA 3 on most benchmarks. On tasks that expose tokenization's weaknesses — character manipulation, rare languages, structured data — it wins clearly.

This post covers why tokenization has always been a compromise, how BLT replaces it, and why this architecture matters.

---

## TL;DR

- Tokenizers are a **lossy, brittle preprocessing step** baked into every modern LLM
- BLT processes **raw bytes** — no vocabulary, no OOV, no tokenization artifacts
- Groups bytes into **variable-length patches** based on information content (entropy)
- Architecture: Local Encoder → Latent Transformer → Local Decoder
- **Matches LLaMA 3** at equal FLOP budget; beats it on character-level tasks
- Especially strong on **multilingual text, code, structured data, and novel formats**
- Patches scale better than tokens as model size increases

---

## The Tokenizer Problem

Before explaining what BLT does, it's worth being specific about what tokenizers get wrong.

### Problem 1: The Vocabulary Is Fixed and Arbitrary

BPE (Byte-Pair Encoding) builds a vocabulary by iteratively merging the most frequent character pairs in a training corpus. The result is deeply English-centric. Common English words get single tokens. Rare words, names, and most non-Latin scripts get shredded into fragments.

The word `"unfortunately"` → 1 token.
The Turkish word `"değiştiremeyebilirsiniz"` → 12+ tokens.

Same information density, wildly different token cost. Multilingual models pay a massive efficiency penalty on non-English text simply because of vocabulary construction choices made before training started.

### Problem 2: Tokenization Creates Invisible Bugs

Because models operate on token IDs rather than characters, they have no direct access to spelling. This causes failures that look baffling from the outside:

- `9.11 > 9.9` — GPT-4 originally said false. The numbers are tokenized as whole units; the model sees `9` and `.11` vs `9` and `.9` without clean positional structure.
- Reverse a string → often wrong, because reversing happens at the character level but the model thinks in tokens.
- Count the letter `r` in `"strawberry"` → famously hard for tokenized models, trivial for humans.
- Rhyming, anagram detection, wordplay → systematically poor.

These aren't reasoning failures. They're tokenization artifacts.

### Problem 3: Tokenization Is Brittle at Boundaries

Adding a space changes the token. Capitalization changes the token. A URL with an unusual structure might produce a token sequence no model has seen during training. Code with uncommon variable names gets fragmented unpredictably.

Models can recover from this because they have enough capacity to memorize patterns — but they're burning parameters on compensating for preprocessing noise.

### Problem 4: The Vocabulary Can't Adapt

If a new domain emerges after the tokenizer is built — a new programming language, a new file format, a new script — the model handles it poorly. The vocabulary is frozen. There's no mechanism to learn better representations for new patterns without retraining the tokenizer and rebuilding the vocabulary from scratch.

---

## What Bytes Give You

At the other extreme: raw bytes.

Every piece of digital information — text in any language, code, HTML, JSON, images, audio — can be represented as a sequence of bytes. A byte is a number from 0 to 255. There are exactly 256 possible values.

A byte-level model has a vocabulary of 256. It never needs to be retrained. It never has an OOV (out-of-vocabulary) problem. Every string in any language on any format is representable without special handling.

The catch: raw bytes produce very long sequences. `"Hello world"` is 11 bytes. A 10,000-word document is ~60,000 bytes. Transformers scale quadratically with sequence length — so naive byte-level transformers are expensive, slow, and don't scale well.

This is why the field defaulted to tokenization in the first place. Tokens compress the sequence. Shorter sequence → cheaper attention → more compute-efficient training.

BLT's contribution is solving the byte-length problem without reintroducing a fixed vocabulary.

---

## The BLT Architecture

BLT replaces the tokenizer with a learned, dynamic **patching** mechanism. Instead of a fixed vocabulary, it groups bytes into variable-length **patches** based on how predictable the bytes are.

The full model has three components:

```text
Raw bytes
    ↓
Local Encoder  (lightweight byte-level transformer)
    ↓
Patch representations  (dynamic, variable-length groups)
    ↓
Latent Transformer  (large, expensive — operates on patches)
    ↓
Patch predictions
    ↓
Local Decoder  (reconstructs byte-level output from patches)
    ↓
Output bytes
```

### Component 1: Local Encoder

A small, lightweight transformer that reads raw bytes and produces byte-level representations. It doesn't try to understand the whole document — it just builds local context around each byte.

The output feeds into the patching step.

### Component 2: Entropy-Based Dynamic Patching

This is the core idea. Instead of splitting text at fixed boundaries (like BPE does), BLT groups bytes into patches based on **next-byte entropy** — a measure of how surprising the next byte is.

High entropy = the model is uncertain what comes next. This byte starts something complex. Start a new patch here.

Low entropy = the next byte is predictable. This byte is part of a familiar pattern. Extend the current patch.

```text
Entropy-based patching example:

Input:   " t h e   q u i c k   b r o w n   f o x "
Entropy: [H  L  L  H  L  L  L  L  H  L  L  L  L  H  L  L  L]
Patches: [_the] [_quick] [_brown] [_fox]
```

Common words and phrases get compressed into single patches. Rare words, code snippets, unusual names, and multilingual text get finer-grained patches — more attention per byte, exactly where it's needed.

This is the key insight: **allocate compute proportional to information content**. Predictable bytes get bundled cheaply. Surprising bytes get resolved carefully.

The patch boundaries are dynamic — they change based on the actual input. A technical paper about nuclear physics gets different patches than a children's story, even at the same byte length.

### Component 3: Latent Transformer

The main, expensive transformer. It operates on **patch representations**, not individual bytes. Because patches compress the sequence — typically by a factor of 4–8× compared to raw bytes — the latent transformer sees much shorter sequences.

This is where the bulk of the model's parameters live and where the "understanding" happens. It uses standard transformer architecture — multi-head attention, feed-forward layers — but on patch-level units.

Critically, patch representations carry context from the Local Encoder via **cross-attention**. The latent transformer can look back at individual byte-level details when it needs to.

### Component 4: Local Decoder

Reconstructs byte-level output from the patch-level predictions. Given a predicted patch, it generates the actual bytes that form the output — one byte at a time, conditioned on both the patch prediction and previously generated bytes.

The decoder is also lightweight. Most of the compute stays in the Latent Transformer.

---

## Why Patches Scale Better Than Tokens

The paper's title makes a specific claim: **patches scale better than tokens**.

Here's the intuition.

As models get larger, they get better at predicting common patterns — frequent words, standard syntax, typical phrases. With fixed tokenization, every token has the same representation cost regardless of how predictable it is. A token for `"the"` takes the same slot in the attention matrix as a token for `"Schrödinger"`.

With entropy-based patching, as the model improves:
- More bytes become predictable → they get compressed into larger patches
- The effective sequence length decreases
- The same latent transformer capacity gets focused on harder problems
- Scaling the model → automatically better compression → more efficient use of compute

This creates a virtuous loop that fixed tokenization can't replicate. Tokenization is static — it doesn't adapt to what the model has learned. BLT's patching is dynamic — it improves with the model.

The paper shows this empirically: at the same compute budget, BLT matches LLaMA 3. At larger scales, the efficiency gap favors BLT more.

---

## Where BLT Clearly Wins

Some tasks are structurally impossible to do well with tokenized models. BLT handles all of them naturally.

### Character-Level Manipulation

Reverse a string, count specific characters, detect palindromes, anagram matching. BLT operates at the byte level — these tasks are as natural for it as scanning a character array is for you.

### Multilingual Text

No vocabulary bias. Turkish, Arabic, Hindi, Chinese, code-switching mid-sentence — all represented at the same byte-level granularity. No 10× token penalty for non-Latin scripts.

### Structured Formats

JSON, CSV, XML, HTML, source code with unusual variable names — all handled without tokenization artifacts. The entropy-based patcher naturally segments these formats at structurally meaningful boundaries.

### Robustness to Typos and Noise

A tokenizer turns a misspelled word into a completely different token sequence. BLT sees character-level corruption as a byte-level signal and can reason about it directly. This matters for real-world noisy input.

### Novel Formats and Future-Proofing

A new programming language, a new markup format, a new domain — BLT handles it without retraining the vocabulary. There's no concept of OOV. The 256-byte vocabulary is universal.

---

## The Numbers

| Benchmark | LLaMA 3 (tokens) | BLT (bytes) |
|---|---|---|
| FLOP budget | Matched | Matched |
| Overall performance | Baseline | Matches or beats |
| Character tasks | Weak | Significantly better |
| Multilingual | Penalized by vocab | Equal cost per byte |
| Robustness | Token-brittle | Byte-robust |
| Vocabulary size | 128K | 256 (universal) |
| OOV handling | Fragmentation | None |

At equal compute: BLT is competitive. At tasks that expose tokenization's weaknesses: BLT wins clearly.

---

## What BLT Doesn't Solve

BLT is a compelling paper but not a solved problem.

**Inference speed is still a challenge.** The Local Encoder and Decoder add overhead at inference time that pure token-based models don't have. The latent transformer may be cheaper due to shorter sequences, but the total wall-clock cost per output byte needs careful engineering.

**The compression factor varies.** On highly predictable text — boilerplate, repetitive formats — patches get large and the latent transformer gets efficient. On information-dense text like dense code or technical writing, patches stay small and the efficiency advantage shrinks.

**Training stability at scale is unproven.** The BLT paper demonstrates results up to a certain model size. Whether the architecture scales to 70B+ parameters with the same efficiency properties is an open question.

**The field hasn't adopted it yet.** LLaMA 4, GPT-5, Gemini 2 all still use tokenizers. BLT is research. The engineering investment to replace tokenization infrastructure at production scale is non-trivial.

---

## Why This Matters

The reason BLT is interesting isn't just the benchmarks.

It's that tokenization has been treated as an unquestioned given — an implementation detail inherited from the NLP of the early 2010s, carried forward because it worked well enough and changing it was hard.

BLT demonstrates that the "work well enough" part was covering for real weaknesses, and that a principled alternative is achievable. It shows that you can beat a carefully tuned tokenized baseline without a fixed vocabulary, using only 256 byte values and a smarter way of grouping them.

That's a meaningful result even if BLT itself never ships in a production model.

The broader lesson: **the preprocessing stack is not fixed**. Tokenization was an engineering convenience that became an invisible constraint. BLT makes the constraint visible — and shows a path around it.

The next generation of models might not start by running your input through a frozen vocabulary lookup. They might start by reading bytes — the way your CPU always has.

---

## Further Reading

- [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871) — Meta AI, 2024
- [BPE tokenization explained — Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter6/5)
- [MegaByte: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185) — earlier byte-level work from Meta
- [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://arxiv.org/abs/2105.13626) — Google's earlier byte model
