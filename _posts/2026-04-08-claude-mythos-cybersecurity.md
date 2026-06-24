---
layout: post
title: "Claude Mythos: Anthropic's Most Powerful Model — Built for Cybersecurity, Not for Everyone"
date: 2026-04-08
description: Anthropic just unveiled Claude Mythos Preview — a frontier model that autonomously finds zero-day vulnerabilities, outperforms every other Claude model on coding benchmarks, and is explicitly not available to the public. Here's everything you need to know about the most capable — and most restricted — AI model Anthropic has ever built.
tags: [Anthropic, Claude, Cybersecurity, LLM, AI Safety, Project Glasswing, Frontier AI]
categories: ai-research
giscus_comments: false
related_posts: false
---

# Claude Mythos: Anthropic's Most Powerful Model — Built for Cybersecurity, Not for Everyone

On April 7, 2026, Anthropic announced something unusual: a new frontier model that they have no intention of making generally available.

**Claude Mythos Preview** is Anthropic's most capable model to date — outperforming Claude Opus 4.6 on every major coding and reasoning benchmark by significant margins. It can autonomously find zero-day vulnerabilities that survived decades of human review and millions of automated security tests. It can chain multiple Linux kernel vulnerabilities for privilege escalation without any human guidance.

And you cannot use it. At least not unless you're one of the ~40+ organizations hand-picked to access it through **Project Glasswing** — a coalition of the largest technology companies in the world, convened specifically to give defenders an advantage over attackers before a model like this falls into the wrong hands.

This post covers what Mythos is, what it can do, how it compares to other Claude models, and what Anthropic's decision to restrict it tells us about the future of frontier AI.

---

## What Is Claude Mythos?

Claude Mythos Preview is a **cybersecurity-specialized frontier model** — Anthropic's most capable AI system ever released (even as a preview), designed from the ground up to assist with defensive security workflows.

The word "preview" is doing real work here. This is not a generally available product. It is a research preview offered to a curated set of organizations under strict access controls, as part of a deliberate strategy to arm defenders before attackers gain access to equivalent capabilities.

It is available via the Claude API, Amazon Bedrock, Google Cloud Vertex AI, and Microsoft Foundry — but only for invited organizations. There is no self-serve sign-up. Access requires application to the **Cyber Verification Program** or membership in Project Glasswing.

**Pricing (for authorized users):** $25 per million input tokens / $125 per million output tokens — roughly 5x the cost of Claude Opus 4.6, reflecting both the model's capability level and the intentionally limited rollout.

---

## Project Glasswing: Why This Exists

To understand Mythos, you first need to understand why Anthropic announced it the way they did.

**Project Glasswing** (announced April 7, 2026) is a coalition that brings together:

> AWS, Anthropic, Apple, Broadcom, Cisco, CrowdStrike, Google, JPMorganChase, the Linux Foundation, Microsoft, NVIDIA, and Palo Alto Networks

The explicit goal: ensure that AI's most powerful security capabilities reach defenders before attackers. Anthropic describes it as "an urgent attempt" — the language is deliberate. They believe a model capable of what Mythos can do is coming regardless of whether Anthropic builds it. The question is who gets it first.

**Commitments made at launch:**

- **$100 million** in Mythos usage credits to Project Glasswing partners
- **$4 million** in usage credits to open-source security organizations
- Active development of cybersecurity safeguards to detect and block the model's most dangerous outputs
- A **Cyber Verification Program** allowing vetted security professionals to access capabilities that would otherwise be blocked

This is not a product launch. It is a coordinated deployment strategy built around a security thesis: defenders need this more urgently than the public needs access to it.

---

## What Mythos Can Actually Do

The headline capabilities are genuinely remarkable — and in some cases, should give you pause.

### Autonomous Zero-Day Vulnerability Discovery

Mythos can find vulnerabilities that human reviewers and automated tools have missed — for years, sometimes decades.

**Documented examples from Anthropic's announcement:**

- **A 27-year-old vulnerability in OpenBSD** — a security-focused operating system that has been under continuous expert review since 1999. Mythos found a flaw that had survived nearly three decades of scrutiny.

- **A 16-year-old flaw in FFmpeg** — the ubiquitous open-source multimedia framework used in virtually every video streaming platform, browser, and media application on the planet. This vulnerability survived **5 million automated test runs** by fuzzing tools specifically designed to find bugs like it.

- **Linux kernel privilege escalation** — Mythos autonomously identified and chained multiple separate vulnerabilities in the Linux kernel to achieve privilege escalation, without any human guidance or steering at any step.

The phrase "entirely autonomously, without any human steering" appears in Anthropic's own description. This is not a model that helps a human security researcher — it is a model that _is_ the security researcher, working independently.

### Agentic Code Execution

Mythos operates as a full agentic system — it doesn't just identify vulnerabilities, it can develop exploits, write and execute code, navigate file systems, and chain tool calls across long autonomous sessions. This puts it in a different capability class from models that primarily generate text for human review.

### Broad Reasoning Superiority

Beyond cybersecurity, Mythos is simply a more capable general reasoner than anything Anthropic has previously released. The benchmark numbers confirm this.

---

## Benchmark Performance

| Benchmark                                 | Mythos Preview | Claude Opus 4.6 | Delta     |
| ----------------------------------------- | -------------- | --------------- | --------- |
| **CyberGym** (Vulnerability Reproduction) | **83.1%**      | 66.6%           | +16.5 pts |
| **SWE-bench Pro** (Real-world coding)     | **77.8%**      | 53.4%           | +24.4 pts |
| **SWE-bench Verified**                    | **93.9%**      | 80.8%           | +13.1 pts |
| **Terminal-Bench 2.0**                    | **82.0%**      | 65.4%           | +16.6 pts |
| **GPQA Diamond** (Graduate reasoning)     | **94.6%**      | 91.3%           | +3.3 pts  |

A few things stand out:

**SWE-bench Pro is the most striking.** This benchmark uses real-world GitHub issues from production codebases — not curated problems. Mythos resolves 77.8% of them. Opus 4.6 resolves 53.4%. That 24-point gap is enormous in practical terms — it means Mythos solves roughly half again as many real engineering problems as the best publicly available model.

**GPQA Diamond shows the narrowest gap.** Graduate-level science reasoning (physics, chemistry, biology) is the one domain where Mythos' lead over Opus 4.6 shrinks to ~3 points. Both models are operating near the ceiling of this benchmark. The differentiation is overwhelmingly in agentic, tool-use, and code-execution tasks.

**CyberGym is a new benchmark.** It measures vulnerability reproduction — given a description of a known CVE, can the model reproduce the exploit? At 83.1%, Mythos is far ahead of anything previously available. This is not a benchmark most models even attempt.

---

## How It Compares to the Full Claude Model Family

Claude currently has three publicly available tiers, plus Mythos as a restricted preview:

| Model                     | Context     | Pricing (in/out per MTok) | Best For                           |
| ------------------------- | ----------- | ------------------------- | ---------------------------------- |
| **Claude Haiku 4.5**      | 200k tokens | $1 / $5                   | Speed, high-volume tasks           |
| **Claude Sonnet 4.6**     | 1M tokens   | $3 / $15                  | Balanced speed + intelligence      |
| **Claude Opus 4.6**       | 1M tokens   | $5 / $25                  | Complex reasoning, agentic tasks   |
| **Claude Mythos Preview** | TBD         | $25 / $125                | Cybersecurity, frontier capability |

Mythos sits at 5x the price of Opus 4.6 — and based on the benchmark numbers, earns it for the right workloads.

---

## The Safety Calculus: Why Anthropic Restricted It

This is the part that makes Mythos genuinely interesting from an AI development perspective.

Anthropic has built a model that is meaningfully more capable than anything publicly available. Rather than releasing it broadly — which would maximize revenue and reach — they've chosen a controlled deployment specifically because of what the model can do.

Their reasoning, as stated:

1. **Offense/defense asymmetry**: A model that can find zero-days is dangerous in attacker hands and invaluable in defender hands. Releasing to defenders first, under access controls, tries to tilt that asymmetry.

2. **Safeguards are still in development**: Anthropic explicitly states they are "developing cybersecurity (and other) safeguards that detect and block the model's most dangerous outputs" — and that these safeguards will be incorporated into future public Claude Opus releases. Mythos Preview ships without the full safety stack because the safety stack isn't ready yet.

3. **Verification before access**: The Cyber Verification Program requires applicants to demonstrate legitimate security work. This isn't perfect — verification can fail — but it's a meaningful friction point that general API access wouldn't provide.

This approach is a departure from the default "release and iterate" model that most AI labs follow. It raises genuinely hard questions: who decides which organizations are legitimate defenders? What happens when a Glasswing partner is compromised? How long before equivalent capabilities are available elsewhere?

Anthropic isn't pretending these questions have clean answers. The framing of "urgent attempt" suggests they believe the window for this strategy is limited.

---

## What This Means for the Industry

**For security teams:** The gap between AI-assisted and unassisted security work is about to widen dramatically. Teams with access to Mythos-class models will find vulnerabilities faster, assess attack surfaces more completely, and respond to incidents with better intelligence than teams without. This is not a marginal advantage.

**For open source:** The $4M in credits to open-source security organizations is significant. Projects like OpenBSD, the Linux kernel, curl, OpenSSL — the infrastructure everything else runs on — will have access to a model that has already demonstrated it can find bugs their own maintainers missed.

**For AI development norms:** Project Glasswing is an experiment in whether a controlled, coalition-based deployment can work for a dual-use capability. If it does — if defenders meaningfully get ahead of attackers because of it — it will influence how future frontier models with dangerous capabilities are released. If it fails, it will inform what stronger interventions might look like.

**For developers:** Mythos will eventually influence what's in public Claude models. Anthropic's statement that the cybersecurity safeguards will be incorporated into future Opus releases suggests a pipeline: build capability at the frontier, develop safety measures, then push both into general availability. The public models you use today are yesterday's frontier models with better guardrails.

---

## How to Get Access (If You Qualify)

There are two paths:

**1. Project Glasswing Partnership**
If your organization works on critical infrastructure security and could benefit from Glasswing partnership, the announcement page at [anthropic.com/glasswing](https://www.anthropic.com/glasswing) is the starting point. This is designed for large organizations — the current partners are some of the largest technology and security companies in the world.

**2. Cyber Verification Program**
For individual security researchers and smaller organizations doing legitimate defensive security work, the Cyber Verification Program offers a path to expanded access to Mythos capabilities. This is the relevant path for penetration testers, vulnerability researchers, and security engineers.

---

## What Comes Next

Anthropic has not announced a timeline for general availability — and given the framing, it seems unlikely Mythos Preview will ever be generally available in its current form. The more likely path:

- Mythos Preview continues as an invitation-only model for Glasswing and verified security work
- Cybersecurity safeguards currently in development get incorporated into future Claude Opus releases
- Future Claude models gain some of Mythos' capability at lower price points, with the safety stack that makes them appropriate for general use
- A "Mythos" successor model — more capable still — becomes the next restricted frontier

This mirrors how the rest of the Claude family has evolved: capabilities that required frontier models a year ago are now standard in Sonnet and Haiku. The frontier keeps moving. Mythos is today's frontier.

---

## Further Reading

### Official Anthropic Sources

- **[Project Glasswing — anthropic.com/glasswing](https://www.anthropic.com/glasswing)** — Official announcement, partner list, and access information
- **[Claude Models Overview](https://platform.claude.com/docs/en/docs/about-claude/models/overview)** — Full model comparison table including Mythos Preview details
- **[Anthropic's Core Views on AI Safety](https://www.anthropic.com/research)** — The safety philosophy behind decisions like the Mythos restricted release

### Background on AI in Cybersecurity

- **[NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)** — The standards framework that defensive security work is built on
- **[Common Vulnerabilities and Exposures (CVE)](https://cve.mitre.org/)** — The database of publicly disclosed vulnerabilities. Mythos-class models change how these get found.
- **[SWE-bench Leaderboard](https://www.swebench.com/)** — Live benchmark tracking for AI on real software engineering tasks
- **[GPQA: A Graduate-Level Benchmark](https://arxiv.org/abs/2311.12022)** — The paper behind the GPQA Diamond benchmark used to evaluate Mythos

### Dual-Use AI and Policy

- **[The Dual-Use Dilemma in AI](https://www.brookings.edu/articles/the-dual-use-dilemma/)** — Brookings analysis of AI capabilities that are simultaneously defensive and offensive
- **[Anthropic's Responsible Scaling Policy](https://www.anthropic.com/responsible-scaling-policy)** — The policy framework that governs when and how Anthropic deploys frontier capabilities
- **[AI Safety Levels (ASL)](https://www.anthropic.com/research/core-views-on-ai-safety)** — Anthropic's internal framework for assessing and managing model risk before deployment

---

_Claude Mythos is the clearest signal yet that frontier AI capabilities are outpacing the safety infrastructure needed to deploy them broadly. Anthropic's choice to restrict it rather than release it is either responsible restraint or an elaborate delay — probably both. Either way, the gap between what the most capable AI can do and what the public has access to just got measurably wider._
