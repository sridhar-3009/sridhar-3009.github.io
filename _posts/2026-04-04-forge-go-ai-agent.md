---
layout: post
title: "I Rebuilt Claude Code in Go — Works Offline with Ollama, No API Key Required"
date: 2026-04-04
description: After the Claude Code source leak revealed how agentic CLI tools are built, I reverse-engineered the core loop and rebuilt it from scratch in Go — with support for both Anthropic's API and fully offline Ollama models. Here's what I learned.
tags: [Go, CLI, Ollama, Claude, AI Tools, Open Source, Developer Tools, LLM]
categories: systems
giscus_comments: false
related_posts: false
---

# I Rebuilt Claude Code in Go — Works Offline with Ollama, No API Key Required

When the Claude Code source leaked, the AI developer community got an unexpected gift: a detailed look at how a production-grade agentic coding assistant is actually built under the hood. Not the marketing pitch — the real scaffolding. The tool loop. The prompt structure. The stream-parse-execute cycle that makes it feel like a colleague rather than a chatbot.

I read every line. Then I built my own version.

**[Forge](https://github.com/sai-sridhar-repo-07/tarra-claw)** is an open-source AI coding agent CLI written in Go. It does what Claude Code does — code review, commit generation, interactive AI chat, file editing, bash execution — but it's a single 16MB binary that starts in under 50ms, runs fully offline via Ollama, and costs nothing if you don't want to pay for an API.

This post is about what I learned building it and why Go was the right choice.

---

## What the Leak Revealed

The core of Claude Code — and every serious agentic coding tool — is deceptively simple. It's a loop:

```
1. Send context + tools to the model
2. Stream the response
3. If the model calls a tool → execute it, append result
4. Repeat until the model stops calling tools
5. Return final response
```

That's it. The magic is entirely in:

- **What tools you give the model** (bash, file read/write, grep, glob, web fetch...)
- **How you manage context** (what you include, what you truncate, how you represent tool results)
- **The quality of your system prompt**

The leak confirmed what interpretability researchers already knew: the "intelligence" in these tools is overwhelmingly in the underlying model. The scaffolding is plumbing. Good plumbing matters — but it's learnable, replicable, and in Go, very fast.

---

## Why Go?

Claude Code is TypeScript/Node. Most AI tooling is Python. I chose Go for three reasons:

**1. Single binary deployment**
`go build` produces one self-contained executable. No `node_modules`, no virtualenv, no runtime dependency hell. You download Forge, you run Forge. That's the entire installation.

**2. Startup time**
Node.js cold-start on a typical dev machine: 200–400ms. Go: under 50ms. When you're running `forge review --staged` as a pre-commit hook dozens of times a day, that difference compounds into real friction.

**3. Goroutines for streaming**
AI responses stream as server-sent events. Go's concurrency model — goroutines + channels — makes reading a stream while processing tool calls genuinely elegant. No async/await callback soup.

---

## The Architecture

Forge has four main layers:

### 1. Provider Interface

```go
type Provider interface {
    Chat(ctx context.Context, messages []Message, tools []Tool) (<-chan Event, error)
    Models(ctx context.Context) ([]Model, error)
}
```

Both `AnthropicProvider` and `OllamaProvider` implement this. Swap the provider, everything else stays the same. This is what makes offline mode work — Ollama speaks a compatible tool-use format, so the agentic engine doesn't know or care which backend it's talking to.

### 2. The Agentic Engine

The send-stream-tools-repeat loop from above, implemented cleanly:

{% raw %}

```go
func (a *Agent) Run(ctx context.Context, task string) error {
    messages := []Message{{Role: "user", Content: task}}

    for {
        events, err := a.provider.Chat(ctx, messages, a.tools)
        // ... stream events

        if len(toolCalls) == 0 {
            break // model is done
        }

        for _, call := range toolCalls {
            result := a.executeTool(ctx, call)
            messages = append(messages, toolResultMessage(call.ID, result))
        }
    }
    return nil
}
```

{% endraw %}

### 3. The 16 Built-in Tools

| Tool         | What it does                                    |
| ------------ | ----------------------------------------------- |
| `bash`       | Execute shell commands                          |
| `read_file`  | Read file contents                              |
| `write_file` | Create or overwrite files                       |
| `edit_file`  | Targeted string replacement (preserves context) |
| `grep`       | Regex search across files                       |
| `glob`       | File pattern matching                           |
| `web_fetch`  | Fetch and summarize web pages                   |
| `git_diff`   | Get staged/unstaged diffs                       |
| `list_dir`   | Directory listing                               |
| `todo_write` | Manage task lists                               |
| + 6 more     | Notebook, task tracking, etc.                   |

This is essentially the same tool surface as Claude Code. The model uses them identically because they're described the same way in the system prompt.

### 4. TUI with Bubble Tea

The interactive mode uses [Bubble Tea](https://github.com/charmbracelet/bubbletea) — a Go framework for terminal UIs following the Elm architecture. It handles the chat interface, streaming output rendering, and keyboard shortcuts cleanly without any JavaScript-framework complexity.

---

## The Two Things That Actually Matter

Building this taught me what the scaffold does and doesn't do.

### Context management is everything

The biggest practical challenge is context window management. As a conversation grows, you hit the model's token limit. The naive fix — truncate old messages — breaks tool-call continuity. The right approach is selective summarization: keep tool results recent, summarize earlier conversation turns, never truncate mid-tool-call.

Getting this wrong means the model loses track of what files it's already edited. Getting it right means long multi-file refactors work coherently.

### System prompt quality determines behavior

I spent more time on the system prompt than on any other single component. The model's "personality" as a coding agent — how it plans before acting, whether it reads files before editing them, how it handles errors — is entirely encoded in that prompt. The code is just infrastructure for delivering it.

---

## Quick Start (60 seconds)

### With Anthropic API

```bash
# Download binary (macOS Apple Silicon)
curl -L https://github.com/sai-sridhar-repo-07/tarra-claw/releases/latest/download/forge-darwin-arm64 -o forge
chmod +x forge && sudo mv forge /usr/local/bin/

# Set your key
export ANTHROPIC_API_KEY=sk-ant-...

# Review your staged changes before committing
forge review --staged
```

### Fully Offline with Ollama (free)

```bash
# Install Ollama
brew install ollama && ollama serve &

# Pull a coding model (4GB, worth it)
ollama pull qwen2.5-coder:7b

# Run Forge against it — no API key needed
FORGE_PROVIDER=ollama forge review --staged
```

### Model Recommendations for Offline Use

| Model               | Size | Best for                           |
| ------------------- | ---- | ---------------------------------- |
| `llama3.2`          | 2GB  | General chat, quick tasks          |
| `qwen2.5-coder:7b`  | 4GB  | Code review, editing (recommended) |
| `deepseek-coder-v2` | 8GB  | Large codebases, complex refactors |

---

## What Forge Does That Claude Code Doesn't

**`forge review --staged`** — Dedicated code review command. Analyzes your git diff before you commit and flags real issues: divide-by-zero bugs, unhandled exceptions, SQL injection risks, logic errors. Not a style linter. Actual bug detection.

**`forge commit`** — Generates a [conventional commit](https://www.conventionalcommits.org/) message from your staged diff. Reads your actual changes, not a template. Saves the 30 seconds of message-writing on every commit.

**`forge review --branch main`** — Review the full diff between your branch and main. Useful before opening a PR.

**Zero lock-in** — Your data never leaves your machine in Ollama mode. No telemetry, no accounts, MIT licensed.

---

## Optional Config

Drop a `~/.config/forge/config.yaml` if you want persistent settings:

```yaml
provider: ollama # or "anthropic"
ollama_model: qwen2.5-coder:7b
max_tokens: 8096
auto_approve: false # set true to skip tool-execution confirmations
```

Or use environment variables for one-off overrides:

```bash
FORGE_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... forge run "refactor the auth module to use JWT"
```

---

## What I'd Build Next

A few things I deliberately left out of v1 that are on the roadmap:

- **MCP server support** — The Model Context Protocol lets tools be served over a network. Forge could consume MCP servers the same way Claude Code does, giving it extensible tool sets without recompiling.
- **Project-level memory** — Persisting a summary of the codebase between sessions so the model doesn't start cold every time.
- **Parallel tool execution** — Right now tools execute sequentially. Most are I/O-bound; running them in parallel goroutines would speed up multi-file tasks significantly.

---

## The Bigger Lesson

The leak wasn't really about Claude Code. It was a reminder that agentic AI tools — however impressive they feel — are thin scaffolding around a powerful model. The scaffolding matters, but it's not magic. It's software. It can be studied, replicated, and improved.

If you're a developer who's been treating these tools as black boxes, I'd encourage you to build one. Even a minimal implementation clarifies how they work in a way that no amount of using them ever will.

Forge is MIT licensed. Read it, fork it, break it, make it better.

**[github.com/sai-sridhar-repo-07/tarra-claw](https://github.com/sai-sridhar-repo-07/tarra-claw)**

---

## References & Further Reading

### Core Technologies Used

- **[Bubble Tea](https://github.com/charmbracelet/bubbletea)** — Elm-architecture TUI framework for Go. The cleanest way to build terminal interfaces.
- **[Cobra](https://github.com/spf13/cobra)** — The standard Go CLI framework. Powers `kubectl`, `hugo`, `gh`, and now Forge.
- **[Anthropic Go SDK](https://github.com/anthropics/anthropic-sdk-go)** — Official Go client for the Claude API with streaming support.
- **[Ollama](https://ollama.com/)** — Run LLMs locally on your machine. One command install, dozens of models.

### Understanding Agentic AI Systems

- **[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)** — Anthropic's guide on when to use agents vs. simpler pipelines. Essential reading before designing any agentic system.
- **[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)** — The paper that formalized the think-act-observe loop that underpins all modern AI agents.
- **[Tool Use in Claude](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)** — Anthropic's official documentation on how tool/function calling works at the API level.

### Go for AI Tooling

- **[Effective Go](https://go.dev/doc/effective_go)** — The canonical guide to writing idiomatic Go. Read this before writing your first production Go tool.
- **[Charm CLI Tools](https://charm.sh/)** — The ecosystem behind Bubble Tea. Lipgloss for styling, Glamour for markdown rendering, Huh for forms. Everything you need for beautiful terminal UIs.
- **[Server-Sent Events in Go](https://pkg.go.dev/net/http)** — How to consume streaming HTTP responses (what all LLM APIs use) idiomatically in Go.

### Conventional Commits & Developer Workflow

- **[Conventional Commits Specification](https://www.conventionalcommits.org/en/v1.0.0/)** — The spec Forge uses for `forge commit` output. Enables automated changelogs and semantic versioning.
- **[pre-commit](https://pre-commit.com/)** — Framework for running hooks before commits. Combine with `forge review --staged` for automated AI review on every commit.
