---
layout: post
title: "MCP Server Starter: Build Tools AI Agents Can Actually Use"
date: 2026-04-21
description: Most tutorials show you how to call an LLM. This one shows you how to build the server on the other side — the tool layer that AI agents call into. MCP in 30 lines of Python.
tags: [MCP, FastAPI, Python, AI Agents, Tools, LLMs]
categories: ai-engineering
giscus_comments: false
related_posts: false
---

# MCP Server Starter: Build Tools AI Agents Can Actually Use

Every demo of an AI agent shows the same thing: the agent calls a tool, gets data back, generates an answer. What nobody shows is **what that tool server actually looks like** and how to build one yourself.

The **Model Context Protocol (MCP)** is the emerging standard for exactly this — a clean interface between AI agents and the tools/APIs they need to interact with. Think of it as the HTTP for agent-tool communication. The agent doesn't need to know how your weather API works internally; it just needs to know what tools exist and how to call them.

This post walks through building an MCP-style server from scratch with Python and FastAPI — the minimal version that teaches you the pattern, plus the upgrade path to production.

---

## The Core Flow

```
AI Agent → MCP Server → Tools / External APIs
```

Three actors:

- **AI Agent** — the LLM runtime (Claude, GPT-4, your own agent loop) that decides which tool to call
- **MCP Server** — your FastAPI app that exposes tools as HTTP endpoints
- **Tools** — functions that do real work: fetch weather, query a database, send a Slack message

The agent doesn't hallucinate tool results because it gets them from your server. Your server doesn't need to understand the agent because it just responds to HTTP requests. Clean separation.

---

## Setup

```bash
pip install fastapi uvicorn requests
```

That's the full dependency list for the starter. FastAPI handles routing and serialization; Uvicorn is the ASGI server.

---

## Project Structure

```bash
mcp-server/
├── main.py
└── README.md
```

Start flat. Add structure when you have more than ~5 tools.

---

## The Server

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/tool/weather")
def weather(city: str):
    return {
        "tool": "weather",
        "city": city,
        "temp": "32°C",
        "status": "Sunny"
    }

@app.get("/tool/users")
def users():
    return {
        "tool": "users",
        "data": [
            {"id": 1, "name": "Rahul"},
            {"id": 2, "name": "Priya"}
        ]
    }

@app.get("/tools")
def tools():
    return {
        "tools": [
            {"name": "weather", "endpoint": "/tool/weather?city=Delhi"},
            {"name": "users", "endpoint": "/tool/users"}
        ]
    }
```

Run it:

```bash
uvicorn main:app --reload
```

Server up at `http://127.0.0.1:8000`.

---

## The Three Endpoints

### Tool Discovery — `/tools`

```bash
GET /tools
```

This is the contract between agent and server. The agent hits this endpoint first, gets back a list of available tools and their call signatures. No hardcoding required — the agent adapts to whatever tools your server advertises.

### Weather Tool — `/tool/weather`

```bash
GET /tool/weather?city=Hyderabad
```

Response:

```json
{
  "tool": "weather",
  "city": "Hyderabad",
  "temp": "32°C",
  "status": "Sunny"
}
```

### Users Tool — `/tool/users`

```bash
GET /tool/users
```

Response:

```json
{
  "tool": "users",
  "data": [
    { "id": 1, "name": "Rahul" },
    { "id": 2, "name": "Priya" }
  ]
}
```

---

## How an AI Agent Uses This

The agent loop looks like this:

1. **Discover** — `GET /tools` → parse available tool names and endpoints
2. **Decide** — LLM sees user query, picks the right tool
3. **Call** — agent hits the chosen endpoint with required parameters
4. **Use** — returned JSON feeds back into the LLM context to generate the final answer

No magic. No framework required. Any agent that can make HTTP requests can use this server — Claude via tool use, GPT-4 function calling, a custom ReAct loop, anything.

---

## Upgrade Path

The starter above is intentionally minimal. Here's where to take it:

**Auth layer** — add API key validation via FastAPI's `Depends`:

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
```

**Real data** — swap the hardcoded returns for actual API calls or database queries:

```python
import httpx

@app.get("/tool/weather")
async def weather(city: str):
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
        )
    data = resp.json()
    return {
        "tool": "weather",
        "city": city,
        "temp": f"{data['main']['temp']}°C",
        "status": data["weather"][0]["description"],
    }
```

**Memory layer** — add a simple in-memory store (or Redis) so agents can persist state across calls:

```python
from collections import defaultdict

memory: dict[str, list] = defaultdict(list)

@app.post("/memory/{session_id}")
def store(session_id: str, payload: dict):
    memory[session_id].append(payload)
    return {"stored": True}

@app.get("/memory/{session_id}")
def recall(session_id: str):
    return {"history": memory[session_id]}
```

**Logging** — every tool call should log `(timestamp, tool_name, params, response_time)`. Non-negotiable in production. Agents fail silently without it.

---

## Production Use Cases

- **Customer support agents** — tools that query your CRM, open tickets, look up order history
- **Internal company assistants** — tools that pull from Confluence, Jira, Slack
- **Sales reporting bots** — tools that hit your data warehouse and return formatted metrics
- **Research agents** — tools that search the web, fetch papers, summarize documents
- **CRM automation** — tools that create contacts, log calls, send follow-up emails

Same pattern in every case: define the tool, expose it as an endpoint, register it in `/tools`. The agent handles the reasoning; your server handles the execution.

---

## Key Takeaways

1. **MCP is just HTTP with a discovery contract** — `/tools` tells the agent what exists; individual endpoints do the work. No exotic protocol needed to start.

2. **Tool discovery is the key endpoint** — it's what makes your server agent-agnostic. Any LLM that can read JSON can use your tools without custom integration.

3. **Keep tools single-responsibility** — one endpoint, one job. Agents get confused by tools that do too much. `/tool/weather` does weather. That's it.

4. **Auth before production** — the starter has no auth. Add API key verification before exposing any real data.

5. **Logging is observability** — you can't debug a multi-agent system without knowing what tool was called, with what params, and what came back.

---

## Further Reading

- **[Model Context Protocol Spec](https://modelcontextprotocol.io/introduction)** — Anthropic's official MCP specification. The standard this starter is inspired by — covers the full protocol including resources, prompts, and sampling beyond just tools.

- **[FastAPI Docs](https://fastapi.tiangolo.com/)** — Best Python API framework for this use case. Dependency injection, auto-generated OpenAPI docs, and async support out of the box.

- **[Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)** — Anthropic's guide on agent patterns. Covers tool use, ReAct loops, and when to use multi-agent vs. single-agent architectures.

- **[LangChain Tools](https://python.langchain.com/docs/concepts/tools/)** — If you want to integrate your MCP server with an existing agent framework, LangChain's tool abstraction maps cleanly onto this pattern.

---

_The agent is the brain. Your MCP server is the hands. Getting the interface between them right — clean endpoints, honest tool descriptions, fast responses — is what separates a demo from a system._
