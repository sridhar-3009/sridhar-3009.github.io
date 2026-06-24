---
layout: post
title: "RAG Beyond Naive Chunking: HyDE, Reranking, ColBERT, and Graph RAG"
date: 2026-04-16
description: Most RAG systems chunk documents, embed them, and call it a day. That's the floor, not the ceiling. Here's what production retrieval actually looks like — HyDE, late interaction, reranking pipelines, and knowledge graphs that reason over structure.
tags: [RAG, Retrieval, HyDE, ColBERT, Reranking, Graph RAG, LLMs, Vector Search, PyTorch]
categories: ai-engineering
giscus_comments: false
related_posts: false
---

# RAG Beyond Naive Chunking: HyDE, Reranking, ColBERT, and Graph RAG

Every company building on LLMs eventually hits the same wall. The model hallucinates facts. It doesn't know about events after its training cutoff. You can't fit your entire knowledge base into a context window. The answer everyone reaches for: **Retrieval-Augmented Generation** (RAG).

The naive version is simple: chunk your documents into paragraphs, embed them with a sentence transformer, store in a vector database, and at query time retrieve the top-K most similar chunks. Stuff them into the prompt. Done.

This works. Up to a point.

Then you discover that your retrieval misses the most relevant documents 30% of the time. That questions about relationships between entities fail completely. That the 500-token chunks you split on periods don't align with how your model reasons. That users asking "what did the CEO say about layoffs last quarter?" get back chunks about layoffs from three years ago because the embeddings didn't capture _recency_.

The gap between naive RAG and production RAG is wide. This post covers the techniques that close it: **HyDE**, **reranking**, **ColBERT**, and **Graph RAG** — with implementations you can actually use.

---

## The Failure Modes of Naive RAG

Before fixing things, name what's broken:

**1. Query-document mismatch**
Queries are short and abstract ("what causes transformer training instability?"). Documents are long and concrete (a paragraph describing gradient norms spiking due to attention softmax saturation). Their embeddings live in different regions of the vector space — not because the content is unrelated, but because the _linguistic form_ is different.

**2. Single-vector bottleneck**
A single embedding vector must compress an entire 512-token chunk. Complex chunks that discuss multiple concepts average their meaning into one point — and retrieval can fail on any of those concepts individually.

**3. Semantic similarity ≠ relevance**
The most semantically similar chunk isn't always the most useful one. A question about a drug's side effects might return chunks that are topically related but don't actually answer the question. Relevance requires understanding the _intent_ of the query, not just its words.

**4. Flat document structure**
Documents have structure — sections, subsections, tables, references between chapters. Chunking destroys this. A chunk about "revenue" in Q3 earnings doesn't know it's three paragraphs after the CFO's statement that preceded it.

Each of these has a targeted fix.

---

## Fix 1: HyDE — Query the Way Documents Answer

**Hypothetical Document Embeddings** (HyDE) attacks the query-document mismatch directly. Instead of embedding the raw query, you ask an LLM to _generate a hypothetical document that would answer the query_ — then embed that hypothetical document and use it for retrieval.

The intuition: a generated answer paragraph lives in the same linguistic register as real document chunks. Embedding it retrieves documents that look like answers, not documents that look like questions.

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def hyde_retrieve(query: str, vector_store, top_k: int = 5):
    """
    HyDE: generate a hypothetical answer, embed it, retrieve with that embedding.
    """
    # Step 1: Generate a hypothetical document that answers the query
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a short, factual paragraph (3-5 sentences) that directly "
                    "answers the following question. Write as if this were an excerpt "
                    "from a technical document. Do not mention that you are generating "
                    "a hypothetical answer."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.0,
    )
    hypothetical_doc = response.choices[0].message.content

    # Step 2: Embed the hypothetical document (not the raw query)
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=hypothetical_doc,
    )
    hyde_embedding = np.array(embedding_response.data[0].embedding)

    # Step 3: Retrieve using the hypothetical document's embedding
    results = vector_store.similarity_search_by_vector(hyde_embedding, k=top_k)
    return results, hypothetical_doc
```

HyDE consistently improves retrieval recall on knowledge-intensive tasks, especially for technical or domain-specific queries where the vocabulary gap between question and answer is large. The cost is one extra LLM call per query — worth it for most production use cases.

**When HyDE helps most:** scientific literature search, legal document retrieval, technical documentation QA.

**When it hurts:** queries where the LLM generates a confidently wrong hypothetical. If the model hallucinates the hypothetical, you retrieve documents that support the hallucination. Always log both the query and the hypothetical for debugging.

---

## Fix 2: Reranking — A Second Opinion on Retrieval

Dense retrieval (embedding similarity) is fast but coarse. It finds roughly-relevant documents. A **reranker** is a cross-encoder that takes `(query, document)` as a pair and produces a precise relevance score — much slower than embedding similarity, but much more accurate.

The two-stage pipeline:

```
Query → Dense Retrieval (fast, recall-focused) → Top 50 candidates
                                                        ↓
                                               Reranker (slow, precision-focused)
                                                        ↓
                                               Top 5 reranked results → LLM
```

The reranker sees both query and document simultaneously — it can model their _interaction_, not just their individual semantics. This catches relevance signals that embedding similarity misses entirely.

```python
from sentence_transformers import CrossEncoder
from typing import List, Tuple

class TwoStageRetriever:
    def __init__(self, vector_store, reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vector_store = vector_store
        self.reranker = CrossEncoder(reranker_model)

    def retrieve(self, query: str, initial_k: int = 50, final_k: int = 5) -> List[dict]:
        # Stage 1: Dense retrieval — cast a wide net
        candidates = self.vector_store.similarity_search(query, k=initial_k)

        # Stage 2: Rerank with cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # Sort by reranker score, take top final_k
        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True
        )
        return [{"document": doc, "score": float(score)} for score, doc in ranked[:final_k]]
```

**Reranker model options:**

| Model                                   | Size | Speed    | Quality          |
| --------------------------------------- | ---- | -------- | ---------------- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2`  | 22M  | Fast     | Good baseline    |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M  | Moderate | Better           |
| `BAAI/bge-reranker-v2-m3`               | 568M | Slow     | Excellent        |
| Cohere Rerank API                       | —    | API call | Production-grade |

The initial_k / final_k ratio matters: retrieve 50, rerank to 5. If you retrieve too few initially, the reranker can't save you — the relevant document was never in the candidate set. If you rerank too many, latency explodes. 30–100 → 3–10 is the typical production range.

---

## Fix 3: ColBERT — Late Interaction for Both Speed and Precision

HyDE and reranking are two ends of a spectrum: fast-and-coarse vs. slow-and-precise. **ColBERT** sits in the middle with a fundamentally different approach: **late interaction**.

In standard dense retrieval, you embed a query into one vector and documents into one vector, then compute dot product. The entire semantic content of the query is compressed into a single point.

ColBERT instead produces one embedding **per token** — both for the query and the document. Relevance is computed as the **sum of maximum similarities** between query tokens and document tokens:

```
Score(q, d) = Σ_{i ∈ query_tokens} max_{j ∈ doc_tokens} (q_i · d_j)
```

Each query token independently finds its best matching document token. A query like "transformer training instability" has its "instability" token match against "gradient explosion" in the document, and its "transformer" token match against "attention layer" — neither match alone would be sufficient, but their sum is.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class ColBERTRetriever:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts: list[str], is_query: bool = False) -> torch.Tensor:
        """
        Encode texts into per-token embeddings.
        Returns: (batch, seq_len, dim)
        """
        prefix = "[Q] " if is_query else "[D] "
        prefixed = [prefix + t for t in texts]

        tokens = self.tokenizer(
            prefixed,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = self.model(**tokens)

        # L2-normalize each token embedding
        embeddings = F.normalize(output.last_hidden_state, p=2, dim=-1)
        return embeddings

    def score(self, query_embs: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """
        Late interaction scoring: sum of max similarities.

        query_embs: (q_len, dim)
        doc_embs:   (d_len, dim)
        Returns: scalar relevance score
        """
        # Similarity matrix: (q_len, d_len)
        sim_matrix = torch.matmul(query_embs, doc_embs.T)

        # Max over document tokens for each query token
        max_sim = sim_matrix.max(dim=-1).values  # (q_len,)

        # Sum across query tokens
        return max_sim.sum()

    def retrieve(self, query: str, doc_embeddings: list, top_k: int = 5):
        query_embs = self.encode([query], is_query=True)[0]  # (q_len, dim)

        scores = []
        for doc_emb in doc_embeddings:
            score = self.score(query_embs, doc_emb)
            scores.append(score.item())

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, scores[i]) for i in top_indices]
```

**The scalability trick:** document token embeddings are precomputed and stored (ColBERT indexes can be built with the `RAGatouille` library). At query time, you only compute query embeddings on the fly. The max-similarity operation over precomputed document embeddings is fast enough for millions of documents using PLAID (the ColBERT production indexing system).

**When ColBERT shines:** long documents where individual passages have multiple distinct topics, queries that require matching several specific concepts simultaneously, technical retrieval where exact terminology matters.

---

## Fix 4: Graph RAG — When Structure Is the Answer

All three techniques above assume documents are independent, unrelated chunks. But real knowledge has structure: entities reference each other, concepts build on each other, facts have provenance. A knowledge base isn't a bag of paragraphs — it's a graph.

**Graph RAG** (popularized by Microsoft Research in 2024) builds an explicit knowledge graph from your documents and uses graph traversal alongside or instead of vector search.

The pipeline has two phases:

### Phase 1: Graph Construction

```python
from openai import OpenAI
import networkx as nx
import json

client = OpenAI()

def extract_entities_and_relations(text: str) -> dict:
    """Use LLM to extract a structured knowledge graph from a text chunk."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Extract entities and relationships from the text.
Return JSON with this structure:
{
  "entities": [{"name": str, "type": str, "description": str}],
  "relations": [{"source": str, "relation": str, "target": str}]
}
Only include clearly stated facts. Be conservative."""
            },
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)


def build_knowledge_graph(chunks: list[str]) -> nx.DiGraph:
    G = nx.DiGraph()

    for chunk in chunks:
        extracted = extract_entities_and_relations(chunk)

        for entity in extracted.get("entities", []):
            G.add_node(
                entity["name"],
                type=entity["type"],
                description=entity["description"]
            )

        for rel in extracted.get("relations", []):
            G.add_edge(
                rel["source"],
                rel["target"],
                relation=rel["relation"]
            )

    return G
```

### Phase 2: Graph-Augmented Retrieval

```python
def graph_rag_retrieve(query: str, graph: nx.DiGraph, vector_store, top_k: int = 5):
    """
    Two-path retrieval:
    1. Vector search for relevant chunks
    2. Entity extraction from query → graph traversal for related context
    """
    # Path 1: Standard vector retrieval
    vector_results = vector_store.similarity_search(query, k=top_k)

    # Path 2: Extract entities from query, traverse the graph
    entity_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract entity names from this query. Return a JSON array of strings."
            },
            {"role": "user", "content": query}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    query_entities = json.loads(entity_response.choices[0].message.content).get("entities", [])

    # Gather neighborhood context for each query entity
    graph_context = []
    for entity in query_entities:
        if entity not in graph:
            continue

        # Direct neighbors (1-hop)
        neighbors = list(graph.neighbors(entity)) + list(graph.predecessors(entity))

        for neighbor in neighbors[:10]:  # cap at 10 neighbors
            edge_data = graph.get_edge_data(entity, neighbor) or graph.get_edge_data(neighbor, entity)
            relation = edge_data.get("relation", "related to") if edge_data else "related to"
            graph_context.append(f"{entity} {relation} {neighbor}")

        # Include entity description
        node_data = graph.nodes.get(entity, {})
        if node_data.get("description"):
            graph_context.append(f"{entity}: {node_data['description']}")

    return {
        "vector_results": vector_results,
        "graph_context": graph_context,
        "query_entities": query_entities,
    }
```

### Why Graph RAG Wins on Relationship Queries

Consider a corpus of earnings call transcripts. The question: _"How has the relationship between Amazon's AWS margins and its advertising revenue changed over the past three years?"_

Vector retrieval returns chunks that mention AWS margins and chunks that mention advertising revenue — independently. It has no way to represent the _relationship between these two things over time_ without explicit graph structure connecting them.

Graph RAG extracts `AWS_Margins --[affects]--> Advertising_Investment`, `Advertising_Revenue --[grew_by]--> 27%_in_Q3_2024`, and so on. Traversing from the query entities gives back the entire chain of relationships — exactly what the question is asking for.

Microsoft's GraphRAG paper showed 3–4× improvement in "global question" tasks (questions requiring synthesis across multiple documents) compared to naive RAG. Local question performance was roughly equivalent. Graph construction is expensive (many LLM calls), but for static corpora, you pay it once.

---

## Putting It All Together: The Production Pipeline

These techniques compose. A production RAG system typically layers several:

```python
class ProductionRAG:
    def __init__(self, vector_store, graph, reranker_model):
        self.vector_store = vector_store
        self.graph = graph
        self.reranker = CrossEncoder(reranker_model)
        self.llm = OpenAI()

    def retrieve(self, query: str) -> dict:
        # 1. HyDE: generate hypothetical answer for better query embedding
        hypothetical = self._generate_hypothetical(query)

        # 2. Dual retrieval: embed both raw query and hypothetical, merge results
        raw_results = self.vector_store.similarity_search(query, k=30)
        hyde_results = self.vector_store.similarity_search(hypothetical, k=30)

        # Deduplicate by document ID
        seen = set()
        candidates = []
        for doc in raw_results + hyde_results:
            if doc.metadata["id"] not in seen:
                seen.add(doc.metadata["id"])
                candidates.append(doc)

        # 3. Graph augmentation: pull in relationship context
        graph_context = self._graph_retrieve(query)

        # 4. Rerank the merged candidate set
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:5]]

        return {
            "retrieved_docs": top_docs,
            "graph_context": graph_context,
            "hypothetical": hypothetical,
        }

    def answer(self, query: str) -> str:
        context = self.retrieve(query)

        doc_text = "\n\n---\n\n".join([d.page_content for d in context["retrieved_docs"]])
        graph_text = "\n".join(context["graph_context"])

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the provided context. "
                        "If the context doesn't contain the answer, say so explicitly. "
                        "Cite sources by referencing document metadata where available."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Document Context:\n{doc_text}\n\n"
                        f"Relationship Context:\n{graph_text}"
                    ),
                },
            ],
        )
        return response.choices[0].message.content

    def _generate_hypothetical(self, query: str) -> str:
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Write a 3-sentence factual answer to this question."},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _graph_retrieve(self, query: str) -> list[str]:
        # Entity extraction + graph traversal (as shown above)
        ...
```

---

## Chunking Strategy: The Foundation Still Matters

Advanced retrieval can't compensate for bad chunks. A few patterns worth knowing:

**Semantic chunking** — instead of fixed-size splits, chunk at semantic boundaries detected by embedding distance between consecutive sentences. When the embedding distance spikes, you've crossed a topic boundary.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunk(text: str, threshold: float = 0.3) -> list[str]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = text.split(". ")
    embeddings = model.encode(sentences)

    # Cosine distance between consecutive sentences
    chunks, current_chunk = [], [sentences[0]]
    for i in range(1, len(sentences)):
        cos_sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
        )
        if 1 - cos_sim > threshold:  # topic boundary detected
            chunks.append(". ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(". ".join(current_chunk))
    return chunks
```

**Parent-child chunking** — index small child chunks (128 tokens) for retrieval precision, but return the larger parent chunk (512 tokens) to the LLM for context. You get the best of both: precise matching, rich context.

**Proposition indexing** — extract atomic factual statements from each chunk, embed and index those instead of the raw text. Each proposition is a single retrievable claim. Much higher precision for fact-seeking queries; higher construction cost.

---

## Key Takeaways

1. **Query-document mismatch is the root cause of most retrieval failures** — HyDE fixes it by generating a hypothetical answer that lives in the same linguistic space as your documents.

2. **Dense retrieval is for recall; reranking is for precision** — retrieve 50, rerank to 5. Never use a single-stage retriever in production.

3. **ColBERT's per-token embeddings capture multi-concept queries** that single-vector retrieval compresses away. Worth the indexing overhead for complex domains.

4. **Graph RAG isn't about replacing vector search — it's about adding structure** that vector search is blind to. Use it when your queries ask about relationships, not just topics.

5. **Chunking strategy is the foundation everything else sits on** — semantic chunking and parent-child indexing are the minimum viable improvements over naive fixed-size splits.

6. **These techniques compose** — HyDE + two-stage reranking + graph augmentation is a coherent production stack, not competing approaches.

---

## Further Reading

### Foundational Papers

- **[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)** — Lewis et al., Facebook AI (2020)
  The original RAG paper. Combines a dense retriever (DPR) with a seq2seq generator (BART). Sets the vocabulary and evaluation benchmarks the field still uses.

- **[Dense Passage Retrieval for Open-Domain Question Answering (DPR)](https://arxiv.org/abs/2004.04906)** — Karpukhin et al., Facebook AI (2020)
  The bi-encoder retrieval model that underpins most dense retrieval systems. Training with in-batch negatives is the key technique.

### HyDE

- **[Precise Zero-Shot Dense Retrieval Without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496)** — Gao et al., CMU (2022)
  The original HyDE paper. Shows consistent improvements across BEIR benchmark tasks, especially for low-resource domains where query-document vocabulary mismatch is severe.

### Reranking

- **[MS MARCO: A Human Generated Machine Reading Comprehension Dataset](https://arxiv.org/abs/1611.09268)** — Bajaj et al., Microsoft (2016)
  The dataset most reranker models are trained on. Understanding its structure helps you evaluate whether a reranker will generalize to your domain.

- **[A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models](https://arxiv.org/abs/2310.09497)** — Zhuang et al. (2023)
  LLM-based reranking — using GPT-4 or similar as a reranker directly. Slower but often better than cross-encoders for out-of-domain queries.

### ColBERT

- **[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)** — Khattab & Zaharia, Stanford (2020)
  The original ColBERT paper. Introduces late interaction and the deferred MaxSim operation that makes it scalable.

- **[ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)** — Santhanam et al., Stanford (2021)
  Improves ColBERT with residual compression — reduces storage 6–10× while maintaining quality. The version used in production systems.

- **[RAGatouille](https://github.com/bclavie/RAGatouille)** — Benjamin Clavié
  The library that makes ColBERT accessible in 10 lines of code. Wraps PLAID indexing, supports Hugging Face datasets, integrates with LangChain/LlamaIndex.

### Graph RAG

- **[From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)** — Edge et al., Microsoft Research (2024)
  The GraphRAG paper. Shows 3–4× improvement on global synthesis tasks. Introduces community detection over the knowledge graph as a summarization strategy — compelling results on large corpora.

- **[KGRAG: Knowledge Graph Enhanced Retrieval-Augmented Generation](https://arxiv.org/abs/2311.13314)** — (2023)
  Alternative approach using existing knowledge graphs (Wikidata, domain KGs) rather than LLM-extracted ones. Lower construction cost, less flexible.

### Chunking & Indexing

- **[Propositions as the Unit of Retrieval (Dense X Retrieval)](https://arxiv.org/abs/2312.06648)** — Chen et al. (2023)
  Atomic proposition indexing — extract and index individual factual claims rather than passages. Significant precision improvement for fact-seeking queries.

- **[RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)** — Sarthi et al., Stanford (2024)
  Builds a hierarchical tree of summaries over your document corpus. Retrieval happens at multiple levels of abstraction simultaneously — strong on multi-hop reasoning tasks.

---

_Naive RAG is a prototype. Production RAG is a system. The difference isn't any single technique — it's building a retrieval stack that thinks about where each failure mode comes from and applies the right fix at the right layer. Get that right, and your LLM stops hallucinating not because you made it smarter, but because you stopped giving it reasons to guess._
