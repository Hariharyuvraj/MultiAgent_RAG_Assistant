# MultiAgent RAG Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system built with a multi-agent architecture. Upload PDF or TXT documents and ask questions — the system retrieves relevant context, reasons over it through a pipeline of specialised AI agents, and delivers grounded, cited answers. For queries outside your documents, it automatically falls back to live web search.

---

## Features

- **Multi-Agent Pipeline** — five specialised agents handle context understanding, query planning, hybrid search, answer generation, and grounding verification
- **Hybrid Search** — combines dense vector search (ChromaDB + BGE embeddings) with BM25 keyword search, fused via Reciprocal Rank Fusion (RRF)
- **Live Web Search Fallback** — out-of-domain and real-time queries are automatically routed to DuckDuckGo web search
- **Content Safety Filter** — LLM-based classifier blocks harmful, illegal, or policy-violating queries before they reach the pipeline
- **Grounding Guard** — cosine similarity check between the generated answer and retrieved chunks; answers that don't meet the threshold are rejected
- **GPT-Style Dark UI** — ChatGPT-inspired Streamlit interface with session history, document management, and source citations
- **Evaluation System** — custom trace logger and metrics reporter; RAGAS-compatible ground truth generation from documents
- **CLI Interface** — ingest documents, run queries, and view eval metrics without the UI

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    Safety Filter                        │
│  LLM classifier — blocks harmful/illegal queries        │
└────────────────────────┬────────────────────────────────┘
                         │ SAFE
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Context Agent                          │
│  Detects follow-up questions, enriches query with       │
│  conversation history                                   │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Planner Agent                          │
│  Breaks query into 2–4 focused retrieval sub-queries    │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Search Agent                           │
│                                                         │
│  Temporal gate  → real-time queries → Web Search        │
│  Relevance gate → out-of-domain   → Web Search          │
│                                                         │
│  In-domain:                                             │
│  Dense Vector (ChromaDB) + BM25 → RRF Fusion (k=60)    │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Analyst Agent                          │
│  Streams answer with inline citations [source:page]     │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Guard Agent                            │
│  Cosine similarity (answer ↔ chunks) ≥ 0.60 threshold  │
│  Retries up to 2× before fallback message               │
└────────────────────────┬────────────────────────────────┘
                         ▼
                   Final Answer
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq — llama-3.1-8b-instant (Google Gemini supported) |
| **Embeddings** | HuggingFace — BAAI/bge-small-en-v1.5 |
| **Vector Store** | ChromaDB (cosine similarity) |
| **Keyword Search** | BM25Okapi via rank-bm25 |
| **Search Fusion** | Reciprocal Rank Fusion (RRF, k=60) |
| **Agent Framework** | Google ADK (SequentialAgent) |
| **LLM Orchestration** | LangChain |
| **Web Search** | DuckDuckGo (ddgs) |
| **UI** | Streamlit with dark GPT-style theme |
| **Database** | SQLite (conversation + session history) |
| **Document Parsing** | PyPDF, python-docx |
| **Eval** | Custom TraceLogger + RAGAS-compatible pipeline |
| **CLI** | Typer + Rich |

---

## Project Structure

```
MultiAgent_RAG_Assistant/
│
├── streamlit_app.py          # Main UI — GPT-style dark Streamlit interface
├── app.py                    # ADK pipeline runner (async)
├── cli.py                    # CLI: ingest / query / eval / generate-gt / ragas
│
├── agents/                   # Individual ADK agent implementations
│   ├── context_agent.py      # Conversation history + query enrichment
│   ├── planner_agent.py      # Query decomposition into sub-queries
│   ├── search_agent.py       # Hybrid retrieval + web fallback
│   ├── analyst_agent.py      # Answer generation with citations
│   └── guard_agent.py        # Grounding verification
│
├── pipeline/
│   └── rag_pipeline.py       # Synchronous pipeline wrapper (UI + eval)
│
├── graph/
│   └── agent_graph.py        # Assembles agents into SequentialAgent
│
├── ingest/
│   ├── doc_loader.py         # PDF / TXT document loader
│   ├── text_splitter.py      # Recursive character text splitter
│   └── embedder.py           # Embeds chunks and stores in ChromaDB
│
├── guardrails/
│   └── grounding_check.py    # Cosine similarity grounding score
│
├── providers/
│   ├── llm_factory.py        # Groq / Gemini LLM factory
│   └── embedding_factory.py  # HuggingFace embeddings factory
│
├── schemas/
│   └── state.py              # AgentState, ChunkResult, WebResult (Pydantic v2)
│
├── db/
│   └── sqlite_manager.py     # SQLite: sessions, messages, documents
│
├── eval/
│   ├── trace_logger.py       # Saves JSON trace per query to eval/traces/
│   ├── metrics.py            # Grounding rate, avg retrieval score, avg latency
│   ├── generate_gt.py        # Auto-generates ground truth CSV from documents
│   └── ragas_runner.py       # RAGAS evaluation runner (Groq + HuggingFace)
│
├── tests/
│   ├── test_agents.py        # Unit tests for all 5 agents
│   ├── test_ingest.py        # Document loader + splitter tests
│   ├── test_guardrails.py    # Grounding score logic tests
│   ├── test_providers.py     # LLM + embeddings factory tests
│   └── test_graph.py         # Pipeline assembly test
│
├── config/
│   └── settings.yaml         # All configuration (LLM, embeddings, retrieval, eval)
│
├── .streamlit/
│   └── config.toml           # Dark theme configuration
│
├── .env.example              # Environment variable template (copy to .env)
├── requirements.txt          # Python dependencies
└── documents/                # Place your PDF / TXT files here (gitignored)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MultiAgent_RAG_Assistant.git
cd MultiAgent_RAG_Assistant
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here      # free at console.groq.com
HF_TOKEN=your_hf_token_here              # optional, huggingface.co/settings/tokens
LLM_PROVIDER=groq
EMBEDDING_PROVIDER=huggingface
```

### 5. Create required directories

```bash
mkdir documents storage
```

---

## Usage

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

- **Upload documents** — click the Documents button in the sidebar, upload PDF or TXT files
- **Ask questions** — type in the chat input; the system retrieves context and streams an answer
- **Session history** — all conversations are saved and grouped by date (Today / Yesterday / Previous 7 Days)
- **New Chat** — start a fresh session with the New Chat button
- **Source citations** — every answer shows which document pages were used

### CLI

```bash
# Index all documents in the documents/ folder
python cli.py ingest

# Ask a single question from the terminal
python cli.py query "What is the tax slab for income above 15 lakhs?"

# View evaluation metrics from saved traces
python cli.py eval

# Generate ground truth Q&A pairs from your documents (uses Groq LLM)
python cli.py generate-gt

# Run RAGAS evaluation against the generated ground truth
python cli.py ragas --max-q 10
```

---

## How It Works

### Document Ingestion

1. PDF/TXT files are loaded page-by-page using `PyPDFLoader`
2. Text is split into 800-character chunks with 100-character overlap
3. Each chunk is embedded using `BAAI/bge-small-en-v1.5`
4. Vectors are stored persistently in ChromaDB at `./storage/vectordb/`

### Query Pipeline

Every user query passes through 5 stages in sequence:

| Stage | Agent | What it does |
|---|---|---|
| 1 | Safety Filter | LLM classifies query — blocks harmful/illegal content immediately |
| 2 | Context Agent | Detects follow-up questions, prepends history to query if needed |
| 3 | Planner Agent | Breaks query into 2–4 specific retrieval sub-queries |
| 4 | Search Agent | Runs hybrid search or web fallback based on query type |
| 5 | Analyst Agent | Streams answer token-by-token with inline source citations |
| 6 | Guard Agent | Checks answer is grounded in retrieved chunks (cosine ≥ 0.60) |

### Hybrid Search

Both retrieval methods run in parallel for every sub-query, then their ranked results are merged using Reciprocal Rank Fusion:

```
Dense search  (ChromaDB cosine similarity) → rank list A
BM25 search   (BM25Okapi keyword scoring)  → rank list B
                        │
              Reciprocal Rank Fusion
              score = Σ 1 / (60 + rank)
                        │
              Top-K merged chunks
```

Dense search captures semantic meaning. BM25 captures exact keyword matches. RRF combines both without needing score normalisation.

### Web Search Routing

Two gates decide when to bypass documents and go to web:

| Gate | Condition | Trigger |
|---|---|---|
| Temporal | Query contains "latest", "current", "who is", "news", "2025", etc. | Direct web search |
| Relevance | Best dense similarity score < 0.50 | Query is out-of-domain |

When web search returns no results, the LLM answers from its training knowledge.

---

## Evaluation

Every query produces a JSON trace saved to `eval/traces/`:

```json
{
  "session_id": "ed2eafe2",
  "query": "Who can file ITR-1?",
  "enriched_query": "...",
  "plan": ["Identify ITR-1 eligibility", "..."],
  "retrieved_chunks": [
    { "source": "ITR_Guide.pdf", "page": 2, "score": 0.81, "content": "..." }
  ],
  "answer": "ITR-1 can be filed by salaried individuals...",
  "grounding_score": 0.849,
  "passed_guard": true,
  "latency_ms": 7054
}
```

### Metrics

| Metric | Description | Target |
|---|---|---|
| Grounding Pass Rate | % of answers where Guard Agent approved | > 80% |
| Avg Retrieval Score | Mean cosine similarity of retrieved chunks | > 0.65 |
| Avg Latency | Mean end-to-end pipeline time | < 6000 ms |

```bash
python cli.py eval
```

### RAGAS Evaluation

Ground truth Q&A pairs are auto-generated from your documents:

```bash
python cli.py generate-gt        # ~57 Q&A pairs from your PDFs
python cli.py ragas --max-q 10   # runs 5 RAGAS metrics
```

RAGAS metrics: **Faithfulness**, **Answer Relevancy**, **Context Precision**, **Context Recall**, **Answer Correctness** — computed using Groq + HuggingFace (no OpenAI key required).

---

## Tests

```bash
pytest tests/ -v
```

| Test File | What it covers |
|---|---|
| `test_agents.py` | All 5 agents with mocked LLM and vector store |
| `test_ingest.py` | Document loader and text splitter |
| `test_guardrails.py` | Cosine similarity grounding score math |
| `test_providers.py` | LLM and embeddings factory functions |
| `test_graph.py` | Pipeline assembly — SequentialAgent with 5 sub-agents |

---

## Configuration

All settings live in `config/settings.yaml`:

```yaml
llm:
  provider: "groq"
  model_name: "llama-3.1-8b-instant"
  temperature: 0.2
  max_tokens: 4096

embeddings:
  model_name: "BAAI/bge-small-en-v1.5"

retrieval:
  top_k: 5
  score_threshold: 0.35
  min_relevance_threshold: 0.50   # below this → web fallback

agents:
  guard:
    grounding_threshold: 0.60     # below this → answer rejected
    max_retries: 2
```

---

## Content Safety

Every query is classified by the LLM before any processing begins. Queries in any of the following categories are blocked with a refusal message:

| Category | Example |
|---|---|
| Illegal activity | Fraud, hacking, weapon/drug synthesis |
| Violence | Planning or glorifying physical harm |
| Hate speech | Content targeting race, religion, gender |
| Self-harm | Suicide or self-injury instructions |
| Explicit content | CSAM or non-consensual content |
| Privacy violation | Doxxing, identity theft |

Legitimate educational, medical, legal, and security research queries are always allowed through.

---

## Requirements

- Python 3.10+
- Groq API key — free tier at [console.groq.com](https://console.groq.com)
- HuggingFace token — optional, [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## License

MIT
