# Grounded Policy QA — Amazon Returns & Refunds RAG Chatbot

Production-shaped RAG chatbot built on Amazon's public returns and refunds
policy corpus. Originally a single-turn course notebook (BU BA840); this v2
re-implementation adds hybrid retrieval, cross-encoder reranking,
three-tier conversation memory, entity tracking, adaptive routing
(Self-RAG), corrective retrieval with web fallback (CRAG), a
multi-LLM provider switch, and a FastAPI web UI.

Built for the SA Cyber AI Digital Transformation Internship technical
interview but designed as a re-usable enterprise pattern.

---

## What it does

Answers Amazon returns and refunds policy questions with cited, auditable
decisions. Every response carries a structured decision label (APPROVE,
DENY, ESCALATE, ABSTAIN) plus citations back to specific policy chunks. A
safety guard forces ABSTAIN whenever the model claims APPROVE or DENY
without citing evidence, preventing hallucinated confidence.

Multi-turn conversations preserve context. The system remembers what
product the user was asking about, what order ID they mentioned, and what
their goal is. It decides on its own whether a given turn needs a policy
lookup at all (chit-chat skips retrieval), and falls back to web search
when its in-corpus retrieval quality is insufficient.

---

## Full per-turn pipeline

```
User question + session_id
        │
        ▼
[L3]  Entity extraction (rule-based + LLM)
      → persist order_id / product / return_reason / intent
        │
        ▼
[Self-RAG]  LLM router: RETRIEVE or SKIP?
        │
        ├─ SKIP ──► Direct LLM reply ──► persist turn ──► return
        │
        ▼ RETRIEVE
[Retrieval]  Query embedded with MiniLM-L6
        │
        ├──► FAISS dense top-20 ──┐
        │                          ├──► Reciprocal Rank Fusion (top-20)
        └──► BM25 sparse  top-20 ──┘
                                   │
                                   ▼
                       Cross-encoder reranker
                       (ms-marco-MiniLM-L-6-v2)
                                   │ top-5
                                   ▼
[CRAG]  LLM judge: HIGH or LOW quality?
        │
        ├─ LOW ──► DuckDuckGo fallback chunks merged into evidence
        │
        ▼
[L2]  If session has > 5 prior turns, summarize older turns
      and keep most recent 3 verbatim
        │
        ▼
[L1]  Pull rolling-buffer conversation history into LLM messages
        │
        ▼
Inject SYSTEM prompt + entity context + summary
        │
        ▼
LLM generation (Claude or OpenAI via .env switch)
        │
        ▼
Parse structured output: DECISION + ANSWER + REQUIRED_INFO + CITATIONS
        │
        ▼
Safety guard: if APPROVE/DENY without citation → force ABSTAIN
        │
        ▼
Persist turn to session buffer
        │
        ▼
Return JSON with decision, evidence, entities, self_rag, crag, summary
```

---

## What changed vs the baseline notebook

| Layer | Baseline (BA840) | v2 (this repo) |
|---|---|---|
| Chunking | 800-char fixed, 150 overlap | Paragraph-greedy packing with sentence-boundary fallback for oversized paragraphs, 2-sentence overlap |
| Retrieval | FAISS dense cosine only | **BM25 sparse + FAISS dense via Reciprocal Rank Fusion (RRF)** |
| Re-ranking | None | **Cross-encoder ms-marco-MiniLM-L-6-v2** over top-20 down to top-5 |
| Generation | Local Qwen 2.5 1.5B | **Anthropic Claude or OpenAI GPT** via `.env` switch (provider-agnostic) |
| Schema | APPROVE / DENY / ESCALATE / ABSTAIN + CITATIONS | Preserved, plus explicit `guarded` flag in output JSON |
| Conversation memory | None (stateless single-turn) | **L1 rolling buffer + L2 summary compression** when long |
| Entity tracking | None | **L3 slot tracker** for order_id / product / return_reason / intent |
| Retrieval routing | Always retrieve | **Self-RAG** classifier skips retrieval on chit-chat |
| Quality control | None on retrieval side | **CRAG** judge scores retrieval, web fallback when LOW |
| Storage | In-memory per notebook run | Pickled chunks plus FAISS index plus BM25 pickle on disk |
| Code structure | Single notebook | Modular `src/` and `webapp/` packages, FastAPI server |
| UI | None | Dark-themed single-page web app with evidence audit trail |

---

## Quickstart

### One-time setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys (see "Configuration" below)
cp .env.example .env
# then open .env and paste your real keys

# Build the corpus and indexes (fetch 6 Amazon help pages, clean, chunk, index)
python3 -m src.fetch
python3 -m src.clean
python3 -m src.chunk
python3 -m src.index
```

### Run

```bash
# CLI: one-shot question
python3 -m src.chat "How long do I have to return an item to Amazon?"

# CLI: 5-question demo grid (normal, ESCALATE, adversarial)
python3 run_demo.py

# Web UI (recommended for demo)
uvicorn webapp.server:app --port 8000
# then open http://localhost:8000
```

---

## Configuration (`.env`)

The chatbot reads provider settings, model names, and API keys from a
`.env` file in the project root. **`.env` is gitignored and must never
be committed.** A template is provided in `.env.example`.

```bash
# .env.example — safe to commit
LLM_PROVIDER=anthropic              # or "openai"
ANTHROPIC_MODEL=claude-sonnet-4-6
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=sk-ant-PASTE-YOUR-ANTHROPIC-KEY-HERE
OPENAI_API_KEY=sk-PASTE-YOUR-OPENAI-KEY-HERE
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=400
```

Switching providers means changing one line in `.env` and restarting the
server. The same prompt template and safety guard apply to both backends.

If `.env` is missing or the key is still the placeholder, the chatbot
boots into heuristic-stub mode so the pipeline can still demo retrieval
and parsing without an API call. The boot log makes the mode explicit.

---

## Web UI

Single-page web app built with FastAPI plus Tailwind plus Alpine.js (no
build step). Surfaces every layer of the stack as visible badges, so the
audit trail is the demo.

```bash
# install deps (one-time)
pip install -r requirements.txt

# boot the server (chatbot loads ONCE at startup, ~3-10s)
uvicorn webapp.server:app --port 8000

# open http://localhost:8000
```

The UI shows:

| Badge | When it appears | Color |
|---|---|---|
| APPROVE / DENY / ESCALATE / ABSTAIN | Every turn | Green / red / amber / gray |
| Safety guard triggered | APPROVE or DENY without a citation | Amber |
| +N prior turns in context | n_prior_turns > 0 | Purple |
| Self-RAG: skipped retrieval | Router classified the question as chit-chat | Cyan |
| CRAG: HIGH | Judge says retrieval was sufficient | Emerald |
| CRAG: LOW + N web fallback | Judge says retrieval was poor, web supplemented | Orange |
| L2: N turns summarized | Buffer compressed when > 5 raw turns | Fuchsia |
| L3 tracked entities | order_id / product / return_reason / intent slots filled | Purple panel |

The right-side **Evidence** panel highlights the chunks cited by the
current answer so the user can trace decision back to source.

### HTTP endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/`           | Single-page UI |
| `POST` | `/api/ask`    | `{"question": "...", "session_id": "..."}` returns full result dict |
| `POST` | `/api/reset`  | `{"session_id": "..."}` clears that session's buffer and entities |
| `GET`  | `/api/health` | Provider info + load time |

---

## Layer-by-layer reference

### Corpus and ingestion (`src/fetch.py`, `src/clean.py`)

Six Amazon help pages on returns and refunds policy. Fetched with a
browser User-Agent header (Amazon returns 403 to default `requests`),
cleaned with BeautifulSoup, boilerplate stripped, deduped, cached to
`data/raw/{doc_id}.html` and `data/text/{doc_id}.txt`. Reproducibility
through a `docs_manifest.csv`.

### Chunking (`src/chunk.py`)

Three-tier hierarchical packing:

1. **Paragraph greedy** is the default: split on blank lines, pack
   paragraphs greedily into roughly 800-char windows.
2. **Sentence-boundary fallback** kicks in for paragraphs over 1200
   chars. Splits on the regex `(?<=[.!?])\s+(?=[A-Z])` which uses
   lookbehind and lookahead so punctuation is preserved and false
   breaks on abbreviations are reduced.
3. **2-sentence overlap** between adjacent chunks preserves coreference.

Constants are `TARGET_SIZE=800`, `MAX_SIZE=1200`, `OVERLAP_SENTS=2`.

### Indexing (`src/index.py`)

Two indexes built from the same chunk corpus:

- **Dense**: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, L2-normalized
  vectors stored in a FAISS `IndexFlatIP`. Inner product on normalized
  vectors equals cosine similarity. Exact search since the corpus is small.
- **Sparse**: `rank_bm25.BM25Okapi` with a lowercase whitespace tokenizer.

Both serialized to `data/index/`.

### Hybrid retrieval (`src/retrieve.py`)

For each query the system runs dense + sparse retrieval in parallel
(top-20 each), then fuses them with **Reciprocal Rank Fusion** (Cormack
et al. 2009, `score = sum(1/(60 + rank))`). RRF is robust to scale
mismatches between cosine and BM25 scores and requires no tuning.

A **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
scores the fused top-20 pairwise against the query and promotes the
top-5. Cross-encoders catch semantic relevance that bi-encoders miss
but cost more per pair, so they only see the top-20 pool, not the full
67-chunk corpus.

### LLM provider abstraction (`src/llm.py`)

`call_llm(system_prompt, user_prompt, history=None, provider=None)`
dispatches to Anthropic or OpenAI based on `LLM_PROVIDER` in `.env`.
Both backends accept a multi-turn `history` argument so conversation
memory works identically for either provider.

### Structured output + safety guard (`src/schema.py`)

The LLM is constrained to a 4-field text format:

```
DECISION: APPROVE | DENY | ESCALATE | ABSTAIN
ANSWER: <one or two sentences>
REQUIRED_INFO: NONE | <missing info>
CITATIONS: NONE | <chunk_id list>
```

`parse_structured()` is the contract enforcer. If the parsed DECISION is
not in the allowed set, it is coerced to ABSTAIN. If DECISION is APPROVE
or DENY but CITATIONS is empty or NONE, the safety guard fires and the
decision is downgraded to ABSTAIN with `guarded=True`. This prevents
the most common hallucination failure mode in policy QA: confident
answers without source attribution.

### L1 conversation memory (`src/memory.py`)

Session-keyed rolling buffer. Each session keeps the last N turns
(default 12), capped to bound token cost. Returns history in the
standard `[{"role": "user|assistant", "content": str}]` format used by
both Anthropic and OpenAI SDKs.

### L2 summary buffer (`src/summary_buffer.py`)

When a session exceeds 5 raw turns, an LLM summarization call folds the
older turns into a single condensed paragraph and prepends it as a
synthetic system note. The 3 most recent turns stay verbatim. Prior
summaries are carried forward so the compression is incremental, not
recomputed every turn.

### L3 entity tracker (`src/entity_tracker.py`)

Slot filler over four entity types:

- `order_id`: Amazon format `\d{3}-\d{7}-\d{7}` caught by regex
- `product`: keyword list (`laptop`, `iphone`, `macbook`, ...) plus LLM gap-fill
- `return_reason`: enum `damaged | defective | wrong-item | changed-mind | late | other`
- `intent`: enum `return | refund | track | exchange | escalate | inquiry`

Rule-based regex runs first (high precision on order IDs). If LLM is
configured and rule pass left slots empty, a small JSON-mode LLM call
fills the rest. Tracked slots get injected into the system prompt on
every subsequent turn as a "Tracked context for this user" block so
the LLM has structured context even when the user uses pronouns.

### Self-RAG router (`src/self_rag.py`)

Before retrieval runs, a binary LLM classifier (Asai et al. 2023,
simplified) decides whether the latest message needs a policy lookup.
Greetings, thank-yous, and out-of-scope questions are routed to
`SKIP`. The downstream LLM answers them directly without retrieval,
saving the 200-500ms retrieval + rerank step and reducing token cost.

A short-string greeting check shortcuts before any LLM call. If
LLM is not configured, everything routes to RETRIEVE as a safe default.

### CRAG corrective layer (`src/crag.py`)

After retrieval and reranking, an LLM judge (Yan et al. 2024,
simplified) scores the top-5 evidence against the question:
`HIGH` (sufficient) or `LOW` (off-topic, insufficient, or contradictory).

When verdict is `LOW`, the system fetches fallback evidence from the
DuckDuckGo Instant Answer API (no API key needed) scoped by the query
plus "Amazon returns refunds". Fallback chunks are merged into the
evidence list with `source: "web"` and a `WEB_` chunk_id prefix so the
UI can distinguish them.

In production, swap DuckDuckGo for Tavily, Brave, Serper, or Bing
depending on coverage and budget.

---

## File structure

```
amazon_rag_v2/
├── .env                              # API keys (gitignored)
├── .env.example                      # safe template
├── .gitignore                        # excludes .env, caches, indexes
├── README.md                         # this file
├── requirements.txt                  # pip deps
├── run_demo.py                       # CLI 5-question demo
├── data/
│   ├── docs_manifest.csv             # source URLs + paths
│   ├── chunks.pkl                    # 67 chunks
│   ├── raw/                          # cached HTML
│   ├── text/                         # cleaned text
│   └── index/
│       ├── faiss.index               # dense vector index
│       ├── bm25.pkl                  # sparse keyword index
│       └── chunks.pkl                # paired chunks for retriever
├── src/
│   ├── fetch.py                      # Amazon HTML fetch + cache
│   ├── clean.py                      # HTML to text
│   ├── chunk.py                      # hierarchical chunking
│   ├── index.py                      # build FAISS + BM25
│   ├── retrieve.py                   # hybrid RRF + cross-encoder rerank
│   ├── schema.py                     # structured output + safety guard
│   ├── llm.py                        # Anthropic + OpenAI + history support
│   ├── memory.py                     # L1 rolling buffer
│   ├── summary_buffer.py             # L2 summary compression
│   ├── entity_tracker.py             # L3 slot filler
│   ├── self_rag.py                   # smart retrieval routing
│   ├── crag.py                       # corrective retrieval + web fallback
│   └── chat.py                       # orchestrates all layers
└── webapp/
    ├── __init__.py
    ├── server.py                     # FastAPI app
    └── static/
        └── index.html                # single-page UI (Tailwind + Alpine CDN)
```

---

## Sample response shape

```jsonc
{
  "question": "Order 112-3456789-1234567, my MacBook arrived damaged. Can I get a refund?",
  "session_id": "demo-1",
  "decision": {
    "decision": "APPROVE",
    "answer": "Yes, you can return a damaged item within 30 days...",
    "required_info": "NONE",
    "citations": "A4_0011, A4_0025, A9_0001",
    "guarded": false
  },
  "evidence": [...],
  "n_prior_turns": 0,
  "skipped_retrieval": false,
  "self_rag":  { "route": "RETRIEVE", "reason": "llm-classified" },
  "crag":      { "quality": "HIGH",   "fallback_count": 0 },
  "summary":   null,
  "entities": {
    "order_id": "112-3456789-1234567",
    "product":  "macbook",
    "return_reason": "damaged",
    "intent":   "return",
    "any_set":  true
  },
  "entities_updated_this_turn": ["order_id", "product", "return_reason", "intent"],
  "provider":   "anthropic / claude-sonnet-4-6",
  "latency_ms": 6084
}
```

---

## Security notes

- **`.env` is gitignored.** Never commit API keys.
- The repo's `.gitignore` excludes `.env`, `__pycache__/`, `.DS_Store`,
  and the built indexes (which can be regenerated by the `src.fetch`
  through `src.index` pipeline).
- The chatbot only fetches from Amazon's public help pages and (when
  CRAG triggers a fallback) the DuckDuckGo Instant Answer API. No
  user data is sent anywhere except the configured LLM provider.
- API keys are loaded only via `python-dotenv` from `.env` at startup.
  They are never logged, never echoed in responses, and never written
  to any pickle or index file.

---

## Roadmap (still not shipped)

1. **Long-term memory**: PostgreSQL user profile, second FAISS index
   over past Q&A pairs as episodic semantic memory, Redis hot-session
   cache with TTL. The session-keyed in-memory store would swap for
   Redis in production without touching call sites.
2. **Evaluation harness**: port the original 30-prompt red-team grid
   (normal, ambiguous, adversarial) into a `pytest` suite with
   precision@k, decision accuracy, citation faithfulness, and
   safety-guard trigger rate metrics. CI integration to gate releases.
3. **Query rewriting before retrieval**: for follow-up turns, use the
   LLM to rewrite the question as a standalone query (resolve pronouns
   with entity context) before running retrieval. Currently the
   retriever sees the raw follow-up text; the LLM uses prior turns at
   generation time only.
4. **Markdown-aware chunking**: if upstream documents shift to a
   format with explicit heading hierarchy, switch from paragraph-greedy
   to heading-aware splitting for cleaner topical chunks.

---

## Credits

Built on top of the BU BA840 baseline notebook (LLM-only vs
retrieval-only vs RAG evaluation harness with 30 red-team prompts).
Retrieval and architecture inspirations: Cormack et al. 2009 (RRF),
Asai et al. 2023 (Self-RAG), Yan et al. 2024 (CRAG), Gao et al. 2022
(HyDE pattern reference).
