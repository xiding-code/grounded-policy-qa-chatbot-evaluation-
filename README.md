This notebook builds and evaluates a customer-support chatbot for **Amazon returns/refunds policy** questions.  
It compares three approaches:

1) **LLM-only** (no retrieval)  
2) **Retrieval-only** (answer from documents only)  
3) **RAG** (retrieval + LLM)

## Project Goal
- Answer returns/refunds policy questions accurately
- Provide consistent decision labels (structured output)
- Test robustness via **red-teaming** and a small evaluation grid

## Data / Corpus
- A curated corpus of **~10 Amazon policy/help pages**
- The notebook converts URLs → clean text and caches them locally
- Uses a manifest file for reproducibility:
  - `data/text/` (cached documents)
  - `docs_manifest.csv` (document index)

### Part A — Domain framing + corpus manifest
- Domain definition and scope
- Build a small document corpus from Amazon pages
- Create `docs_manifest.csv` and sanity-check document text

### Part B — Chatbot (LLM-only vs Retrieval-only vs RAG)
- Chunking strategy
- Embeddings + FAISS index
- Retriever (top-k = 2 or 5)
- Unified prompting + 3-mode implementation
- End-to-end tests

### Part C — Red-Teaming & Evaluation
- Domain-specific red-team questions
- Evaluation configuration
- Runner to execute and compare modes
