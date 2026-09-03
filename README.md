# HR Policy RAG — Version 5.2.1 (Version 1 + Version 2)

This branch documents the evolution of the HR-policy RAG application. The
pipeline is intentionally small so retrieval behavior can be measured and
debugged.

## Version 1: semantic RAG baseline

Version 1 follows:

```text
policy documents → chunks → local embeddings → semantic search → grounded LLM answer
```

It provides recursive and structure-aware chunking, local BGE embeddings,
Chroma vector storage, semantic retrieval, citation-aware Groq generation, and
refusal gates for unsupported questions.

Version 1 is the comparison point. Keep its embedding model and chunking
configuration unchanged while measuring a Version 2 improvement.

## Version 2: retrieval inspection and safer operation

Version 2 adds:

- BM25 keyword search and hybrid semantic + BM25 search using RRF
- region and document metadata on every chunk
- optional local Jina Turbo cross-encoder reranking
- PDF/Markdown upload with incremental indexing under both chunkers
- JSONL traces for generated answers and refusals
- seeded trace sampling and replay support
- stricter refusal handling and tighter explicit-ID redaction
- deterministic query normalization for common typos and spacing noise
- query understanding that detects explicitly named indexed regions/policy IDs
- parent-context expansion: neighboring chunks are supplied to generation while
  citations remain anchored to the original retrieved chunk

The Streamlit modes are:

1. **Ask** — retrieve chunks, apply the refusal gate, and generate a cited answer.
2. **Retrieve** — show top-k retrieved chunks and metadata without generation.
3. **Rerank** — retrieve candidates, apply the local cross-encoder, and show original versus reranked scores.

The reranker currently runs locally with `jinaai/jina-reranker-v1-turbo-en`.
Set `RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2` to reproduce the old baseline.
An external LLM reranker is a future experiment and must be evaluated separately.

## Current corpus

- `corpus/SoftSuave-Employee-Handbook-2025.pdf`
- `corpus/Acme-Leave-Policy-2026.md`
- `corpus/Northstar-Remote-Work-Policy-2026.md`

The SoftSuave 2025 handbook is the temporary base until the real 2026 handbook
is uploaded. The Acme and Northstar documents provide independent topics for
testing retrieval, filtering, refusals, and citations.

## Install and run

```powershell
cd D:\Project\HR-Policy
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install streamlit fastembed langchain langchain-core langchain-text-splitters langchain-google-genai langchain-groq langchain-chroma chromadb pymupdf python-dotenv numpy
```

Build the index after adding or changing documents:

```powershell
.\.venv\Scripts\python.exe scripts\01_ingest.py
```

Start the app without activating PowerShell scripts:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501`.

Generation requires a Groq key in `.env`:

```text
GROQ_API_KEY=your_key_here
GROQ_GENERATION_MODEL=openai/gpt-oss-120b
```

Retrieve and local rerank modes do not require a generation API call.

## Uploading a new policy

In the sidebar, expand **Upload a policy document**:

1. Select a `.pdf`, `.md`, or `.markdown` file.
2. Enter its policy ID, region, and effective date.
3. Click **Index uploaded document**.

The file is copied into `corpus/`, metadata is persisted, and both Chroma
collections are updated incrementally. Existing chunks are not re-embedded
unnecessarily. The uploaded document can then be queried in all three modes.

## Search strategies

- **Semantic** — meaning-based retrieval; default.
- **BM25 keyword** — exact names, codes, and section identifiers.
- **Hybrid** — combines semantic and keyword ranks using RRF.

Chunk metadata includes `source_file`, `policy_id`, `region`,
`effective_date`, `section`, and deterministic `chunk_id`.

## Chunking strategies

- **Recursive baseline:** approximately 1,000-character windows with 150-character overlap.
- **Structure-aware:** splits on policy headings, retains section numbers, and re-prepends headings to oversized pieces.

The structure-aware strategy is the preferred chunker for section-numbered
policies. Parent-context expansion is applied at retrieval time, so no second
embedding index is required.

## Week 4 measurement

The recorded experiment compared semantic retrieval with hybrid search:

```text
Semantic baseline: 8/8 hit@3, MRR 1.000
Hybrid search:     8/8 hit@3, MRR 0.917
```

Hybrid did not improve this already-saturated challenge set, so semantic search
remains the default. BM25 and hybrid remain available for inspection.

## Week 5 tracing

Generated answers and refusals are recorded in `output/traces.jsonl`. Each trace
contains a trace ID, timestamp, prompt version, redacted question, retrieved
chunk IDs and scores, model settings, refusal status, and raw output.

Sample and replay traces with:

```powershell
.\.venv\Scripts\python.exe scripts\07_week5_trace.py
```

Week 5 still requires reading a seeded sample of 20 real traces and writing the
observations and taxonomy in `notes.md` and `taxonomy.md`. Those observations
must be based on real outputs; they are not generated automatically.

## Useful commands

```powershell
.\.venv\Scripts\python.exe scripts\ask.py "How many sick-leave days does Acme provide?"
.\.venv\Scripts\python.exe scripts\ask.py --search "Acme carry-forward limit"
.\.venv\Scripts\python.exe scripts\ask.py --search --search-method hybrid "carry-forward limit"
.\.venv\Scripts\python.exe scripts\ask.py --search --rerank local "remote work days"
```

## Important boundaries

- Keep API keys in `.env`; never commit them.
- Do not run two ingest writers simultaneously.
- Clear only generated `chroma/` and trace files when resetting an experiment.
- Preserve the baseline branch so Version 1 and Version 2 remain comparable.
