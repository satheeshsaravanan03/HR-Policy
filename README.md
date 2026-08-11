# HR Policy RAG — how to run it

Week 3 Task Set C. Results and analysis are in [results.md](results.md).

## Embeddings: free and unlimited by default

Retrieval runs on a **local** embedding model (`BAAI/bge-base-en-v1.5` via
fastembed). No API key, no quota, no per-day cap, works offline. This is the
default, so nothing stalls mid-session.

The Gemini backend is still available and still indexed. It produced the numbers
in `results.md`, and it has a hard free-tier limit of **1000 embed requests per
day** — enough to build the index and measure once, not enough to keep testing.

```powershell
# free and unlimited (default)
python scripts/ask.py "how many sick leaves for Bangalore?"

# reproduce the graded numbers
$env:EMBED_BACKEND = "gemini"; python scripts/02_measure.py
$env:EMBED_BACKEND = "local"          # switch back
```

The two backends live in separate Chroma collections (`structure` vs
`structure_local`) because their vectors have different dimensionality — 3072 and
768 — and are not interchangeable. Switching backends never invalidates the other
index.

Generation still uses Gemini Flash, which has its own separate quota. Retrieval
never touches the network on the local backend, so `--search` mode is entirely
free and offline.

## Prerequisites

Already done on this machine, listed for a fresh setup:

```powershell
python -m pip install langchain langchain-text-splitters langchain-google-genai langchain-chroma chromadb pymupdf python-dotenv
```

A free Gemini API key from https://aistudio.google.com/apikey, placed in `.env`:

```
GOOGLE_API_KEY=your_key_here
```

The index is already built (`chroma/`, 21 MB), so you can query immediately.
You only need `01_ingest.py` if you add documents or delete `chroma/`.

## Try it in the browser (easiest)

```powershell
cd e:\Tech_Lab\HR-policy
streamlit run app.py
```

Then open http://localhost:8501. Ctrl+C in the terminal to stop it.

Three modes in the sidebar:

- **Ask** — retrieve, gate, then generate. Shows the answer, a citation table
  with a resolves yes/no per citation, and every retrieved chunk with scores.
- **Search only** — retrieval with no generation, so it spends no generation quota.
- **Compare both chunkers** — the same question against both strategies side by
  side. The `section` column is the whole argument: `structure` names the policy
  section, `recursive` shows a dash.

The two tabs above the input are one-click presets: the 8 known-answer questions
(each showing its expected `policy_id` + section, so you can check the answer
against ground truth) and the 3 that must be refused.

Streamlit's install upgrades `starlette` past the pin `fastapi` wants, so pip
prints a dependency conflict. It is latent — chromadb's persistent client does
not use that path, and retrieval was verified working afterwards.

## Or use the terminal console

```powershell
cd e:\Tech_Lab\HR-policy
python scripts/ask.py
```

Then type questions at the `>` prompt. **Type only the question** — not the whole
command. Type `quit` to exit.

Questions that should answer, with a citation:

```
how many privilege leaves for Chennai office?
how many casual or sick leaves does the Bangalore office get?
maximum maternity leave and how much before delivery?
how many holidays do Michigan faculty get?
how much vacation can Arizona staff carry forward?
```

Questions that should **refuse** — the more interesting test:

```
What is Apple's parental leave entitlement?
What is the gratuity payout formula on resignation?
What is the annual tuition reimbursement cap?
```

Watch the `score` column on the refusals. It stays around 0.65–0.73, which is
why a similarity threshold alone would not catch them.

## One-shot queries

```powershell
python scripts/ask.py "how many sick leaves does Bangalore office get?"

# retrieval only -- no generation, spends no generation quota
python scripts/ask.py --search "carry over cap for unused leave"

# see the actual chunk text
python scripts/ask.py --search --show-text "carry over cap"

# retrieve more than 5
python scripts/ask.py --search -k 10 "maternity leave"
```

### Compare the two chunkers on the same question

```powershell
python scripts/ask.py --search --strategy recursive "carry over cap for unused leave"
python scripts/ask.py --search --strategy structure "carry over cap for unused leave"
```

The `section` column is the thing to watch: `structure` reports `9.3`, while
`recursive` reports `-` for every result. Both find the right text; only one can
cite where it came from.

### See the region filter change the answer

```powershell
python scripts/ask.py --search "how many days of unused leave can I carry over?"
python scripts/ask.py --search --region India "how many days of unused leave can I carry over?"
```

Unfiltered, the top result is Arizona's 320-hour cap. Filtered to India, it is
Soft Suave section 9.3 and its nine-leave cap.

## Reproduce the graded deliverables

Each writes to `output/`. Run in order.

| Command | Produces |
| --- | --- |
| `python scripts/01_ingest.py` | Two-phase incremental index (only needed after adding documents) |
| `python scripts/02_measure.py` | The two hit-in-top-5 numbers + full per-question dump |
| `python scripts/03_filter.py` | Unfiltered vs region-filtered lists, with scores |
| `python scripts/04_generate.py` | 3 cited answers + 3 refusal transcripts |
| `python scripts/05_metrics.py` | hit@1 / @3 / @5, MRR, citability |

## Notes

**Free-tier rate limits.** Gemini's free embed quota is tight. The measurement
scripts each spend about 16 query embeddings and are fine, but re-running
`01_ingest.py` from scratch embeds 917 chunks and takes several minutes, pausing
65s whenever it hits the quota. It resumes where it left off, so an interrupted
run is safe to re-run.

**`PL�s` in output is not a bug.** The corpus uses proper `U+2019` quotes; the
Windows console just cannot render them. Verified: zero replacement characters in
the extracted text.

**Do not run two writers at once.** `01_ingest.py` writes to the Chroma store;
running two copies concurrently will conflict. Queries are read-only and safe.

## Layout

```
corpus/          the 6 policy documents
rag/             loader, manifest, chunkers, index, retrieve, generate, questions
scripts/         01_ingest -> 05_metrics, plus ask.py
output/          captured runs for the write-up
chroma/          the vector store (gitignored)
results.md       the graded write-up
```
