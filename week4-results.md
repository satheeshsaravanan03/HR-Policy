# Week 4 — Retrieval Debugging: Hybrid Search Experiment

## One change tested

The baseline is the existing semantic/vector search over the `structure_local`
Chroma collection. The only change is **hybrid retrieval**: BM25 keyword search
over the same stored chunks plus semantic search, combined using Reciprocal Rank
Fusion (RRF, `k=60`). No reranker, query rewriting, HyDE, MMR, chunking change,
or new embedding model was added.

## Inspection and failure separation

| Case | Evidence | Classification | What hybrid can do |
| --- | --- | --- | --- |
| `W4-G1`: employee joined eight months ago; carry forward | A correct answer needs Soft Suave `9.2` (eligibility) and `9.3` (nine-leave cap). The retrieved context did not contain both. | Generation/context failure | Not reliably fixed: it needs parent/related-section retrieval. |
| Regionless carry-forward question | Arizona's 320-hour rule can outrank India's nine-leave rule without `region=India`. | Retrieval/jurisdiction failure | Not fixed by BM25/RRF: the user must supply region or the app must infer/request it. |
| Exact identifiers, office names, policy codes | BM25 ranks literal matches such as `USM 3-113.5`, `Bangalore Office`, and `jury duty voucher` directly. | Retrieval capability under test | Can promote exact matches where semantic rank is weak. |

## Reproducible metric

Run:

```powershell
python scripts/06_week4_evaluate.py
```

The pre-registered eight-case set is in `rag/week4.py`. A hit requires the
expected `policy_id` **and** a chunk matching the expected evidence pattern in
the first three results. The observed run on the local BAAI/BGE index was:

| Method | hit-rate@3 | MRR |
| --- | --- | --- |
| Semantic baseline | 8/8 (100%) | 1.000 |
| Hybrid BM25 + semantic RRF | 8/8 (100%) | 0.917 |

## Decision

Hybrid search did **not** improve hit-rate@3 on this small, already-saturated
corpus and reduced MRR on the challenge set. Therefore the application keeps
**semantic search as its default**. The BM25 and hybrid modes remain selectable
for inspection and further evaluation; they are not presented as a proven
production improvement.

This is an evidence-based result, not a failure hidden by metric selection.
The next useful experiment is either a larger held-out failure set with genuine
semantic misses, or a separate parent-context retrieval change for `W4-G1`.
