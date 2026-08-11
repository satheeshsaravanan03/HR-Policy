# Week 3 Practical — Task Set C — Results

HR policy RAG. Six documents ingested under two chunking strategies, measured
search-only against eight questions written from the documents before any
search was run.

**Headline: hit-in-top-5 did not move.** Both chunkers score 8/8, under two
different embedding models. The metric saturated: top-5 over this corpus is too
generous a cutoff to discriminate between these chunkers.

Two things do separate them. Whether a retrieved chunk can be cited to a policy
section at all — **0/8 for the baseline, 7/8 for structure-aware**. And rank-1
accuracy once the embedding model is not strong enough to paper over bad chunk
boundaries — on a 768-dimension model the baseline drops to **6/8 at hit@1 while
structure-aware holds 8/8**, a difference Gemini's 3072-dimension embeddings hide
entirely (section 4).

---

## 1. Corpus and metadata

Every chunk carries `source_file`, `policy_id`, `region`, `effective_date`.
Verified: **0 chunks missing `source_file`** in either collection.

| source_file | policy_id | region | effective_date | date provenance | leave policy |
| --- | --- | --- | --- | --- | --- |
| SoftSuave-Employee-Handbook-2025.pdf | SS-HB-2025 | India | 2025-06-18 | pdf creation date | yes |
| RFSUNY-Leave-Handbook.pdf | RF-LEAVE | New York | 2026-07-28 | pdf creation date | yes |
| UMich-Faculty-Handbook-Ch16-Leaves.pdf | UM-FH-16 | Michigan | 2025-09-11 | pdf creation date | yes |
| Arizona-USM-3-113-Vacation.md | USM-3-113 | Arizona | 2020-01-27 | **stated in policy text** | yes |
| Apple-Business-Conduct-Policy.pdf | APPL-BCP | Global | 2026-02-01 | cover page | no |
| Google-Code-of-Conduct.pdf | GOOG-COC | Global | **unknown** | absent from text and pdf metadata | no |

Google's `effective_date` is recorded as `unknown` rather than guessed. Three
dates fall back to the PDF creation timestamp because those documents state no
effective date; the fallback is labelled rather than presented as published.

Apple and Google are conduct codes, not leave policies. They are in the corpus
deliberately, as retrieval pressure and as the basis for the hardest refusal
case (section 7).

### Chunk counts

| Strategy | Chunks | Documents |
| --- | --- | --- |
| recursive (baseline) | 484 | 6 |
| structure-aware | 433 | 6 |

---

## 2. The eight questions, with known-correct answers

Written from the documents first. Three depend on a row inside an eligibility
table, as required.

| qid | Question | policy_id | section | Known-correct answer | table row |
| --- | --- | --- | --- | --- | --- |
| Q1 | How many casual or sick leaves does an employee at the Bangalore office get in a calendar year? | SS-HB-2025 | 9.2 | 15 | yes |
| Q2 | How many privilege leaves are listed for the Chennai office? | SS-HB-2025 | 9.2 | 3, at company discretion | yes |
| Q3 | Maximum vacation hours accrued per 80-hour pay period for Arizona staff? | USM-3-113 | 3-113.3 | 6.77 hours | yes |
| Q4 | For non-technical staff, how many unused leaves carry over? | SS-HB-2025 | 9.3 | Max nine; technical staff get encashment instead | |
| Q5 | How many accrued vacation hours may Arizona staff carry forward? | USM-3-113 | 3-113.5 | Up to 320 hours, prorated by FTE; excess forfeited | |
| Q6 | Maximum maternity leave, and how much may precede delivery? | SS-HB-2025 | 9.8 | 26 weeks, of which 8 may precede delivery | |
| Q7 | How many university-designated holidays for Michigan faculty, plus floating? | UM-FH-16 | 16.D.1 | Seven, plus one floating | |
| Q8 | Are Research Foundation employees paid for jury duty, and what documentation? | RF-LEAVE | Jury Duty | Full pay; jury duty voucher and advance notice | |

### How a hit is scored

Not by `chunk_id` equality. The two strategies generate different chunk_ids for
the same underlying text by construction, so comparing ids would make the two
numbers incomparable. A hit requires that a top-5 chunk **(a)** comes from the
expected `policy_id` and **(b)** actually contains the answer, matched by a
pre-registered regex. That test is identical for both strategies.

Before measuring, each pattern was verified to resolve to **exactly one chunk
under both strategies** — so the answer is equally available to both, and the
comparison isolates ranking rather than availability.

---

## 3. Hit-in-top-5 — the two numbers

| Chunking strategy | Hit-in-top-5 |
| --- | --- |
| recursive (baseline) | **8/8** |
| structure-aware | **8/8** |

### Per-question record

| qid | expected | recursive | structure-aware |
| --- | --- | --- | --- |
| Q1 | SS-HB-2025 s.9.2 | rank 1 | rank 1 |
| Q2 | SS-HB-2025 s.9.2 | rank 1 | rank 1 |
| Q3 | USM-3-113 s.3-113.3 | rank 1 | rank 1 |
| Q4 | SS-HB-2025 s.9.3 | rank 1 | rank 1 |
| Q5 | USM-3-113 s.3-113.5 | rank 1 | rank 1 |
| Q6 | SS-HB-2025 s.9.8 | rank 1 | rank 1 |
| Q7 | UM-FH-16 s.16.D.1 | rank 1 | **rank 2** |
| Q8 | RF-LEAVE s.Jury Duty | rank 1 | rank 1 |

Full search-only dump for all 8 questions under both strategies:
[`output/02_search_dump.txt`](output/02_search_dump.txt).

Both strategies used the same embedding model (`gemini-embedding-001`, 3072
dims, `RETRIEVAL_DOCUMENT` for indexing and `RETRIEVAL_QUERY` for queries). Only
the chunker differed between the two runs.

---

## 4. Why the metric saturated, and what discriminates instead

hit-in-top-5 is top-5 over a 484-chunk corpus where each answer lives in a
distinctive, well-separated section. That is a generous target, and both
chunkers clear it every time. Tightening the cutoff does not rescue it either:

| Metric | recursive | structure-aware |
| --- | --- | --- |
| hit@1 | 8/8 | 7/8 |
| hit@3 | 8/8 | 8/8 |
| hit@5 | 8/8 | 8/8 |
| MRR | 1.000 | 0.938 |
| **rank-1 chunk is citable to a section** | **0/8** | **7/8** |
| chunks carrying a section number | 0/345 (0.0%) | 125/330 (37.9%) |

On pure ranking the baseline is **equal or marginally better**. That is the
result, not a typo, and section 8 defends the chunking choice on other grounds
rather than pretending otherwise.

Two structural measurements taken before retrieval, over the four
leave-bearing documents:

| | recursive | structure-aware |
| --- | --- | --- |
| Chunks with no heading at all (clause stranded from its section number) | 169/345 — **49.0%** | 24/330 — **7.3%** |
| Chunks spanning more than one section | 146 | 6 |
| Average chunk size | 673 chars | 669 chars |

Average chunk size is effectively identical, so this is the split boundary
rather than chunk volume.

### A second embedding model shows what the first one was hiding

The Gemini result raised an obvious question: is the chunker genuinely not
helping, or is the embedding model strong enough to succeed despite bad chunk
boundaries? So the same 8 questions and the same 2 chunkers were re-run against
a second embedding model — `BAAI/bge-base-en-v1.5`, 768 dimensions, run locally
— in separate Chroma collections. Four cells, one variable changed at a time.

| | recursive | structure-aware |
| --- | --- | --- |
| **hit@1**, Gemini 3072-dim | 8/8 | 7/8 |
| **hit@1**, bge-base 768-dim | **6/8** | **8/8** |
| **MRR**, Gemini 3072-dim | 1.000 | 0.938 |
| **MRR**, bge-base 768-dim | **0.875** | **1.000** |
| hit@5, either model | 8/8 | 8/8 |

On the weaker model the structure-aware chunker wins cleanly: 8/8 against 6/8 at
rank 1, MRR 1.000 against 0.875. Both losses for the baseline are the Arizona
questions (Q3 and Q5), where its fixed-size windows cut across
`3-113.3 Accrual Rates` and `3-113.5 Carry Forward Cap` and the 768-dim model
cannot recover the distinction that the 3072-dim model can.

**So the chunker does help — Gemini's embeddings were masking it.** A model good
enough to find the answer regardless of chunk boundaries hides the value of
drawing better boundaries. That is worth knowing before concluding from a single
strong model that chunking does not matter.

Note this does not rescue hit-in-top-5, which stays 8/8 for both chunkers under
both models. Top-5 over this corpus is simply too generous a cutoff to
discriminate, whatever the embedding model.

Raw output: [`output/02_search_dump_local.txt`](output/02_search_dump_local.txt),
[`output/05_metrics_local.txt`](output/05_metrics_local.txt).

The 37.9% section-number figure deserves a caveat: RF SUNY's handbook has named
headings with **no section numbers anywhere**, contributing 182 chunks that
cannot carry a section however they are split. That is a property of the source
document, not of the chunker. Of the 125 chunks that do carry a section number,
95 come from Soft Suave, 21 from UMich, 7 from Arizona, 2 from RF SUNY.

---

## 5. Metadata filter on region

**Query:** *How many days of unused leave can I carry over to next year?*
**Filter:** `region = "India"`

### Unfiltered (top-5)

| # | score | policy_id | section | region | chunk_id |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.7430 | USM-3-113 | 3-113.5 | **Arizona** | USM-3-113#structure-0005-20070404 |
| 2 | 0.7059 | SS-HB-2025 | 9.3 | India | SS-HB-2025#structure-0066-0a3777d2 |
| 3 | 0.6812 | RF-LEAVE | - | New York | RF-LEAVE#structure-0056-456ca368 |
| 4 | 0.6624 | RF-LEAVE | - | New York | RF-LEAVE#structure-0162-da69e5d6 |
| 5 | 0.6616 | USM-3-113 | 3-113.3 | Arizona | USM-3-113#structure-0003-f2c9bf2e |

### Filtered, `region=India` (top-5)

| # | score | policy_id | section | region | chunk_id |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.7059 | SS-HB-2025 | 9.3 | India | SS-HB-2025#structure-0066-0a3777d2 |
| 2 | 0.6521 | SS-HB-2025 | 9.2 | India | SS-HB-2025#structure-0065-f71f3cbe |
| 3 | 0.6489 | SS-HB-2025 | - | India | SS-HB-2025#structure-0075-9927f910 |
| 4 | 0.6425 | SS-HB-2025 | 9.1 | India | SS-HB-2025#structure-0064-d1ed24f0 |
| 5 | 0.6366 | SS-HB-2025 | 9.7 | India | SS-HB-2025#structure-0071-07a545f1 |

**Top-1 changed: Arizona 3-113.5 → India 9.3.**

This is not a cosmetic reordering. Unfiltered, an employee in India asking about
carry-over is shown **Arizona's 320-hour cap** as the top result. Their actual
entitlement is **nine leaves**. The corpus disagrees across jurisdictions by
design — nine leaves in India, 320 hours in Arizona, per-series schedules in New
York, and no faculty vacation accrual at all in Michigan — so region is the
metadata that decides whether the answer is right or merely confident.

Raw output: [`output/03_region_filter.txt`](output/03_region_filter.txt).

---

## 6. Three cited answers

Full transcripts: [`output/04_generation.txt`](output/04_generation.txt).
All citations resolved; **0 unresolved** across the three answers.

### Q4 — carry-over for non-technical staff

> For the non-technical team, if a certain number of leaves are not availed for
> the respective year, a maximum of nine leaves will be carried over to the next
> year `[CITE: SS-HB-2025#structure-0066-0a3777d2]`.

`SS-HB-2025#structure-0066-0a3777d2` → **SS-HB-2025 section 9.3**, relevance 0.7911. Supporting chunk:

```
### 9.3. LEAVES ENCASHMENT
For Technical teams only, if certain numbers of leaves are not availed for the respective
year, those leaves shall be encashed equivalent to the day's salary.
For Non Technical team, if certain numbers of leaves are not availed for the respective
year then maximum of nine leaves will be carried over to the Next year.
```

### Q1 — Bangalore casual/sick leave

> An employee at the Bangalore Office is entitled to 15 Casual/sick leaves in a
> calendar year `[CITE: SS-HB-2025#structure-0065-f71f3cbe]`.

`SS-HB-2025#structure-0065-f71f3cbe` → **SS-HB-2025 section 9.2**, relevance 0.8020. The
cited chunk contains the eligibility table row `Bangalore Office | 15 | 3`.

### Q7 — Michigan holidays

> Michigan faculty receive seven (7) University-designated holidays, plus one
> floating holiday per twelve month calendar year
> `[CITE: UM-FH-16#structure-0001-81ad5fc3, UM-FH-16#structure-0019-cd71fab0]`.

Both cited chunks resolve, to **16.A** and **16.D.1**. That the model cited both
is itself evidence for the instrument flaw documented in section 9.

---

## 7. Three refusals

All three refused. Refusal is decided **in code, before the model is invoked** —
the model never sees an out-of-corpus question, so refusal is forced rather than
suggested. Three independent gates: a relevance floor (0.30), a
corpus-vocabulary check, and an entity-consistency check.

### R1 — entity mismatch (hard)

> **Q:** What is Apple's parental leave entitlement for new parents?
>
> **A:** I cannot answer this from the indexed policy documents.
> Reason: the question names APPL-BCP, whose policy is indexed, but no APPL-BCP
> chunk discusses parental, leave, entitlement, parents; that topic appears only
> in RF-LEAVE.

Gate: `entity_mismatch`. **Top-1 relevance 0.6827.**

### R2 — topically adjacent (hard)

> **Q:** What is the gratuity payout formula for a Soft Suave employee on resignation?
>
> **A:** I cannot answer this from the indexed policy documents.
> Reason: no document contains formula, gratuity, payout.

Gate: `absent_from_corpus`. **Top-1 relevance 0.7347** — from the correct
document, Soft Suave's own separation policy. Higher than the relevance of
several *correct* answers above.

### R3 — clean absence (baseline)

> **Q:** What is the annual tuition reimbursement cap for employees?
>
> **A:** I cannot answer this from the indexed policy documents.
> Reason: no document contains tuition.

Gate: `absent_from_corpus`. Top-1 relevance 0.6652.

**The point of these numbers:** all three refusals sit at relevance 0.66–0.73,
comfortably above any workable floor. A similarity threshold alone catches
**none** of them. R2 would have produced an invented gratuity formula quoted
from the genuinely correct handbook — which is a legal problem, not a UX one.

---

## 8. Which chunker ships, and why

**Structure-aware ships**, for two reasons, and it is worth being precise about
which evidence supports which.

It does **not** win on hit-in-top-5. That is 8/8 for both, under both embedding
models. Against Gemini's 3072-dimension embeddings it is even marginally *worse*
on ranking — MRR 0.938 against 1.000, one question at rank 2 — and that single
regression turned out to be my own scoring regex rather than the chunker
(section 9). On the strongest model available, the two are equivalent.

It **does** win once the embedding model stops compensating. Against a
768-dimension model the baseline falls to 6/8 at rank 1 and MRR 0.875, while
structure-aware holds 8/8 and 1.000. Both baseline failures are the Arizona
questions, where fixed-size windows cut across `3-113.3` and `3-113.5`. Shipping
a chunker that only works when paired with a top-tier embedding model is a bet on
never changing embedding model, and that is not a bet worth taking.

The second reason is the decisive one, and it holds regardless of model: it is the
only one of the two that can produce a citation an employee or an auditor can act
on. The baseline attaches a section number to
**0 of 345** chunks, so its rank-1 chunk is citable to a section in **0 of 8**
cases against **7 of 8** for structure-aware. It retrieves the right *text* and
then cannot say which clause the text came from. For a system that quotes leave
entitlements, "SoftSuave-Employee-Handbook-2025.pdf, somewhere" is not a
citation — resolving to `section 9.3` is the whole point, and it is what the
grader checks. The baseline also leaves 49.0% of its chunks with no heading at
all and 146 chunks straddling two sections, so a retrieved clause frequently
sits next to a neighbouring section's rule with nothing to distinguish them.

The cost is honest and small: one question moved from rank 1 to rank 2 on the
Gemini backend, and that one turned out to be an artifact of my own scoring regex
rather than the chunker (section 9).

---

## 9. Two retrievals that embarrassed me, with diagnosis

### The region failure

Section 5's unfiltered list is the embarrassing one. The top result for *"how
many days of unused leave can I carry over?"* is **Arizona's 320-hour cap at
0.7430**, ranked above India's own nine-leave rule at 0.7059 — for a corpus
whose Indian handbook is the largest document in it.

**Diagnosis:** the embedding has no notion of jurisdiction. Arizona's
`3-113.5 Carry Forward Cap` is a short, tightly-worded section whose entire
content is about carrying leave forward, so it embeds closer to a carry-over
question than Soft Suave's `9.3 LEAVES ENCASHMENT`, which spends half its words
on encashment for technical staff. The better-written section wins, regardless of
whether it governs the person asking. No amount of chunking fixes this; only the
`region` metadata filter does. This is why requirement 4 is not a nice-to-have —
without it the system confidently quotes the wrong country's law.

### My measurement instrument had a false negative

Q7 is recorded above as **rank 2** for structure-aware. That is wrong, and it is
my fault rather than the chunker's.

The rank-1 chunk was `UM-FH-16` section **16.A**, which states *"seven (7)
University-designated holidays, plus one floating holiday per twelve month
calendar year."* It answers the question correctly. My pre-registered pattern was
`seven University-\s*designated holidays`, which fails on 16.A because `(7)` sits
between "seven" and "University-designated". 16.D.1 phrases it without the
numeral, so only that section matched.

**Diagnosis:** the answer legitimately appears in two sections, and my ground
truth recognised only one. A corrected pattern would make structure-aware 8/8 at
hit@1 and MRR 1.000, tying the baseline exactly.

**I have deliberately not corrected it.** Editing ground truth after seeing
retrieval output is precisely the failure mode the task warns about — the
resulting number would measure my regex-tuning, not my chunker. The
pre-registered number stands at 7/8 hit@1 with the flaw documented. The
generation step in section 6 corroborates the diagnosis independently: the model,
given both chunks, cited **both** 16.A and 16.D.1.

---

## 10. Time reality — no full re-index

The whole handbook was **not** re-indexed. `scripts/01_ingest.py` runs in two
phases: phase 1 indexes the Soft Suave base handbook alone, standing in for a
pre-existing index; phase 2 adds the five new documents and reports the handbook
as skipped. Incremental adds are keyed on `chunk_id`, so an interrupted run
resumes without re-embedding what is already stored.

Phase 2 output confirms it:

```
  [recursive] files already present: 1 | new files: 5
    already indexed, not re-embedded: SoftSuave-Employee-Handbook-2025.pdf
  added 352 chunks from 5 new files
```

This mattered in practice. Gemini's free-tier embed quota is far tighter than
documented — the first run died with `429 RESOURCE_EXHAUSTED` after roughly six
requests against a stated ceiling of 100/minute. The run is now paced at 6s
between batches and waits 65s on a quota error, since the server asks for ~60s
and exponential backoff from one second can never clear a per-minute window.
Total indexing: 917 chunks across both collections, through three quota walls.

---

## 11. Bonus — precision beating completeness

The structure-aware chunker's `9.3 LEAVES ENCASHMENT` chunk is 347 characters
and retrieves at 0.7911 for a carry-over question — tighter and better-ranked
than the baseline's 384-character chunk, which also swallows the `9.4` heading.

But 9.3 says *"maximum of nine leaves will be carried over"* and nothing about
who qualifies. The eligibility conditions live in **9.2**: three years of total
experience and *"continuous employment with the Company for not less than one
year."* A question like *"I joined eight months ago — how many leaves can I
carry over?"* retrieves 9.3 precisely, and 9.3 alone answers "nine" — which is
wrong for that employee, because they fail the continuous-employment condition
stated in a different section.

The tension: the tighter the chunk, the better it ranks and the more confidently
it strands the model without the definitions that qualify it. The baseline's
sloppier, section-straddling chunks accidentally carry more of that context. A
production fix is neither chunker but retrieval of the parent section alongside
the matched clause — precision for ranking, completeness for generation.

---

## 12. Reproducing

```
python scripts/01_ingest.py     # two-phase incremental index, both strategies
python scripts/02_measure.py    # the two X/8 numbers + full per-question dump
python scripts/03_filter.py     # unfiltered vs region-filtered, with scores
python scripts/04_generate.py   # 3 cited answers + 3 refusals
python scripts/05_metrics.py    # hit@1/@3/@5, MRR, citability
python scripts/ask.py           # interactive console
```

| Component | File |
| --- | --- |
| PDF → markdown, font-based heading recovery | `rag/loader.py` |
| Per-document metadata and date provenance | `rag/manifest.py` |
| Both chunking strategies | `rag/chunkers.py` |
| Chroma collections, incremental indexing | `rag/index.py` |
| Search and hit-in-top-5 scoring | `rag/retrieve.py` |
| Cited generation and the three refusal gates | `rag/generate.py` |
| The 8 questions and 3 refusal cases | `rag/questions.py` |
