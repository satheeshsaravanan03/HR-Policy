# Week 11 Presentation - Production, Observability & Cost

## Purpose

Week 11 is about making the HR-policy assistant easier to investigate, measure, and improve. The goal is not to add complexity for its own sake: when an answer is questioned, we should be able to find the request, see what evidence it used, and understand its latency and estimated model cost.

## What we implemented

The first Week 11 implementation is in the existing **Ask (RAG)** mode:

- Each request gets a trace ID so a past answer can be located.
- Request traces record the question, retrieval settings, retrieved policy references, outcome, and errors in `output/traces.jsonl`.
- The app measures retrieval, safety-check, generation, total response time, and first-visible-token latency.
- It captures token usage when the Groq SDK provides it and estimates cost using editable sidebar rates. The defaults match Groq GPT-OSS 120B; users should adjust them if they change models. The displayed cost is an estimate, not a billing total.
- Employee-specific answer and retrieved content are omitted from traces; employee IDs, emails, and demo employee names are redacted.
- Eleven focused Week 11 tracing tests pass, including checks for privacy redaction, cost calculations, streaming compatibility, and false-refusal wording.

## How an Ask request works

```text
Question
   |
Retrieve policy chunks
   |
Run relevance and safety checks
   +-- Insufficient evidence -> refuse and record the trace
   +-- Evidence accepted -> generate a cited answer
                              |
                    Record timings, usage, outcome, and trace ID
```

The trace helps us investigate what happened. It does not by itself prove that an answer is correct; the retrieved policy evidence and citations still need to support the answer.

## Prompt caching, semantic caching, and LiteLLM

**Prompt caching:** We have not built a separate prompt-cache system into the app. Groq may automatically reuse repeated prompt prefixes for supported models. We record cached-token usage only when the provider response exposes it; caching is not guaranteed. This is provider-side behavior, not an answer saved by our application.

**Semantic answer caching:** This is not implemented yet. It would let a similar question reuse a previous answer, reducing model calls and potentially improving response time. We deferred it because similarity alone is unsafe for HR answers: a changed region, policy version, employee, or retrieved evidence could make an old answer wrong. A safe implementation needs to verify the policy evidence and relevant settings, avoid sharing employee-specific answers, invalidate entries when documents change, and measure quality as well as cost.

**LiteLLM and model routing:** Ask mode currently calls Groq directly. LiteLLM is present through the CrewAI dependency, but it is not the Ask-mode router. We have not added routing or automatic provider fallbacks because they would add another behavior to measure and troubleshoot before the current single-provider traces and cost measurements are complete.

**Fine-tuning:** We did not fine-tune a model. Fine-tuning is not the first fix for wrong or missing policy evidence; retrieval, grounding, and regression tests should be addressed first. It would also require a reviewed, suitable training set.

## Why this order makes sense

Before trying to optimize cost or introduce cached answers, we need reliable visibility into individual requests. Traces and timings give us a baseline and help distinguish retrieval delays from generation delays. We also handle HR and employee-specific data, so a cache must not return a plausible answer from the wrong policy or person. Measuring first and adding safeguards before caching keeps the learning experiment useful without treating a lower cost number as proof of a better system.

## What remains for Week 11

- Complete the support drill: find the planted incorrect Acme notice-period answer from a vague report and document its trace and evidence.
- Add the notice-period failure as an end-to-end regression case.
- If proceeding with semantic caching, run the same fixed question set before and after it; report quality, cache hits, model calls, latency, tokens, and estimated cost.
- Write the short 10x traffic note: likely bottlenecks and practical mitigations.

These items are not represented as complete yet. Provider prompt caching is automatic when available; it is not a substitute for the semantic-cache experiment.
