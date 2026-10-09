# Week 11 Plan — Production: Observability, Cost & the Failure-to-Test Loop

**Module:** 6 — Production & Capstone  
**Track:** C — HR policy  
**Format:** Build week  
**Goal:** Make the app's answers traceable, measure and reduce cost per request, and turn a wrong notice-period answer into a regression test.

## Deliverable

- Per-request traces with a trace ID, answer, retrieved policy sources, outcome, and useful request settings.
- Per-step timing and token usage, with estimated cost when provider pricing is configured.
- A completed support drill that finds the planted wrong Acme notice-period answer from a vague complaint.
- A measured before-and-after cost improvement using semantic caching.
- A regression test guarding against the notice-period failure.
- A short plan for what would break first at 10× traffic.

## Build sequence

### 1. Inspect the existing request and logging flow

Follow an HR question through retrieval, safety checks, answer generation, and citation handling. Note what the existing trace records and identify missing fields, timings, and failure information.

### 2. Add per-request traces

Assign each request a trace ID and record its timestamp, question, model and retrieval settings, retrieved policy sources, final answer, and outcome. Redact employee IDs and email addresses before saving logs. Record separate timings for retrieval, generation, and checks. Capture token usage and estimated cost when available; mark cost unavailable when pricing is not configured.

### 3. Complete the support drill

Use the complaint: “The assistant gave a permanent Acme employee the wrong notice period.” Search the logs to locate the matching request and trace ID. Compare its answer and cited source with the Acme policy. The policy states **30 calendar days**; the planted answer says **60 calendar days**. Record the likely cause: evidence from another policy was applied to Acme.

### 4. Measure cost per request

Run a fixed set of representative HR questions. Record latency, input and output tokens, and estimated cost for each request. Keep model, questions, retrieval settings, and configured token prices the same for the before-and-after comparison.

### 5. Add and measure semantic caching

Cache answers for semantically similar questions so paraphrases can reuse a previous response and avoid another generation call. Include the applicable policy evidence and model or prompt version in cache validation; invalidate an entry when the underlying evidence changes. Compare uncached and cached runs using cache-hit rate, model calls, tokens, latency, and estimated cost. Check that the cached answer remains grounded in the right policy.

### 6. Turn the failure into a regression test

Add a test for the permanent Acme employee notice-period question. Guard the expected **30 calendar days** answer against the planted **60 calendar days** error, and check that the evidence belongs to Acme. Run the project's relevant evaluation checks and retain the failing example as a permanent case.

### 7. Write a 10× traffic note

Estimate what will bottleneck first as request volume grows, such as provider rate limits, generation latency, or trace storage. Write one practical mitigation for each likely bottleneck.

## Topics to study

- **Prompt caching:** understand provider-side reuse of repeated prompt prefixes; use it only if the configured model/provider supports it and the savings can be measured.
- **Semantic caching:** implement this as the week's measured cost improvement, with policy-evidence-aware invalidation.
- **Model routing, fallbacks, and rate limits:** document how easy questions could use a cheaper model and how requests behave when a provider is unavailable or throttles traffic.
- **Fine-tuning:** treat as a last resort. First fix retrieval, grounding, routing, and regression coverage; fine-tuning does not correct the wrong policy evidence by itself.
- **The data flywheel:** use reviewed failures to create evaluation and regression cases, with human checks before treating logged answers as ground truth.

## Friday mentor review

Demonstrate that you can find the planted answer from the vague complaint, inspect per-step timings and token usage, compare cost before and after semantic caching, and run the regression test. This review is informal and ungraded.

## Week 12 evaluation evidence

Bring the trace log, support-drill findings, before-and-after cost report, regression test result, and 10× traffic note.
