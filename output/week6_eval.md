# Week 6 Evaluation

Rule-based assertions run before any optional LLM judge.

| Version | Overall | Answerable | Refusal | Regression |
|---|---:|---:|---:|---:|
| baseline | 96.4% | 100.0% | 100.0% | 83.3% |
| improved | 98.2% | 100.0% | 100.0% | 91.7% |

## Per-case assertions

| Version | Set | ID | Score | Assertions |
|---|---|---|---:|---|
| baseline | answerable | Q1 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | answerable | Q2 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | answerable | Q3 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | answerable | Q4 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | answerable | Q5 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | answerable | Q6 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | answerable | Q7 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | answerable | Q8 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | refusal | R1 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | refusal | R2 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | refusal | R3 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | regression | REG-01 | 50.0% | retrieved=PASS, expected_answer_found=FAIL, expected_refusal=FAIL, citation_source_available=PASS |
| baseline | regression | REG-02 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| baseline | regression | REG-03 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | answerable | Q1 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | answerable | Q2 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | answerable | Q3 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | answerable | Q4 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | answerable | Q5 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | answerable | Q6 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | answerable | Q7 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | answerable | Q8 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | refusal | R1 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | refusal | R2 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | refusal | R3 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | regression | REG-01 | 75.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=FAIL, citation_source_available=PASS |
| improved | regression | REG-02 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |
| improved | regression | REG-03 | 100.0% | retrieved=PASS, expected_answer_found=PASS, expected_refusal=PASS, citation_source_available=PASS |

The improved configuration is hybrid RRF retrieval with the local cross-encoder reranker; the baseline is semantic retrieval without reranking.
