"""Five predefined workflows for HR Policy RAG (Week 7).

Each workflow executes a deterministic sequence of steps and returns a
standard structured dictionary:

{
    "workflow": str,
    "status": "success" | "refused" | "needs_input" | "error",
    "steps": [{"name": str, "status": str, "details": str}],
    "hits": list[Hit],
    "citations": list[Citation],
    "answer_context": str,
    "missing_information": list[str],
    "evidence_ok": bool,
    "reason": str,
}
"""

from __future__ import annotations

import re
from typing import Any

from .generate import (
    ORGANISATIONS,
    RELEVANCE_FLOOR,
    STOPWORDS,
    Citation,
    _citations,
    _context,
    corpus_vocabulary,
    refusal_check,
)
from .manifest import DOCUMENTS
from .employee_data import (
    calculate_employee_case,
    format_employee_answer,
    format_employee_comparison,
    identifier_from_query,
    identifiers_from_query,
    lookup_record,
)
from .retrieve import (
    HYBRID,
    RERANK_LOCAL,
    Hit,
    normalize_query,
    search,
    understand_query,
)

# Comparison framing words to exclude from topical vocabulary checks in comparisons
COMPARISON_FRAMING = frozenset(
    {"compare", "contrast", "difference", "differences", "versus", "vs", "between", "similarities"}
)

# Extended organization mapping including current corpus entities
ALL_ORGS: dict[str, str] = dict(ORGANISATIONS)
ALL_ORGS.update({
    "acme": "ACME-LEAVE-2026",
    "northstar": "NORTHSTAR-REMOTE-2026",
    "softsuave": "SS-HB-2025",
    "soft suave": "SS-HB-2025",
})
for _doc in DOCUMENTS:
    _stem = _doc.policy_id.split("-")[0].lower()
    ALL_ORGS[_stem] = _doc.policy_id


def _step(name: str, status: str, details: str) -> dict[str, str]:
    return {"name": name, "status": status, "details": details}


# ----------------------------------------------------------------------
# Workflow 1: Policy lookup
# ----------------------------------------------------------------------
def policy_lookup(
    query: str,
    *,
    strategy: str = "structure",
    top_k: int = 5,
    region: str | None = None,
    method: str = HYBRID,
    rerank: str = RERANK_LOCAL,
) -> dict[str, Any]:
    """Execute a structured policy lookup for a single query.

    Steps:
      1. Normalize query
      2. Infer metadata (policy ID, region)
      3. Hybrid retrieval + cross-encoder rerank
      4. Expand parent-context
      5. Validate evidence via refusal gates
    """
    steps: list[dict[str, str]] = []

    # 1. Normalize query
    norm_query = normalize_query(query)
    steps.append(_step("normalize_query", "success", f"Normalized query: {norm_query}"))

    # 2. Infer metadata
    inferred_policy, inferred_region = understand_query(norm_query)
    target_region = region or inferred_region
    steps.append(
        _step(
            "infer_metadata",
            "success",
            f"Inferred policy_id={inferred_policy or 'none'}, region={target_region or 'none'}",
        )
    )

    # 3. Retrieval & Rerank (Context expansion is handled within search for structure chunking)
    hits = search(
        strategy,
        norm_query,
        top_k=top_k,
        region=target_region,
        method=method,
        rerank=rerank,
    )
    steps.append(
        _step(
            "hybrid_retrieval_rerank",
            "success",
            f"Retrieved and reranked {len(hits)} chunk(s) via {method} (rerank={rerank})",
        )
    )

    # 4. Context expansion verification
    expanded_count = sum(1 for h in hits if h.context_content)
    steps.append(
        _step(
            "expand_context",
            "success",
            f"Parent-context available for {expanded_count}/{len(hits)} chunk(s)",
        )
    )

    # 5. Validate evidence
    refuse, gate, reason = refusal_check(norm_query, hits)
    if refuse:
        steps.append(_step("validate_evidence", "refused", f"Gate fired: {gate} — {reason}"))
        return {
            "workflow": "policy_lookup",
            "status": "refused",
            "steps": steps,
            "hits": hits,
            "citations": [],
            "answer_context": "",
            "missing_information": [],
            "evidence_ok": False,
            "reason": reason,
        }

    steps.append(_step("validate_evidence", "success", "Evidence passed all refusal gates"))
    return {
        "workflow": "policy_lookup",
        "status": "success",
        "steps": steps,
        "hits": hits,
        "citations": [],
        "answer_context": _context(hits),
        "missing_information": [],
        "evidence_ok": True,
        "reason": "",
    }


# ----------------------------------------------------------------------
# Workflow 2: Policy comparison
# ----------------------------------------------------------------------
def policy_comparison(
    query: str,
    *,
    strategy: str = "structure",
    top_k: int = 6,
) -> dict[str, Any]:
    """Retrieve and align evidence from multiple policies or regions.

    Steps:
      1. Identify distinct policies or regions
      2. Retrieve evidence separately for each policy/region
      3. Align and interleave relevant chunks
      4. Validate comparative evidence
      5. Prepare unified comparison context
    """
    steps: list[dict[str, str]] = []
    norm_query = normalize_query(query)
    lowered = norm_query.lower()

    # 1. Identify distinct policies or organizations mentioned
    detected_orgs: list[str] = []
    for org_name, policy_id in ALL_ORGS.items():
        if org_name in lowered and policy_id not in detected_orgs:
            detected_orgs.append(policy_id)

    # Check distinct regions
    detected_regions: list[str] = []
    for doc in DOCUMENTS:
        if doc.region.lower() in lowered and doc.region not in detected_regions:
            detected_regions.append(doc.region)

    entities = detected_orgs or detected_regions
    steps.append(
        _step(
            "identify_targets",
            "success",
            f"Identified comparison entities: {entities if entities else 'generic multi-policy comparison'}",
        )
    )

    # 2. Retrieve per entity or broad search if none isolated
    combined_hits: list[Hit] = []

    # Strip comparison framing words for targeted search
    search_query_words = [w for w in re.findall(r"[a-z0-9]+", lowered) if w not in COMPARISON_FRAMING]
    search_topic = " ".join(search_query_words)

    if len(entities) >= 2:
        k_each = max(2, top_k // len(entities))
        for entity in entities:
            if entity in detected_regions:
                hits_entity = search(
                    strategy, search_topic, top_k=k_each, region=entity, method=HYBRID, rerank=RERANK_LOCAL
                )
            else:
                hits_all = search(strategy, f"{entity} {search_topic}", top_k=10, method=HYBRID, rerank=RERANK_LOCAL)
                hits_entity = [h for h in hits_all if h.policy_id == entity][:k_each]
                if not hits_entity:
                    hits_entity = hits_all[:k_each]
            combined_hits.extend(hits_entity)
            steps.append(
                _step("retrieve_policy", "success", f"Retrieved {len(hits_entity)} chunks for '{entity}'")
            )
    else:
        # Broad retrieval across policies
        broad_hits = search(strategy, search_topic, top_k=top_k * 2, method=HYBRID, rerank=RERANK_LOCAL)
        seen_policies: set[str] = set()
        for h in broad_hits:
            if h.policy_id not in seen_policies:
                seen_policies.add(h.policy_id)
            combined_hits.append(h)
            if len(combined_hits) >= top_k:
                break
        steps.append(
            _step(
                "retrieve_policy",
                "success",
                f"Retrieved {len(combined_hits)} chunks across policies: {list(seen_policies)}",
            )
        )

    # 3. Align relevant sections
    unique_chunk_ids: set[str] = set()
    aligned_hits: list[Hit] = []
    for h in combined_hits:
        if h.chunk_id not in unique_chunk_ids:
            unique_chunk_ids.add(h.chunk_id)
            aligned_hits.append(h)
    aligned_hits = aligned_hits[:top_k]
    steps.append(_step("align_sections", "success", f"Aligned {len(aligned_hits)} unique comparison chunks"))

    # 4. Validate evidence
    if not aligned_hits:
        steps.append(_step("validate_evidence", "refused", "No comparative evidence found"))
        return {
            "workflow": "policy_comparison",
            "status": "refused",
            "steps": steps,
            "hits": [],
            "citations": [],
            "answer_context": "",
            "missing_information": [],
            "evidence_ok": False,
            "reason": "No comparative evidence retrieved",
        }

    # Relevance check
    semantic_scores = [h.semantic_score for h in aligned_hits if h.semantic_score is not None]
    top_score = max(semantic_scores) if semantic_scores else aligned_hits[0].score
    if top_score < RELEVANCE_FLOOR:
        reason = f"top relevance {top_score:.3f} is below the {RELEVANCE_FLOOR} floor"
        steps.append(_step("validate_evidence", "refused", f"Relevance floor triggered: {reason}"))
        return {
            "workflow": "policy_comparison",
            "status": "refused",
            "steps": steps,
            "hits": aligned_hits,
            "citations": [],
            "answer_context": "",
            "missing_information": [],
            "evidence_ok": False,
            "reason": reason,
        }

    # Vocabulary check on salient topic words
    vocab = corpus_vocabulary()
    topic_terms = [
        w for w in re.findall(r"[a-z]{4,}", lowered)
        if w not in STOPWORDS and w not in COMPARISON_FRAMING and w not in ALL_ORGS
    ]
    unknown = [w for w in topic_terms if w not in vocab]
    if unknown:
        reason = f"no document contains {', '.join(sorted(set(unknown)))}"
        steps.append(_step("validate_evidence", "refused", f"Absent from corpus: {reason}"))
        return {
            "workflow": "policy_comparison",
            "status": "refused",
            "steps": steps,
            "hits": aligned_hits,
            "citations": [],
            "answer_context": "",
            "missing_information": [],
            "evidence_ok": False,
            "reason": reason,
        }

    steps.append(_step("validate_evidence", "success", "Comparison evidence validated"))
    return {
        "workflow": "policy_comparison",
        "status": "success",
        "steps": steps,
        "hits": aligned_hits,
        "citations": [],
        "answer_context": _context(aligned_hits),
        "missing_information": [],
        "evidence_ok": True,
        "reason": "",
    }


# ----------------------------------------------------------------------
# Workflow 3: Evidence validation
# ----------------------------------------------------------------------
def evidence_validation(
    query: str,
    *,
    hits: list[Hit] | None = None,
    strategy: str = "structure",
    top_k: int = 5,
) -> dict[str, Any]:
    """Validate whether retrieved evidence is sufficient and safe to answer.

    Steps:
      1. Inspect retrieved chunks
      2. Check policy IDs and section numbers
      3. Check citation support and refusal gates
      4. Approve answer or refuse
    """
    steps: list[dict[str, str]] = []
    norm_query = normalize_query(query)

    if hits is None:
        hits = search(strategy, norm_query, top_k=top_k, method=HYBRID, rerank=RERANK_LOCAL)
        steps.append(_step("fetch_candidates", "success", f"Fetched {len(hits)} candidates for validation"))

    # 1. Inspect chunks
    if not hits:
        steps.append(_step("inspect_chunks", "refused", "No chunks available to validate"))
        return {
            "workflow": "evidence_validation",
            "status": "refused",
            "steps": steps,
            "hits": [],
            "citations": [],
            "answer_context": "",
            "missing_information": [],
            "evidence_ok": False,
            "reason": "retrieval returned nothing",
        }

    steps.append(_step("inspect_chunks", "success", f"Inspecting {len(hits)} chunk(s)"))

    # 2. Check policy ID & section presence
    valid_sections = [h for h in hits if h.policy_id and h.section]
    steps.append(
        _step(
            "check_metadata",
            "success",
            f"{len(valid_sections)}/{len(hits)} chunks carry explicit policy IDs and sections",
        )
    )

    # 3. Refusal gates
    refuse, gate, reason = refusal_check(norm_query, hits)
    if refuse:
        steps.append(_step("gate_check", "refused", f"Failed gate '{gate}': {reason}"))
        return {
            "workflow": "evidence_validation",
            "status": "refused",
            "steps": steps,
            "hits": hits,
            "citations": [],
            "answer_context": "",
            "missing_information": [],
            "evidence_ok": False,
            "reason": reason,
        }

    steps.append(_step("gate_check", "success", "All evidence gates passed"))
    return {
        "workflow": "evidence_validation",
        "status": "success",
        "steps": steps,
        "hits": hits,
        "citations": [],
        "answer_context": _context(hits),
        "missing_information": [],
        "evidence_ok": True,
        "reason": "",
    }


# ----------------------------------------------------------------------
# Workflow 4: Policy applicability
# ----------------------------------------------------------------------
def policy_applicability(
    query: str,
    *,
    employee_info: dict[str, Any] | None = None,
    strategy: str = "structure",
    top_k: int = 5,
) -> dict[str, Any]:
    """Determine rule applicability for specific employee circumstances.

    Identifies what employee information is missing and asks for clarification
    rather than guessing employee eligibility facts.
    """
    steps: list[dict[str, str]] = []
    norm_query = normalize_query(query)
    lowered = norm_query.lower()
    info = employee_info or {}

    # Check corpus vocabulary first so unsupported questions refuse immediately
    vocab = corpus_vocabulary()
    terms = [
        w for w in re.findall(r"[a-z]{4,}", lowered)
        if w not in STOPWORDS and w not in ALL_ORGS and w not in COMPARISON_FRAMING
    ]
    unknown = [w for w in terms if w not in vocab]
    if unknown:
        reason = f"no document contains {', '.join(sorted(set(unknown)))}"
        steps.append(_step("vocabulary_check", "refused", f"Absent from corpus: {reason}"))
        return {
            "workflow": "policy_applicability",
            "status": "refused",
            "steps": steps,
            "hits": [],
            "citations": [],
            "answer_context": "",
            "missing_information": [],
            "evidence_ok": False,
            "reason": reason,
        }

    # Extract conditions
    region = info.get("region") or info.get("location")
    if not region:
        for d in DOCUMENTS:
            if d.region.lower() in lowered:
                region = d.region
                break
        if "chennai" in lowered or "bangalore" in lowered:
            region = "India"

    emp_type = info.get("employment_type")
    if not emp_type:
        for t in ("full-time", "full time", "permanent", "probation", "contractor", "intern", "part-time", "part time"):
            if t in lowered:
                emp_type = t
                break

    tenure = info.get("tenure")
    if not tenure:
        match = re.search(r"\b(\d+\s*(?:year|month|day)s?)\b", lowered)
        if match:
            tenure = match.group(1)

    steps.append(
        _step(
            "extract_conditions",
            "success",
            f"Extracted: region={region or 'unknown'}, employment_type={emp_type or 'unknown'}, tenure={tenure or 'unknown'}",
        )
    )

    # Detect if query is asking for eligibility where conditions are absent
    is_eligibility_query = any(
        phrase in lowered
        for phrase in (
            "am i eligible",
            "am i entitled",
            "can i work",
            "can an employee",
            "how many days can i",
            "do i qualify",
            "eligibility",
        )
    )

    missing: list[str] = []
    if is_eligibility_query:
        if not region and not any(org in lowered for org in ALL_ORGS):
            missing.append("office location / employer organization")
        if not emp_type and "probation" not in lowered and "permanent" not in lowered:
            missing.append("employment type (e.g. permanent, probation, contractor)")

    if missing:
        steps.append(
            _step(
                "check_missing_information",
                "needs_input",
                f"Missing employee facts: {', '.join(missing)}",
            )
        )
        return {
            "workflow": "policy_applicability",
            "status": "needs_input",
            "steps": steps,
            "hits": [],
            "citations": [],
            "answer_context": "",
            "missing_information": missing,
            "evidence_ok": False,
            "reason": f"Eligibility cannot be determined without: {', '.join(missing)}.",
        }

    steps.append(_step("check_missing_information", "success", "All required conditions provided"))

    # Retrieve applicable sections
    hits = search(strategy, norm_query, top_k=top_k, region=region, method=HYBRID, rerank=RERANK_LOCAL)
    steps.append(_step("retrieve_applicable_rules", "success", f"Retrieved {len(hits)} applicable rule chunks"))

    refuse, gate, reason = refusal_check(norm_query, hits)
    if refuse:
        steps.append(_step("validate_rules", "refused", f"Rule validation refused: {gate} — {reason}"))
        return {
            "workflow": "policy_applicability",
            "status": "refused",
            "steps": steps,
            "hits": hits,
            "citations": [],
            "answer_context": "",
            "missing_information": [],
            "evidence_ok": False,
            "reason": reason,
        }

    steps.append(_step("validate_rules", "success", "Applicability rules validated"))
    return {
        "workflow": "policy_applicability",
        "status": "success",
        "steps": steps,
        "hits": hits,
        "citations": [],
        "answer_context": _context(hits),
        "missing_information": [],
        "evidence_ok": True,
        "reason": "",
    }


# ----------------------------------------------------------------------
# Workflow 6: Employee case lookup and calculation
# ----------------------------------------------------------------------
def employee_case(
    query: str,
    *,
    identifier: str | None = None,
) -> dict[str, Any]:
    """Resolve an editable employee record and calculate requested values."""
    steps: list[dict[str, str]] = []
    key = identifier or identifier_from_query(query)
    if not key:
        return {
            "workflow": "employee_case", "status": "needs_input", "steps": [
                _step("identify_employee", "needs_input", "Provide an employee ID, customer ID, or email")
            ], "hits": [], "citations": [], "answer_context": "",
            "missing_information": ["employee_id, customer_id, or email"],
            "evidence_ok": False, "reason": "employee identifier is required",
            "record": None, "calculation": None, "answer": "",
        }
    steps.append(_step("identify_employee", "success", f"Looking up structured record for {key}"))
    record = lookup_record(key)
    if record is None:
        return {
            "workflow": "employee_case", "status": "needs_input", "steps": steps + [
                _step("lookup_employee", "needs_input", f"No record found for {key}")
            ], "hits": [], "citations": [], "answer_context": "",
            "missing_information": [f"a valid employee record for {key}"],
            "evidence_ok": False, "reason": "employee record not found",
            "record": None, "calculation": None, "answer": "",
        }
    steps.append(_step("lookup_employee", "success", f"Loaded {record['employee_id']} from local structured data"))
    calculation = calculate_employee_case(record, query)
    steps.append(_step("calculate_employee_values", "success", f"Calculated {', '.join(calculation['requested'])}"))
    answer = format_employee_answer(record, calculation)
    steps.append(_step("return_structured_result", "success", "Returned calculated values without embedding employee data"))
    return {
        "workflow": "employee_case", "status": "success", "steps": steps,
        "hits": [], "citations": [], "answer_context": "", "missing_information": [],
        "evidence_ok": True, "reason": "", "record": record,
        "calculation": calculation, "answer": answer,
    }


def employee_policy_case(query: str, *, strategy: str = "structure", top_k: int = 5) -> dict[str, Any]:
    """Combine a structured employee record with policy evidence from Qdrant."""
    key = identifier_from_query(query)
    if not key:
        return {"workflow": "employee_policy_case", "status": "needs_input", "steps": [
            _step("identify_employee", "needs_input", "Provide an employee ID, customer ID, or email")
        ], "hits": [], "citations": [], "answer_context": "", "missing_information": ["employee identifier"],
        "evidence_ok": False, "reason": "employee identifier is required", "record": None, "answer": ""}
    record = lookup_record(key)
    if not record:
        return {"workflow": "employee_policy_case", "status": "needs_input", "steps": [
            _step("lookup_employee", "needs_input", f"No record found for {key}")
        ], "hits": [], "citations": [], "answer_context": "", "missing_information": [f"valid employee record for {key}"],
        "evidence_ok": False, "reason": "employee record not found", "record": None, "answer": ""}
    hits = search(strategy, query, top_k=top_k, method=HYBRID, rerank=RERANK_LOCAL, policy_id=record["policy_id"])
    if any(term in query.lower() for term in ("what policy applies", "which policy applies", "applicable policy")):
        calculation = {"requested": [], "values": {}}
        answer = f"Employee `{record['employee_id']}` is associated with policy `{record['policy_id']}` for region `{record['region']}`."
    else:
        calculation = calculate_employee_case(record, query)
        answer = format_employee_answer(record, calculation)
    if hits:
        sections = sorted({h.section for h in hits if h.section})
        answer += f"\nPolicy evidence retrieved from Qdrant: sections {', '.join(sections) if sections else 'document text'}."
    if not hits:
        answer += f"\n\nPolicy evidence for `{record['policy_id']}` was not found in Qdrant, so these are not verified policy-rule claims."
    return {
        "workflow": "employee_policy_case", "status": "success", "steps": [
            _step("lookup_employee", "success", f"Loaded {record['employee_id']}"),
            _step("policy_lookup", "success" if hits else "missing", f"Found {len(hits)} chunks for {record['policy_id']}"),
            _step("calculate_values", "success", f"Calculated {', '.join(calculation['requested'])}"),
        ], "hits": hits, "citations": [], "answer_context": _context(hits) if hits else "",
        "missing_information": [], "evidence_ok": bool(hits), "reason": "" if hits else "policy evidence not indexed",
        "record": record, "calculation": calculation, "answer": answer,
    }


def employee_comparison(query: str, *, strategy: str = "structure", top_k: int = 5) -> dict[str, Any]:
    """Resolve and compare every employee mentioned in a question."""
    keys = identifiers_from_query(query)
    records = [lookup_record(key) for key in keys]
    records = [record for record in records if record]
    if len(records) < 2:
        return {"workflow": "employee_comparison", "status": "needs_input", "steps": [
            _step("identify_employees", "needs_input", "Provide two valid employee IDs, customer IDs, or emails")
        ], "hits": [], "citations": [], "answer_context": "", "missing_information": ["two valid employee records"],
        "evidence_ok": False, "reason": "two employee records are required", "records": records, "answer": ""}
    all_hits: list[Hit] = []
    for record in records:
        all_hits.extend(search(strategy, query, top_k=max(1, top_k // len(records)), method=HYBRID, rerank=RERANK_LOCAL, policy_id=record["policy_id"]))
    answer = format_employee_comparison(records)
    if len(all_hits) < 2:
        answer += "\n\nOne or more policy documents are not indexed, so this comparison uses structured demo-record values only."
    return {"workflow": "employee_comparison", "status": "success", "steps": [
        _step("identify_employees", "success", f"Resolved {len(records)} employee records"),
        _step("lookup_policies", "success" if all_hits else "missing", f"Retrieved {len(all_hits)} policy chunks"),
        _step("compare_values", "success", "Compared balance, carry-forward, compensatory leave, and policy IDs"),
    ], "hits": all_hits, "citations": [], "answer_context": _context(all_hits) if all_hits else "",
    "missing_information": [], "evidence_ok": len(all_hits) >= 2, "reason": "" if all_hits else "policy evidence missing",
    "records": records, "answer": answer}


# ----------------------------------------------------------------------
# Workflow 7: Policy audit
# ----------------------------------------------------------------------
def policy_audit(
    query: str,
    *,
    answer_text: str = "",
    citations: list[Citation] | list[dict[str, Any]] | None = None,
    hits: list[Hit] | None = None,
    strategy: str = "structure",
    top_k: int = 5,
) -> dict[str, Any]:
    """Audit generated answers and citations before delivery.

    Steps:
      1. Inspect answer claims and citations
      2. Verify all citations resolve to supplied chunks
      3. Check policy IDs and sections
      4. Detect conflicting cross-policy evidence
      5. Approve, revise, or refuse
    """
    steps: list[dict[str, str]] = []
    norm_query = normalize_query(query)

    if hits is None:
        hits = search(strategy, norm_query, top_k=top_k, method=HYBRID, rerank=RERANK_LOCAL)

    # 1. Parse citations if needed
    parsed_citations: list[Citation] = []
    if citations is not None:
        for c in citations:
            if isinstance(c, Citation):
                parsed_citations.append(c)
            elif isinstance(c, dict):
                parsed_citations.append(
                    Citation(
                        chunk_id=c.get("chunk_id", ""),
                        policy_id=c.get("policy_id", ""),
                        section=c.get("section", ""),
                        resolves=bool(c.get("resolves", False)),
                    )
                )
    elif answer_text and hits:
        parsed_citations = _citations(answer_text, hits)

    steps.append(
        _step("inspect_citations", "success", f"Inspected {len(parsed_citations)} citation(s) in answer")
    )

    # 2. Check citation resolution
    unresolved = [c for c in parsed_citations if not c.resolves]
    if unresolved:
        steps.append(
            _step(
                "resolve_citations",
                "error",
                f"{len(unresolved)} citation(s) failed to resolve: {[c.chunk_id for c in unresolved]}",
            )
        )
        return {
            "workflow": "policy_audit",
            "status": "error",
            "steps": steps,
            "hits": hits,
            "citations": parsed_citations,
            "answer_context": _context(hits),
            "missing_information": [],
            "evidence_ok": False,
            "reason": f"Audit failed: {len(unresolved)} citation(s) do not resolve to supplied chunks.",
        }

    steps.append(_step("resolve_citations", "success", "All citations resolved to supplied chunks"))

    # 3. Check for conflicting policies in hits when query targets a specific entity
    lowered = norm_query.lower()
    targeted_org = next((policy_id for org, policy_id in ALL_ORGS.items() if org in lowered), None)
    if targeted_org and hits:
        conflicting_policies = {h.policy_id for h in hits if h.policy_id != targeted_org}
        if conflicting_policies:
            steps.append(
                _step(
                    "conflict_check",
                    "success",
                    f"Noted cross-policy context: targeted {targeted_org}, but also found {list(conflicting_policies)}",
                )
            )
        else:
            steps.append(_step("conflict_check", "success", "No cross-policy conflicts detected"))
    else:
        steps.append(_step("conflict_check", "success", "Audit check complete"))

    return {
        "workflow": "policy_audit",
        "status": "success",
        "steps": steps,
        "hits": hits,
        "citations": parsed_citations,
        "answer_context": _context(hits),
        "missing_information": [],
        "evidence_ok": True,
        "reason": "",
    }


WORKFLOWS = {
    "policy_lookup": policy_lookup,
    "policy_comparison": policy_comparison,
    "evidence_validation": evidence_validation,
    "policy_applicability": policy_applicability,
    "policy_audit": policy_audit,
}
