"""Streamlit console for exercising the HR policy RAG pipeline by hand.

    streamlit run app.py

This is a testing surface, not a deliverable -- the task awards no marks for
UI. It exists because three things are much easier to see than to read in a
log: the two chunkers side by side on one question, the region filter changing
the top result, and a refusal firing while the relevance scores stay high.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.chunkers import RECURSIVE, STRUCTURE  # noqa: E402
from rag.generate import answer  # noqa: E402
from rag.manifest import DOCUMENTS  # noqa: E402
from rag.questions import QUESTIONS, REFUSALS  # noqa: E402
from rag.retrieve import (  # noqa: E402
    HYBRID,
    RERANK_LOCAL,
    RERANK_OFF,
    RERANK_OPTIONS,
    SEARCH_METHODS,
    SEMANTIC,
    search,
)

st.set_page_config(page_title="HR Policy RAG", page_icon="📄", layout="wide")

REGIONS = ["(no filter)"] + sorted({d.region for d in DOCUMENTS})


def provider_notice(exc: Exception) -> bool:
    """Explain expected provider failures instead of showing a traceback."""
    text = str(exc)
    if "GROQ_API_KEY" in text or "XAI_API_KEY" in text or "GOOGLE_API_KEY" in text or "GEMINI_API_KEY" in text or "API key required" in text:
        st.error("The generation-provider API key is missing or invalid.")
        st.markdown(
            "Add `GROQ_API_KEY=your_groq_key` to the project's `.env` file, then "
            "restart Streamlit. **Search only** continues to work without a key."
        )
        return True

    if "UNAVAILABLE" in text or "503" in text or "high demand" in text.lower():
        st.warning("Groq is temporarily busy. Your policy search completed, but answer generation did not.")
        st.markdown(
            "Wait a short time and run the question again, or switch to **Search only** "
            "to view the retrieved policy chunks without using Groq."
        )
        return True

    if "model_not_found" in text or "does not exist" in text.lower():
        st.error("The configured Groq model is unavailable for this API key.")
        st.markdown(
            "Update `GROQ_GENERATION_MODEL` in `.env` to a model enabled for your "
            "Groq account, then restart Streamlit. The currently tested model is "
            "`openai/gpt-oss-120b`."
        )
        return True

    if "429" in text and "RESOURCE_EXHAUSTED" not in text:
        st.warning("Groq rate limit reached. Your policy search completed, but answer generation did not.")
        st.markdown("Wait a moment and retry, or switch to **Search only** to inspect the local results.")
        return True

    if "RESOURCE_EXHAUSTED" not in text:
        return False
    per_day = "PerDay" in text or "limit: 1000" in text
    st.error("The provider embedding quota is exhausted.")
    if per_day:
        st.markdown(
            "This is the **daily** cap — 1000 embed requests per day for "
            "`gemini-embedding-001`. It resets on Google's daily schedule; there is "
            "no way to speed it up on the free tier."
        )
    else:
        st.markdown("This is the **per-minute** cap. Wait about a minute and retry.")
    st.info(
        "Already-asked questions still work: their embeddings are cached on disk, "
        "so they cost no quota. Try one of the preset buttons you have used before, "
        "or re-ask a previous question verbatim."
    )
    return True


def hit_rows(hits) -> list[dict]:
    return [
        {
            "final rank": h.rank,
            "reranker score": round(h.rerank_score, 4) if h.rerank_score is not None else "-",
            "original rank": h.retrieval_rank if h.retrieval_rank is not None else "-",
            "original score": round(h.retrieval_score, 4) if h.retrieval_score is not None else "-",
            "retrieval score": round(h.score, 4) if h.rerank_score is None else "-",
            "policy_id": h.policy_id,
            "section": h.section or "—",
            "region": h.region,
            "chunk_id": h.chunk_id,
        }
        for h in hits
    ]


def show_hits(hits, caption: str = "") -> None:
    if not hits:
        st.info("Nothing retrieved.")
        return
    st.dataframe(hit_rows(hits), hide_index=True, width="stretch")
    if caption:
        st.caption(caption)
    with st.expander(f"Chunk text ({len(hits)} retrieved)"):
        for h in hits:
            st.markdown(
                f"**{h.rank}. {h.policy_id} section {h.section or '—'}** "
                f"· relevance `{h.score:.4f}` · `{h.chunk_id}`"
            )
            st.code(h.content, language="markdown")


# ----------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Mode",
        ["Ask (retrieve + generate)", "Search only", "Compare both chunkers"],
        help="Search only spends no generation quota.",
    )

    strategy = st.selectbox(
        "Chunking strategy",
        [STRUCTURE, RECURSIVE],
        help="structure splits on policy headers; recursive is the fixed-size baseline.",
        disabled=mode == "Compare both chunkers",
    )

    search_method = st.selectbox(
        "Search method",
        SEARCH_METHODS,
        index=SEARCH_METHODS.index(SEMANTIC),
        format_func=lambda value: {
            SEMANTIC: "Semantic (vector baseline)",
            "bm25": "Keyword (BM25)",
            HYBRID: "Hybrid (semantic + BM25, RRF)",
        }[value],
        help="Hybrid combines semantic and keyword rankings with reciprocal-rank fusion.",
    )

    rerank = st.selectbox(
        "Reranking",
        RERANK_OPTIONS,
        format_func=lambda value: {
            RERANK_OFF: "Off (retriever order)",
            RERANK_LOCAL: "Local cross-encoder (recommended)",
        }[value],
        help="Locally rescores the top 20 retrieved chunks. No Groq API key or quota is used.",
    )

    region_choice = st.selectbox("Region filter", REGIONS)
    region = None if region_choice == "(no filter)" else region_choice

    top_k = st.slider("Chunks to retrieve (top-k)", 1, 15, 5)

    st.divider()
    st.caption(
        "Search-only avoids generation calls. Semantic and hybrid searches create "
        "one query embedding; BM25 runs locally over stored chunk text."
    )

# ----------------------------------------------------------------- presets

st.title("HR Policy RAG")
st.caption(
    "6 policy documents · 4 jurisdictions · two chunking strategies indexed "
    "under the same embedding model"
)

if "query" not in st.session_state:
    st.session_state.query = ""
if "autorun" not in st.session_state:
    st.session_state.autorun = False

st.markdown("### Ask anything")
st.caption(
    "Type any question below and press **Enter**, or use a preset button to fill "
    "the box and run it straight away."
)

known, refusal = st.tabs(["Preset: known-answer questions", "Preset: should be refused"])

with known:
    st.caption("The 8 questions used for the measurement, with known-correct answers.")
    for q in QUESTIONS:
        cols = st.columns([1, 11])
        if cols[0].button(q.qid, key=f"btn_{q.qid}", width="stretch"):
            st.session_state.query = q.query
            st.session_state.autorun = True
        cols[1].markdown(
            f"{q.query}  \n<small>expected <code>{q.policy_id}</code> section "
            f"<code>{q.section}</code> — {q.known_answer}"
            f"{' · <b>table row</b>' if q.from_table_row else ''}</small>",
            unsafe_allow_html=True,
        )

with refusal:
    st.caption(
        "Out-of-corpus questions. Watch the relevance scores stay high while the "
        "answer is still refused — a similarity threshold alone catches none of these."
    )
    for case in REFUSALS:
        cols = st.columns([1, 11])
        if cols[0].button(case.qid, key=f"btn_{case.qid}", width="stretch"):
            st.session_state.query = case.query
            st.session_state.autorun = True
        cols[1].markdown(
            f"{case.query}  \n<small>{case.difficulty}</small>",
            unsafe_allow_html=True,
        )

st.divider()

# A form so that pressing Enter in the box submits. Without it, Enter merely
# triggers a rerun, the Run button reads False, and the app looks like it only
# accepts the preset questions.
with st.form("ask", clear_on_submit=False):
    query = st.text_input(
        "Your question",
        key="query",
        placeholder="e.g. how many privilege leaves for the Chennai office?",
    )
    submitted = st.form_submit_button("Run", type="primary")

# A preset button sets the query and asks for an immediate run; consume the flag
# so a later rerun does not fire the same question again.
autorun = st.session_state.autorun
st.session_state.autorun = False

if not ((submitted or autorun) and query.strip()):
    st.caption("Waiting for a question.")
    st.stop()

# ----------------------------------------------------------------- results


def render_compare(query: str, region, top_k: int, search_method: str, rerank: str) -> None:
    st.subheader("Same question, same search method, different chunker")
    left, right = st.columns(2)
    for column, name in ((left, STRUCTURE), (right, RECURSIVE)):
        with column:
            st.markdown(f"### `{name}`")
            with st.spinner("searching"):
                hits = search(name, query, top_k=top_k, region=region, method=search_method, rerank=rerank)
            citable = sum(1 for h in hits if h.section)
            st.metric("Results citable to a section", f"{citable}/{len(hits)}")
            show_hits(hits)
    st.info(
        "The **section** column is the difference. `structure` names the policy "
        "section; `recursive` shows a dash because it carries no structural "
        "metadata, so a citation can name the file but not the clause."
    )


def render_search(query: str, strategy: str, region, top_k: int, search_method: str, rerank: str) -> None:
    st.subheader("Retrieved chunks")
    with st.spinner("searching"):
        hits = search(strategy, query, top_k=top_k, region=region, method=search_method, rerank=rerank)
    show_hits(hits, f"strategy={strategy}" + (f" · region={region}" if region else ""))


def render_answer(query: str, strategy: str, region, top_k: int, search_method: str, rerank: str) -> None:
    with st.spinner("retrieving and generating"):
        result = answer(strategy, query, top_k=top_k, region=region, method=search_method, rerank=rerank)

    if result.is_refusal:
        st.error(f"**Refused** — gate: `{result.gate}`")
        st.markdown(result.text)
        if result.hits:
            st.warning(
                f"Top-1 relevance was **{result.hits[0].score:.4f}** from "
                f"`{result.hits[0].policy_id}`. Comfortably above any workable "
                "similarity floor — which is why the refusal is decided in code, "
                "before the model is ever called."
            )
    else:
        st.success("Answered from the corpus")
        st.markdown(result.text)

        if result.citations:
            st.subheader("Citations")
            st.dataframe(
                [
                    {
                        "resolves": "yes" if c.resolves else "NO",
                        "chunk_id": c.chunk_id,
                        "policy_id": c.policy_id,
                        "section": c.section or "—",
                    }
                    for c in result.citations
                ],
                hide_index=True,
                width="stretch",
            )
            unresolved = [c for c in result.citations if not c.resolves]
            if unresolved:
                st.error(
                    f"{len(unresolved)} citation(s) do not resolve to a supplied "
                    "chunk — the model invented an id."
                )
        else:
            st.warning("The answer carries no citations.")

    st.subheader("Retrieved chunks")
    show_hits(result.hits)


try:
    if mode == "Compare both chunkers":
        render_compare(query, region, top_k, search_method, rerank)
    elif mode == "Search only":
        render_search(query, strategy, region, top_k, search_method, rerank)
    else:
        render_answer(query, strategy, region, top_k, search_method, rerank)
except Exception as exc:  # noqa: BLE001 - explained to the user, never swallowed
    if not provider_notice(exc):
        raise
