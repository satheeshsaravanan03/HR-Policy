# Building Production-Grade RAG Applications: A Senior-Level Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Chunking Strategies](#chunking-strategies)
4. [Embedding & Retrieval](#embedding--retrieval)
5. [Generation with Citations](#generation-with-citations)
6. [Refusal Mechanisms](#refusal-mechanisms)
7. [Complete Implementation](#complete-implementation)
8. [Testing & Measurement](#testing--measurement)

---

## Architecture Overview

### Why RAG Exists

**Problem**: LLMs hallucinate when asked about specific facts they weren't trained on or when information changes after training cutoff.

**Solution**: Retrieve relevant documents from a knowledge base BEFORE generation, grounding the LLM's response in actual source material.

```
User Query
    ↓
[Embedding] → Query Vector
    ↓
[Vector DB] ← Document Chunks with Embeddings
    ↓
[Retrieval] → Top-K Relevant Chunks (+ metadata)
    ↓
[LLM] ← Chunks + Query + System Prompt
    ↓
Generated Response with Citations
```

### Why This Approach Solves Real Problems

1. **Hallucination Control**: Model can only cite what exists in chunks
2. **Freshness**: Update docs without retraining the model
3. **Auditability**: Every claim traces back to a source (chunk_id → page → section)
4. **Cost**: Don't pay for a larger model; retrieval handles specificity

---

## Core Components

### 1. Document Ingestion Pipeline

**Why separate?** 
- Decouples document handling from retrieval
- Allows testing chunking strategies independently
- Enables versioning and rollback

```python
class DocumentIngester:
    """
    Responsibility: Convert raw documents into chunks with complete metadata.
    
    Why this structure:
    - Single responsibility: ingest → chunk → enrich
    - Testable: unit tests per chunk strategy
    - Extensible: add metadata, custom parsers without touching retrieval
    """
    
    def __init__(self, chunk_strategy: ChunkingStrategy):
        self.chunker = chunk_strategy
    
    def ingest(self, file_path: str, metadata: dict) -> List[Chunk]:
        """
        Args:
            file_path: Path to document (markdown, PDF, etc)
            metadata: {source_file, page_id, sdk_version, page_type, ...}
        
        Returns:
            List[Chunk] where each has:
            - content: str (the actual text)
            - chunk_id: str (unique identifier)
            - metadata: {source_file, page_id, line_range, ...}
            - embedding: np.array (computed later)
        
        Critical checks:
        - No chunk without source_file metadata (requirement 1)
        - chunk_id must be deterministic and traceable
        """
        raw_text = self._load_file(file_path)
        chunks = self.chunker.split(raw_text, metadata)
        
        # Validation: fail fast on missing metadata
        for chunk in chunks:
            assert chunk.metadata.get('source_file'), \
                f"Chunk {chunk.chunk_id} missing source_file metadata"
        
        return chunks
```

### 2. Chunking Strategy (The Heart of RAG)

**Critical Insight**: Chunking is where most retrieval failures originate. Poor chunks = poor retrieval.

#### Strategy A: Token-Based Chunking (Baseline)

```python
class TokenChunker(ChunkingStrategy):
    """
    Why use this:
    - Fast, simple, works for unstructured text
    - Language-model-aware (counts actual tokens, not characters)
    - Handles overlaps to preserve context
    
    When it fails:
    - Cuts tables/code fences in half
    - Separates parameter defaults from their headers
    - Loses structural meaning
    
    Good for: Unstructured prose (blog posts, guides)
    Bad for: Technical reference docs with tables/code
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        chunk_size: tokens per chunk
        overlap: tokens to repeat in adjacent chunks (context preservation)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def split(self, text: str, metadata: dict) -> List[Chunk]:
        tokens = self.tokenize(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk_text = self.detokenize(chunk_tokens)
            
            chunks.append(Chunk(
                content=chunk_text,
                chunk_id=f"{metadata['source_file']}#token_{i}",
                metadata={
                    **metadata,
                    'chunk_start_token': i,
                    'chunk_end_token': i + len(chunk_tokens),
                }
            ))
        
        return chunks
```

#### Strategy B: Structure-Aware Chunking (Production)

```python
class StructureAwareChunker(ChunkingStrategy):
    """
    Why use this:
    - Respects document semantics (headers, tables, code fences)
    - Keeps parameter rows with headers → table defaults stay attached
    - Never splits code fences → examples stay intact
    - Produces fewer but more coherent chunks
    
    How it works:
    1. Parse markdown structure (headers, fences, tables)
    2. Split ONLY on semantic boundaries (h2, h3)
    3. Keep tables and code as atomic units
    
    Example:
    Input text:
        ## Client.send()
        Sends a message. Parameters:
        | Name | Type | Default |
        | retry_backoff_ms | int | 100 |
        
        ```python
        client.send(msg, retry_backoff_ms=200)
        ```
    
    Token chunker might do:
        Chunk 1: "## Client.send() Sends a message. Parameters: | Name"
        Chunk 2: "| Type | Default | retry_backoff_ms | int | 100..."
    
    Structure-aware chunker does:
        Chunk 1: [entire section including table and code]
    """
    
    def split(self, text: str, metadata: dict) -> List[Chunk]:
        chunks = []
        
        # Step 1: Parse structure
        sections = self._parse_sections(text)  # Split on h2, h3
        
        for section in sections:
            # Step 2: Keep tables and code fences atomic
            content = self._preserve_structures(section['content'])
            
            # Step 3: Size check (if section too large, split on h4)
            if self._token_count(content) > 1500:
                subsections = self._parse_subsections(content)
                for subsection in subsections:
                    chunks.append(self._make_chunk(
                        subsection,
                        metadata,
                        section['header']
                    ))
            else:
                chunks.append(self._make_chunk(
                    content,
                    metadata,
                    section['header']
                ))
        
        return chunks
    
    def _parse_sections(self, text: str) -> List[dict]:
        """Split on h2 boundaries only"""
        sections = []
        current_section = None
        
        for line in text.split('\n'):
            if line.startswith('## '):
                if current_section:
                    sections.append(current_section)
                current_section = {
                    'header': line[3:],
                    'content': [line]
                }
            elif current_section:
                current_section['content'].append(line)
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _preserve_structures(self, text: str) -> str:
        """
        Ensure code fences and tables are never split.
        Mark them so we don't split inside them.
        """
        lines = []
        in_code_fence = False
        in_table = False
        
        for line in text.split('\n'):
            if line.strip().startswith('```'):
                in_code_fence = not in_code_fence
            if '|' in line and '-' in line:
                in_table = True
            elif in_table and '|' not in line:
                in_table = False
            
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _make_chunk(self, content: str, metadata: dict, 
                    section_header: str) -> Chunk:
        """Create chunk with rich metadata for traceability"""
        return Chunk(
            content=content,
            chunk_id=f"{metadata['source_file']}#{section_header.lower().replace(' ', '_')}",
            metadata={
                **metadata,
                'section_header': section_header,
                'has_code_fence': '```' in content,
                'has_table': '|' in content,
                'token_count': self._token_count(content),
            }
        )
```

**Why Structure-Aware Wins for Technical Docs:**
- Parameter default values stay with headers → retrieval finds context
- Code examples aren't mangled → user sees working code
- Sections are topical units → better semantic coherence
- Metadata flags (has_code_fence) enable smart filtering

---

## Embedding & Retrieval

### Why Embeddings Matter

**Problem**: Keyword search can't find "maximum attempts between requests" when docs say "retry_backoff_ms"

**Solution**: Embed both documents and queries in the same semantic space. Similar meanings cluster together.

```python
class EmbeddingStore:
    """
    Responsibility: Convert chunks to vectors, store, retrieve by similarity.
    
    Design decision: Separate from chunking
    Why: Test chunking without waiting for embedding API calls
    """
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Why text-embedding-3-small:
        - Small: 1536 dims (vs 3072 for large)
        - Fast: inference is 3x faster
        - Good enough: 95% accuracy vs large for semantic retrieval
        - Cost: 1/8 the price of -large
        - Trade-off: accept 5% accuracy loss to ship faster, measure impact
        """
        self.model = embedding_model
        self.client = OpenAI()
        self.vector_store = {}  # or Pinecone, Weaviate, etc
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Add embeddings to chunks. Batched for efficiency.
        """
        # Batch requests (embedding APIs charge per request)
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i+100]
            texts = [chunk.content for chunk in batch]
            
            embeddings = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            for chunk, embedding in zip(batch, embeddings.data):
                chunk.embedding = np.array(embedding.embedding)
                self.vector_store[chunk.chunk_id] = chunk
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = 5, 
                 filters: dict = None) -> List[RetrievalResult]:
        """
        Retrieve top-k chunks by cosine similarity.
        
        Args:
            query: User question
            top_k: Number of results (5 is standard; more = higher recall, cost)
            filters: Optional {sdk_version: "v3"} to exclude v2 docs
        
        Why filters matter (Requirement 3):
        Query: "What is retry_backoff_ms?"
        Without filter: v2 Client.send() ranks #1 (older, but same parameter)
        With filter sdk_version=v3: v3 Client.send() ranks #1 (current API)
        
        This demonstrates why metadata is critical in production RAG.
        """
        # Embed the query
        query_embedding = self.client.embeddings.create(
            model=self.model,
            input=[query]
        ).data[0].embedding
        
        query_embedding = np.array(query_embedding)
        
        # Score all chunks
        scores = []
        for chunk_id, chunk in self.vector_store.items():
            # Skip filtered chunks
            if filters:
                if chunk.metadata.get('sdk_version') != filters.get('sdk_version'):
                    continue
            
            # Cosine similarity = dot product of normalized vectors
            similarity = np.dot(
                query_embedding, 
                chunk.embedding
            ) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding))
            
            scores.append((chunk_id, chunk, similarity))
        
        # Sort by score, return top-k
        scores.sort(key=lambda x: x[2], reverse=True)
        
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                content=chunk.content,
                score=score,
                metadata=chunk.metadata,
                embedding_rank=i
            )
            for i, (chunk_id, chunk, score) in enumerate(scores[:top_k])
        ]
```

### Why Top-K = 5 Works

- **Recall**: Balances finding the right answer vs noise
- **Cost**: ~1000 tokens sent to LLM per query
- **Latency**: Sub-100ms retrieval
- **Empirical**: In practice, if answer isn't in top-5, it's not in docs

---

## Generation with Citations

### Why Citations are Non-Negotiable

**Without citations**: "This parameter has a default value of 100." → Is this from v2 or v3? Can the user trust it?

**With citations**: "Default is 100 (Client.send()#parameters, row 3)" → User can click/trace to source.

```python
class CitedGenerator:
    """
    Responsibility: Take retrieved chunks + query, generate answer with citations.
    
    Design principle: Chains of Thought (CoT) → show reasoning
    Why: Model explicitly states which chunk it's using for each claim
    """
    
    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model
        self.client = OpenAI()
    
    def generate(self, query: str, 
                 retrieved: List[RetrievalResult]) -> GeneratedAnswer:
        """
        Generate answer citing specific chunks.
        
        Key prompt design:
        1. Instruct model to cite chunk_id for EVERY claim
        2. Give model exact chunk contents with IDs
        3. Force refusal if chunks don't cover the question
        """
        
        # Build context with explicit chunk markers
        context_text = ""
        for result in retrieved:
            context_text += f"""
[CHUNK_ID: {result.chunk_id}]
Source: {result.metadata['source_file']} (SDK v{result.metadata['sdk_version']})
{result.content}
---
"""
        
        system_prompt = """You are a documentation assistant.
        
CRITICAL RULES:
1. EVERY claim must cite a chunk_id in format [CITE: chunk_id]
2. If the retrieved chunks do NOT answer the question, you must REFUSE
3. Do not use knowledge outside the provided chunks
4. If unsure, explicitly state uncertainty and cite your source

Example:
Q: What is retry_backoff_ms?
A: The retry_backoff_ms parameter on Client.send() has a default value of 100 milliseconds [CITE: path/to/file#client_send]. It controls the delay between retry attempts [CITE: path/to/file#parameters]. If you set it to 200, the client will wait 200ms before retrying [CITE: path/to/file#code_example].
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Context chunks:
{context_text}

Question: {query}

Remember:
- Cite every claim with [CITE: chunk_id]
- Refuse if chunks don't answer
- Be specific and grounded"""}
            ],
            temperature=0.3  # Lower temp = more deterministic, fewer hallucinations
        )
        
        answer_text = response.choices[0].message.content
        
        # Extract citations from answer
        citations = self._extract_citations(answer_text, retrieved)
        
        # Validate citations (must exist in retrieved chunks)
        validated = self._validate_citations(citations, retrieved)
        
        return GeneratedAnswer(
            answer=answer_text,
            citations=validated,
            retrieved_chunks=retrieved,
            model=self.model,
            num_input_tokens=response.usage.prompt_tokens,
            num_output_tokens=response.usage.completion_tokens,
        )
    
    def _extract_citations(self, text: str, 
                           retrieved: List[RetrievalResult]) -> List[Citation]:
        """Parse [CITE: chunk_id] markers from answer"""
        import re
        citations = []
        
        for match in re.finditer(r'\[CITE:\s*([^\]]+)\]', text):
            cited_id = match.group(1).strip()
            
            # Find which chunk this refers to
            for result in retrieved:
                if result.chunk_id == cited_id:
                    citations.append(Citation(
                        chunk_id=cited_id,
                        page_id=result.metadata.get('page_id'),
                        source_file=result.metadata.get('source_file'),
                        position_in_text=match.start(),
                    ))
                    break
        
        return citations
    
    def _validate_citations(self, citations: List[Citation],
                           retrieved: List[RetrievalResult]) -> List[Citation]:
        """
        Ensure every citation exists in retrieved chunks.
        
        Why this matters:
        Sometimes models hallucinate chunk IDs or make up references.
        This validation catches that. Bad citation = we mark it.
        """
        valid_ids = {r.chunk_id for r in retrieved}
        validated = []
        
        for citation in citations:
            if citation.chunk_id in valid_ids:
                validated.append(citation)
            else:
                # Log the hallucination for monitoring
                print(f"WARNING: Cited non-existent chunk: {citation.chunk_id}")
        
        return validated
```

---

## Refusal Mechanisms

### Why Refusals are Critical

**Problem**: Without hard refusals, model will confidently invent answers.

**Example**:
- Q: "What's the rate limit for the v3 API?"
- No docs exist for this.
- Bad system: Model invents "100 requests per minute"
- Good system: "I cannot find rate limit information in the v3 SDK reference."

```python
class RefusalFilter:
    """
    Responsibility: Detect when a query cannot be answered from corpus.
    
    Strategy 1: Retrieval-Based Refusal
    If top-5 chunks have low similarity, question is out-of-scope.
    
    Strategy 2: Embedding-Based Refusal
    If retrieved chunks are too different from query (low semantic overlap),
    model is likely to hallucinate.
    
    Strategy 3: Explicit Refusal Prompt
    Force model to refuse by design.
    """
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        similarity_threshold: If max similarity < threshold, refuse
        
        Calibration:
        - 0.7 threshold: Very strict, might miss valid answers
        - 0.5 threshold: Sweet spot for technical docs
        - 0.3 threshold: Very loose, will try to answer anything
        
        Find yours by testing against known-good questions.
        """
        self.threshold = similarity_threshold
    
    def should_refuse(self, query: str, 
                     retrieved: List[RetrievalResult],
                     semantic_overlap: float) -> tuple[bool, str]:
        """
        Returns: (should_refuse, reason)
        
        Reason matters for monitoring. If we're refusing too many valid
        questions, we see a pattern in the reasons.
        """
        
        # Check 1: Are top results even relevant?
        if not retrieved:
            return (True, "No documents in corpus match query")
        
        top_score = retrieved[0].score
        if top_score < self.threshold:
            return (
                True,
                f"Top result similarity {top_score:.2f} below threshold {self.threshold}"
            )
        
        # Check 2: Is top result MUCH better than others?
        if len(retrieved) > 1:
            score_gap = retrieved[0].score - retrieved[1].score
            if score_gap < 0.05:  # Scores are clustered = ambiguous
                return (
                    True,
                    "Multiple chunks equally match; answer would be ambiguous"
                )
        
        # Check 3: Do chunks contain keywords from query?
        # Prevents model from answering questions about things not in docs
        query_words = set(query.lower().split())
        doc_words = set()
        for chunk in retrieved[:5]:
            doc_words.update(chunk.content.lower().split())
        
        overlap = len(query_words & doc_words) / len(query_words)
        if overlap < 0.2:  # < 20% of query words in docs
            return (
                True,
                "Query and retrieved chunks have low lexical overlap"
            )
        
        return (False, "")
```

### Forced Refusal in Prompts

```python
def generate_with_refusal(self, query: str, 
                         retrieved: List[RetrievalResult]) -> GeneratedAnswer:
    """
    Two-stage process:
    1. Check if refusal is warranted
    2. If no, generate; if yes, REFUSE (don't ask model to decide)
    """
    
    refusal_filter = RefusalFilter(similarity_threshold=0.5)
    should_refuse, reason = refusal_filter.should_refuse(query, retrieved)
    
    if should_refuse:
        return GeneratedAnswer(
            answer=f"I cannot answer this question based on the available documentation.\n\nReason: {reason}",
            citations=[],  # No citations for refusals
            retrieved_chunks=retrieved,
            is_refusal=True,
            refusal_reason=reason,
        )
    
    # Only generate if refusal check passes
    return self.generate(query, retrieved)
```

**Key insight**: Refusal is FORCED by the system, not suggested to the model. The prompt doesn't say "if you can't answer, refuse" — it prevents the model from ever seeing out-of-scope questions.

---

## Complete Implementation

### End-to-End Flow

```python
class RAGApplication:
    """
    Orchestrates: ingest → embed → retrieve → generate → cite → refuse
    
    Design: Each stage is independent and testable
    """
    
    def __init__(self, chunker: ChunkingStrategy, embedding_model: str):
        self.chunker = chunker
        self.embedder = EmbeddingStore(embedding_model)
        self.generator = CitedGenerator()
        self.refusal_filter = RefusalFilter(similarity_threshold=0.5)
    
    def ingest_documents(self, file_paths: List[str], 
                        metadata: dict) -> None:
        """
        Stage 1: Load files → Chunk → Embed → Store
        
        Metadata must include:
        - source_file: "api_reference_v3.md"
        - page_id: "client_send"
        - sdk_version: "v3"
        - page_type: "reference"
        """
        ingester = DocumentIngester(self.chunker)
        
        all_chunks = []
        for file_path in file_paths:
            chunks = ingester.ingest(file_path, metadata)
            all_chunks.extend(chunks)
        
        print(f"✓ Ingested {len(all_chunks)} chunks")
        
        # Embed all chunks
        self.embedder.embed_chunks(all_chunks)
        print(f"✓ Embedded {len(all_chunks)} chunks")
    
    def answer(self, query: str, filters: dict = None) -> GeneratedAnswer:
        """
        Stage 2: Retrieve → Check refusal → Generate with citations
        """
        # Retrieve
        retrieved = self.embedder.retrieve(
            query, 
            top_k=5, 
            filters=filters  # {sdk_version: "v3"}
        )
        
        # Check if we should refuse
        should_refuse, reason = self.refusal_filter.should_refuse(
            query, 
            retrieved
        )
        
        if should_refuse:
            return GeneratedAnswer(
                answer=f"I cannot answer this based on available docs.\n({reason})",
                is_refusal=True
            )
        
        # Generate with citations
        return self.generator.generate(query, retrieved)
    
    def measure_retrieval(self, questions: List[dict]) -> RetrieverResults:
        """
        Test retrieval without generation.
        
        Input: [
            {
                "query": "What is the default value of retry_backoff_ms?",
                "expected_chunk_id": "api_reference_v3#client_send",
                "expected_rank": 1  # Should be in top-5 (rank <= 4)
            },
            ...
        ]
        
        Output: hit_in_top_5 count for this chunker
        """
        hits = 0
        
        for question in questions:
            retrieved = self.embedder.retrieve(question['query'], top_k=5)
            
            # Check if expected chunk is in results
            result_ids = [r.chunk_id for r in retrieved]
            
            if question['expected_chunk_id'] in result_ids:
                hits += 1
            else:
                print(f"MISS: {question['query']}")
                print(f"  Expected: {question['expected_chunk_id']}")
                print(f"  Got: {result_ids}")
        
        return RetrieverResults(
            total_questions=len(questions),
            hits=hits,
            hit_rate=f"{hits}/{len(questions)}"
        )
```

---

## Testing & Measurement

### Requirement 2: Write 8 Known-Answer Questions FIRST

**Critical**: Write questions BEFORE looking at retrieval results.

```markdown
## Test Questions (Known Answers)

1. **Q: What is the default value of retry_backoff_ms on Client.send()?**
   - Expected chunk: api_reference_v3#client_send
   - Expected section: Parameters table, row 3
   - Known answer: 100 milliseconds

2. **Q: How do you pass a custom retry_backoff_ms when calling Client.send()?**
   - Expected chunk: api_reference_v3#code_example
   - Expected section: Code fence showing usage
   - Known answer: client.send(message, retry_backoff_ms=200)

3. **Q: What type is the retry_backoff_ms parameter?**
   - Expected chunk: api_reference_v3#client_send
   - Expected section: Parameters table, type column
   - Known answer: integer (int)

4. **Q: Is retry_backoff_ms required for Client.send()?**
   - Expected chunk: api_reference_v3#client_send
   - Expected section: Parameters table, required column
   - Known answer: No, it's optional (default provided)

5. **Q: What does the timeout parameter on send() do?**
   - Expected chunk: api_reference_v3#parameters
   - Known answer: Sets maximum time to wait for response

6. **Q: How do you configure logging in the v3 SDK?**
   - Expected chunk: api_reference_v3#logging
   - Known answer: [specific config from page]

7. **Q: What changed about retry behavior from v2 to v3?**
   - Expected chunk: api_reference_v3#migration_guide
   - Known answer: [specific change documented]

8. **Q: What is the signature of Client.connect()?**
   - Expected chunk: api_reference_v3#client_connect
   - Known answer: [parameters and return type from reference]
```

### Requirement 3: Measure Both Chunkers Against Same Questions

```python
def compare_chunkers(all_questions: List[dict]) -> dict:
    """
    Run same 8 questions against 2 chunkers.
    Measure hit-in-top-5 for each.
    """
    
    # Chunker 1: Token-based
    app_token = RAGApplication(
        TokenChunker(chunk_size=512),
        embedding_model="text-embedding-3-small"
    )
    app_token.ingest_documents(file_paths, metadata)
    results_token = app_token.measure_retrieval(all_questions)
    
    # Chunker 2: Structure-aware
    app_struct = RAGApplication(
        StructureAwareChunker(),
        embedding_model="text-embedding-3-small"  # SAME model, only change chunker
    )
    app_struct.ingest_documents(file_paths, metadata)
    results_struct = app_struct.measure_retrieval(all_questions)
    
    print(f"""
    | Chunker | Hit-in-Top-5 |
    |---------|--------------|
    | Token-based | {results_token.hit_rate} |
    | Structure-aware | {results_struct.hit_rate} |
    """)
    
    return {
        "token_based": results_token,
        "structure_aware": results_struct,
    }
```

### Requirement 4: Demonstrate Metadata Filter Impact

```python
def demonstrate_filter_impact(query: str):
    """
    Show how sdk_version filter changes top-1 result
    """
    
    app = RAGApplication(StructureAwareChunker(), "text-embedding-3-small")
    app.ingest_documents(file_paths, metadata)
    
    # Unfiltered: might rank v2 first if it has more mentions
    unfiltered = app.embedder.retrieve(query, top_k=5)
    
    print("UNFILTERED RESULTS:")
    for i, result in enumerate(unfiltered):
        print(f"{i+1}. {result.chunk_id} (v{result.metadata['sdk_version']}) - {result.score:.3f}")
    
    # Filtered: only v3
    filtered = app.embedder.retrieve(
        query, 
        top_k=5, 
        filters={'sdk_version': 'v3'}
    )
    
    print("\nFILTERED RESULTS (sdk_version=v3):")
    for i, result in enumerate(filtered):
        print(f"{i+1}. {result.chunk_id} (v{result.metadata['sdk_version']}) - {result.score:.3f}")
    
    # The bug: unfiltered top-1 might be v2, filtered top-1 is v3
    if unfiltered[0].metadata['sdk_version'] != 'v3':
        print(f"\n⚠️  BUG REPRODUCED: v2 ({unfiltered[0].chunk_id}) ranked above v3 API")
```

### Requirement 5: Run Through Generation with Refusals

```python
def test_generation_pipeline():
    """
    3 answerable questions → generation with citations
    3 unanswerable questions → refusals
    """
    
    app = RAGApplication(StructureAwareChunker(), "text-embedding-3-small")
    app.ingest_documents(file_paths, metadata)
    
    # Answerable
    answerable = [
        "What is retry_backoff_ms?",
        "How do you enable debug logging in v3?",
        "What's the Client.send() signature?",
    ]
    
    print("=== ANSWERABLE QUESTIONS ===\n")
    for q in answerable:
        answer = app.answer(q)
        print(f"Q: {q}")
        print(f"A: {answer.answer}\n")
        
        if answer.citations:
            for cite in answer.citations:
                print(f"  [CITE] {cite.chunk_id} ({cite.source_file})")
        print()
    
    # Unanswerable
    unanswerable = [
        "What's the rate limit for the v3 API?",
        "How much does the v3 SDK cost?",
        "What Python version does v3 require?",
    ]
    
    print("=== UNANSWERABLE QUESTIONS ===\n")
    for q in unanswerable:
        answer = app.answer(q)
        print(f"Q: {q}")
        print(f"A: {answer.answer}")
        print(f"Refusal: {answer.is_refusal}")
        print()
```

---

## Production Checklist

### Before Shipping

```python
def pre_production_audit():
    """
    Audit your RAG system for common failure modes
    """
    
    checks = {
        "No chunks missing source_file": validate_metadata_completeness(),
        "Chunk IDs are deterministic": validate_chunk_id_stability(),
        "Retrieval threshold calibrated": validate_similarity_threshold(),
        "Citations validate against chunks": validate_citations_resolvable(),
        "Refusal prompt is airtight": validate_refusal_prompt(),
        "Embedding model matches prod": validate_embedding_consistency(),
    }
    
    for check, passed in checks.items():
        print(f"{'✓' if passed else '✗'} {check}")
    
    if all(checks.values()):
        print("\n✓ Ready for production")
    else:
        print("\n✗ Failures above must be fixed")
```

---

## Key Design Decisions Summary

| Decision | Why | Trade-off |
|----------|-----|-----------|
| **Structure-aware chunking** | Keeps tables/code atomic | Slightly larger chunks, fewer results |
| **text-embedding-3-small** | Fast, cheap, 95% accuracy | 5% accuracy loss vs large |
| **Top-k=5** | Balances recall vs cost | Miss some answers if relevant answer is ranked 6+ |
| **Similarity threshold=0.5** | Good for technical docs | Tune per domain (0.6 stricter, 0.4 looser) |
| **Forced refusal in code** | Prevents hallucinations | Might over-refuse edge cases |
| **Citation extraction + validation** | Catches model hallucinations | Adds latency |

---

## Common Failure Modes & Diagnosis

### "My retriever is hallucinating answers"
**Root cause**: Refusal threshold too low or prompt doesn't force refusal
**Fix**: Implement RefusalFilter with similarity check, NOT prompt suggestion

### "Important table defaults are being split across chunks"
**Root cause**: Token-based chunker doesn't understand structure
**Fix**: Switch to StructureAwareChunker, validate with test questions

### "v2 API docs outrank v3 in results"
**Root cause**: More mentions of v2 in corpus, or better-trained embeddings on v2
**Fix**: Add metadata filters, increase v3 docs, or rescore with recency

### "I'm missing questions that reference code examples"
**Root cause**: Code fences are being split; example gets separated from description
**Fix**: Make code fences atomic in chunker (StructureAwareChunker does this)

---

## Next Steps

1. **Start with StructureAwareChunker** — most RAG failures come from poor chunking
2. **Write 8 test questions from your docs** BEFORE measuring
3. **Measure retrieval first** (top-k hit rate) — don't jump to generation
4. **Add citations before deployment** — untraced answers are liabilities
5. **Implement RefusalFilter** — the difference between production and prototype

Good luck! The most important thing: measure what moved. A number beats intuition every time.
