# RAG Fundamentals

## What is it?

Retrieval-Augmented Generation: instead of asking an LLM to answer from what it memorized in training, first
**retrieve** relevant passages from your own knowledge base, then have the LLM **generate** an answer using
only those passages. The LLM becomes a reader/synthesizer, not a database.

## Why did we need it?

An LLM alone doesn't know a store's current stock, prices or sizes, can't cite where a claim came from, and
will confidently invent answers. Fine-tuning bakes in knowledge that goes stale and can't be attributed. RAG
keeps knowledge in an updatable store and lets us show sources.

## How did we implement it?

`RAGService.answer()` ([app/services/rag_service.py](../app/services/rag_service.py)):

```text
question → input guard → (rewrite if history) → retrieve → weak-retrieval check
         → content guard → (rerank) → context token budget → numbered context [1][2]…
         → LLM with SYSTEM_PROMPT → answer + sources + cited_source_count
```

- The system prompt is a hard instruction: answer from the supplied context, say "insufficient information"
  if it can't be determined, don't invent, cite with `[n]`.
- Context is built as numbered blocks; `extract_cited_indices()` parses `[n]` back out of the answer so a claim
  can be traced to one exact chunk. Out-of-range numbers are discarded.
- Retrieval and generation are separately testable: `/api/search` never calls an LLM.

## What alternatives existed?

| Alternative | Why not (here) |
|---|---|
| Fine-tune an LLM on the data | Stale, no citations, expensive to update |
| Stuff the whole corpus in the prompt | Context limits, cost, "lost in the middle" |
| Text-to-SQL over the source DB | Gives the chat path direct DB access (security risk); poor for fuzzy/semantic questions |
| Keyword search only | Misses paraphrases (see [hybrid-search.md](hybrid-search.md)) |
| Managed RAG / vector DB service | Less to learn; the project goal was understanding each part |

## Why did we choose this approach?

Build every concept by hand first (no LangChain until Plan Set 22) so each design choice is understood, keep
each stage behind an interface, and keep the chat path read-only against a store the RAG system owns.

## What problems did we encounter?

- **RAG reduces hallucination; it doesn't eliminate it.** With the stub LLM, an off-topic query ("capital of
  France") still retrieved weak-similarity context (0.24). Fixed at the pipeline level with a similarity
  threshold (0.35) that skips the LLM entirely — a rule, not a model judgment.
- **Sources ≠ proof of support.** "4 sources retrieved, 1 cited" is why we expose `cited_source_count`. A stub
  bug where block `[2]`'s header leaked into the answer produced a false-positive citation.

## How did we test it?

Unit tests with fake retrievers/LLMs for the service wiring; API tests against a live pgvector DB (skipped when
the DB isn't running); the 52-question end-to-end evaluation ([evaluation.md](evaluation.md)).

## What would we change at scale?

Wire a real LLM and add an LLM-as-judge for groundedness; add hybrid retrieval to the chat path; stream
responses; cache frequent query embeddings; add an ANN index.

## How would I explain it in an interview?

"RAG is open-book exam for an LLM. I retrieve the top passages from a vector store I own, put them in numbered
blocks, and instruct the model to answer only from those and cite by number. The value isn't the LLM call — it's
everything around it: chunking, retrieval quality, guardrails so weak retrieval never reaches the model, and
evaluation so I know if a change helped. It reduces hallucination but doesn't remove it, so I measure
groundedness and citation correctness rather than assume them."
