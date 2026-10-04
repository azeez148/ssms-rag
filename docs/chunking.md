# Chunking

## What is it?

Splitting each document into smaller, coherent pieces (chunks) before embedding. Code:
[app/ingestion/chunking.py](../app/ingestion/chunking.py) — `ChunkingStrategy` ABC with two implementations:
`RecursiveTokenChunker` (hand-written) and `LangChainRecursiveChunker` (current default, wraps
`RecursiveCharacterTextSplitter.from_tiktoken_encoder`). Defaults: **500 tokens, 50 overlap**; overlap must be
smaller than size (validated).

## Why did we need it?

One vector per whole document is a blurry average of every topic in it, so similarity gets imprecise; large
documents can exceed the embedding model's input limit; and retrieving "the whole document" defeats top-K
(the LLM gets everything regardless of relevance).

## How did we implement it?

Recursive split on progressively finer separators (paragraph → line → sentence → word) until each piece fits
the token budget, then greedily merge small pieces back up to `chunk_size` while carrying a sliding window of
`chunk_overlap` trailing tokens into the next chunk. Size is measured in **tokens (tiktoken)**, not
characters — the unit models and context budgets actually use.

## What alternatives existed?

Fixed-size windows (cuts mid-sentence), sentence-based, semantic chunking (split where embedding similarity
drops), document-structure-aware (headings/HTML/markdown), parent-document retrieval (search small, return
large).

## Why did we choose this approach?

Recursive splitting respects natural boundaries cheaply and deterministically, and overlap protects ideas that
straddle a boundary ("…sizes S, M, L [cut] and XL" would otherwise leave XL orphaned). Trade-off: small chunks
= precise but can cut ideas and cost more rows; large chunks = more context but coarser matching.

## What problems did we encounter?

1. **Measured, then an honest null result.** On `prompt.md` (long document), 300/30 → 20 chunks averaging 277
   tokens; 800/100 → 7 chunks averaging 812. But on **Sports Store** retrieval, Recall/MRR were identical
   across 300/30, 500/50 and 800/100: product descriptions are short, so every config produced one chunk per
   document. Chunk size can't matter on this corpus — the experiment only says something on long documents.
2. **A real bug in our own chunker.** `_merge` summed pieces' token counts before joining, but the joined
   string's token count isn't always that sum (tokenizer boundary effects). At chunk_size=500, **7 of 12**
   manual chunks exceeded the limit (up to 526 tokens); LangChain's produced **0 of 13**. Same at 300
   (7 of 20 over) and 800 (6 of 7 over). This survived 21 plan sets of tests because no test measured the final
   chunks' real token counts against the budget.
3. So chunking was the one component migrated to LangChain — and only because it's tricky, has no
   project-specific invariant, and sits behind an ABC (one import changed per call site, zero changes to
   `IngestionService`). The manual chunker is kept as the "from first principles" comparison.

## How did we test it?

`tests/test_chunking.py`, `tests/test_langchain_chunking.py` (including that the full ingestion service runs
unchanged with the LangChain chunker), and `scripts/chunking_experiment.py`, which compares both chunkers
across three configs on the same document.

## What would we change at scale?

Evaluate chunking on a corpus with long documents (files, wikis) using retrieval metrics, not intuition; try
structure-aware and parent-document strategies; add a regression test asserting no chunk exceeds
`chunk_size`; record chunker config on each document so re-chunking can be targeted.

## How would I explain it in an interview?

"Chunking sets the granularity of retrieval. Too big and vectors blur and you waste context; too small and you
cut ideas apart. I compared 300/500/800 and found no difference — because my product rows are one chunk each,
so I said that plainly rather than claim a winner. I also found my hand-rolled chunker overflowed its token
budget on more than half the chunks, and moving to a well-tested splitter fixed it. The lesson: measure
outputs against the invariant you care about, and use a library where the component is subtle and generic."
