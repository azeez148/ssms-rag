# Embeddings

## What is it?

A function mapping text to a fixed-length vector so that texts with similar *meaning* land near each other.
We use `all-MiniLM-L6-v2` (sentence-transformers): **384 dimensions**, free, runs locally on CPU.
Code: [app/embeddings/](../app/embeddings/) — `EmbeddingProvider` ABC (`embed_text`, `embed_batch`,
`dimensions`), a sentence-transformers implementation, and a factory keyed on `EMBEDDING_PROVIDER`.

## Why did we need it?

Keyword matching can't find "national team soccer shirt for supporters" when the catalog says "FC ARGENTINA
25-26 jersey". Vectors compare meaning, so retrieval works despite zero shared words.

## How did we implement it?

- `embed_batch` encodes many chunks in one model call (one per document during ingestion) — far faster than a
  loop of single calls. The blocking model call runs in `asyncio.to_thread` so it doesn't stall the event loop.
- `normalize_embeddings=True`, so vectors are unit length and cosine distance is well-behaved.
- Stored in `chunks.embedding` (`vector(384)`, sized from `EMBEDDING_DIMENSIONS`).
- Queries are embedded with the **same model** as documents; pgvector's `<=>` gives cosine distance and we
  report `similarity = 1 − distance`.
- Unknown provider name → the factory raises rather than silently falling back.
- Model weights are baked into the Docker image, so there's no HuggingFace download at startup.

## What alternatives existed?

| Option | Pros | Cons |
|---|---|---|
| Hosted (OpenAI/Cohere/Voyage) | Usually higher retrieval quality, no local compute | Cost per token, network dependency, data leaves the system |
| Larger local model (bge/e5) | Better quality | More RAM/latency |
| Fine-tuned domain model | Best fit | Needs labeled data |

## Why did we choose this approach?

Zero cost and no API key, so the whole pipeline is testable end-to-end offline. A deliberate trade against
retrieval quality — and it shows: the "semantic" eval category has Recall@5 of only 0.333
(see [evaluation.md](evaluation.md)).

## What problems did we encounter?

- **Dimension mismatch**: the column started as `vector(1536)`; changing to a 384-dim model made inserts fail
  until the column changed. The dimension is a property of the model, not a runtime option.
- **Same model for queries and documents is mandatory.** Vectors from different models aren't comparable even
  at equal dimension; swapping models means re-embedding every stored chunk (and, for a different dimension,
  a schema migration).
- A test proves the core claim directly: two Argentina jersey descriptions score closer to each other than
  either to a kitchen blender.

## How did we test it?

`tests/test_embeddings.py` — correct dimensions, one vector per batch input, identical text → identical
embedding, the semantic-similarity ordering above, and a clear error for an unsupported provider (loads the
real model, so it's slower than most tests).

## What would we change at scale?

Hosted or bigger model benchmarked against our eval set before switching; store `embedding_model` per chunk
so mixed-model corpora are detectable; batch/queue embedding jobs; cache query embeddings; consider
quantization (halfvec/binary) to cut storage and speed up search.

## How would I explain it in an interview?

"An embedding turns text into coordinates in a space where meaning is distance. I picked a small local
384-dim model so everything runs free and offline. The rules that matter: use the same model for queries and
documents, the dimension is baked into the schema, and switching models means re-embedding the whole corpus.
The measured cost of that choice is weak performance on pure paraphrase questions, which is exactly where a
better model or hybrid/rerank helps."
