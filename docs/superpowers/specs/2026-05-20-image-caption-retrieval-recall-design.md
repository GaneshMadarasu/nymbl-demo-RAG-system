# Image-caption retrieval recall — design

**Date:** 2026-05-20
**Status:** Approved — not yet implemented
**Scope:** Targeted change to the retrieval layer (`backend/db.py`, with deterministic tests). No change to ingestion, the caption pipeline, or the data model. No removal of existing behavior.

## Problem

The system captions every ingested image and stores the caption as a first-class chunk (embedded into pgvector + indexed in `tsvector`). The intent is that a natural-language query about visual content — *"show me a painting with a red coat"* — retrieves the matching image. In practice it returns **"I don't know."**

Debugging (2026-05-20) reproduced the failure against the live `paintings.pdf` (61 image chunks) and found the matching caption (chunk 56, *"a young man … dressed in a rich red cloak over a black tunic"*) is excluded from **both** retrieval paths:

```
Query: "show me a painting with a red coat"   (doc k=8 → hybrid uses top k*2=16 per path)

DENSE (vector):  ranks 1–25 are ALL text chunks (sim 0.54–0.61).
                 image 56 ranks BELOW 25 → never enters the dense CTE.
SPARSE (BM25):   plainto_tsquery → 'show' & 'paint' & 'red' & 'coat'   (ANDs every term)
                 → 0 rows: no caption contains all of {show, paint, red, coat}
                 ('coat' ≠ 'cloak', and 'show' isn't in the caption).
```

Probes confirmed the mechanism precisely — the image is retrievable **only when every content word of the query already appears in the caption**:

| Query | Image retrieved? | Why |
|---|---|---|
| `which painting has a man in a red cloak` | yes (chunk 56) | lexemes `man & red & cloak & paint` all in caption → BM25 matches |
| `red cloak` | yes (chunk 56) | `red & cloak` both in caption → BM25 matches |
| `show me a painting with a red cloak` | **no** | adds `show` → AND fails → BM25 zero; dense crowds it out |
| `show me a painting with a red coat` | **no** | `coat` ≠ `cloak` → AND fails → BM25 zero; dense crowds it out |

### Two root-cause mechanisms

1. **BM25 uses AND-semantics.** `plainto_tsquery` requires *all* query lexemes. A single synonym (`coat` vs `cloak`) or filler word (`show`) drops the caption to zero matches. Captions' biggest asset — literal color/object words — is disabled for ordinary phrasing.
2. **Dense retrieval is dominated by the document's own prose.** In a book that is *about* paintings, the ~256-token prose chunks are semantically closer to a painting query than a terse ~30-word caption. Image-caption similarities (~0.42–0.50) sit systematically **below** every text-chunk similarity (~0.55–0.61), so images never reach the top `k*2` window. (This is why the team previously bolted on the `GET /doc/images` "fuzzy-link" frontend workaround — a symptom of this gap.)

Because images always lose a *global* top-`k` race in art-heavy documents, OR-ing BM25 alone is insufficient; some form of guaranteed image representation is required.

## Goal

A natural-language query about visual content reliably retrieves the most relevant image chunks — even when the caption uses a synonym (`cloak`/`suit` vs `coat`) or the query carries filler words — without injecting irrelevant images into ordinary text Q&A, and without reintroducing LLM re-ranking.

## Non-goals (flagged follow-ups, deliberately out of scope)

- **Single-document `TRUNCATE` issue.** `run_ingest` calls `clear_all_chunks()` (`ingest.py:825`) → `TRUNCATE TABLE chunks` on every ingest, so ingesting any new PDF wipes the previous document, including its image chunks. This is a separate, real bug (it is *why* the art book appeared "gone" during debugging) and gets its own spec.
- **Synonym dictionaries / query expansion.** Empirically unnecessary: dense similarity already bridges `coat → cloak/suit` (see Calibration). Revisit only if calibration disproves it.
- **LLM re-ranking.** Deliberately removed earlier (added latency without quality lift on this corpus). Not reintroduced.

## Design

Two changes, both confined to the retrieval layer.

### Mechanism A — OR-based BM25 (recall)

In `_HYBRID_SQL`, replace the AND-semantics tsquery with an OR'd one, reusing Postgres's own normalization and stop-word removal:

```sql
-- before
... plainto_tsquery('english', $4) ...
-- after
... replace(plainto_tsquery('english', $4)::text, ' & ', ' | ')::tsquery ...
```

`plainto_tsquery('english','show me a painting with a red coat')` → `'show' & 'paint' & 'red' & 'coat'`; after the swap → `'show' | 'paint' | 'red' | 'coat'`. A caption now matches if it contains **any** query lexeme, ranked by `ts_rank_cd` (captions matching more / rarer terms rank higher). Applies everywhere `$4` is used in the `sparse` CTE (the `@@` filter, the `ts_rank_cd` ORDER BY, and the inner ranking).

Edge case: a stop-word-only query yields an empty `plainto_tsquery`; the `replace` leaves it empty and it casts to an empty `tsquery` that matches nothing — the `sparse` CTE is simply empty and `dense` still runs. No special-casing needed.

### Mechanism B — reserved image slots with a similarity floor (load-bearing)

Add a guaranteed image path so the best-matching captions are always considered, gated by a floor so ordinary text questions aren't polluted.

**Contract for `search_chunks(pool, doc_id, query_embedding, question, k)`:**

> Returns at most `k` chunks. If the document contains image chunks whose **dense** cosine similarity to the query is `≥ IMAGE_FLOOR`, then at least `min(N_img, #qualifying)` of the returned chunks are the highest-similarity such images — even if hybrid fusion would otherwise rank them outside the top `k`. The remaining slots are filled by the existing hybrid fusion (now using OR-BM25). Results are de-duplicated by `chunk_index`. When the document has no qualifying image chunks, behavior is identical to today.

Reserved images **displace the lowest-ranked non-image fused results** to stay within `k`; they never displace a higher-similarity image. This keeps the context window (and the tuned `k`) unchanged.

Implementation note (for the plan, not binding): add an `img` CTE to `_HYBRID_SQL` selecting the top `N_img` image chunks by dense similarity with `similarity ≥ IMAGE_FLOOR`, and merge in `search_chunks` so the reserved images are guaranteed within the returned `k`. Keep the merge in one place (`search_chunks`) so the guarantee is testable in isolation.

**Row-shape constraint.** `query.py` serializes every returned chunk into the `sources` SSE event and reads `c["similarity"]` and `c["rrf_score"]` (`query.py:198–199`). Reserved image rows must therefore carry the same columns as the existing fused rows — `chunk_index, text, parent_text, page_number, chunk_type, similarity, rrf_score`. Set a reserved image's `similarity` to its dense cosine and its `rrf_score` to an RRF-style value derived from its dense rank (`1.0 / (60.0 + dense_rank)`), so existing serialization, ordering, and the `[Chunk N]` labelling continue to work unchanged.

### Self-gating (why no intent classifier)

- **Text-only docs** (e.g. the loaded *Algorithm Design* textbook — 0 image chunks): the `img` path returns nothing → exact current behavior.
- **Textual question on a mixed doc**: the top image's similarity falls below `IMAGE_FLOOR` → no images reserved.
- **Visual question**: relevant captions clear the floor → reserved.

The floor does the gating that an intent model would, with no extra moving parts.

## Parameters

Module-level constants in `backend/db.py` (matching the existing `_MIN_IMAGE_PX` / `_GEMINI_CONCURRENCY` style):

| Constant | Default | Meaning |
|---|---|---|
| `N_IMG_RESERVED` | `3` | Max image slots reserved per query. |
| `IMAGE_FLOOR` | `0.40` (to be calibrated) | Min dense cosine similarity for an image to be eligible for reservation. |

## Calibration (done during implementation)

`IMAGE_FLOOR` must be set from data, not guessed. Measured against `paintings.pdf`:

- Visual queries — top relevant image similarities ≈ **0.45–0.50** (e.g. for "red coat": chunk 83 *"young man in a red suit"* = 0.504, chunk 56 *"red cloak"* = 0.459). These must be **admitted**.
- A clearly textual query (e.g. "who published this book") — measure its top-image similarity; the floor must sit **above** that value so images are **rejected**.

Pick `IMAGE_FLOOR` in the gap between those two bands (the `0.40` default is a starting point pending the textual-query measurement). Record the chosen value and its justification in code comments.

Confirmed by calibration data: for "red coat", the top images by dense similarity are a *red suit* and a *red cloak* — the synonym gap is bridged by dense similarity alone, validating the "no synonym dictionary" non-goal.

## Testing

Tests run against the real Postgres `pool` fixture (`tests/conftest.py`) with Gemini mocked — so retrieval logic is tested deterministically with hand-crafted embeddings and captions (the pattern already used in `tests/test_db.py`).

New tests in `tests/test_db.py`:

1. **OR-BM25 partial match** — insert a caption "a man in a red cloak"; query with lexemes spanning `red coat`; assert the row matches via "red" (would not match under AND-semantics).
2. **Image reservation beats crowding** — insert several text chunks with high-similarity embeddings and one image chunk with a lower-but-`≥ floor` embedding; assert the image chunk is present in the `k` results despite losing the global similarity race.
3. **Floor gating** — insert an image chunk with a below-floor embedding; assert it is **not** reserved/returned for an otherwise-textual query.
4. **No-image doc unchanged** — a text-only doc returns exactly the pre-change results (guards the "identical when no qualifying images" clause).

Regression: existing `tests/test_db.py` and `tests/test_query.py` must still pass.

Manual end-to-end (live `paintings.pdf`, already ingested): `POST /query "show me a painting with a red coat"` → expect ≥ 1 image chunk (a red-clothing painting) in `sources`, a `Passing N image(s) to Gemini` log line, and an answer that is **not** "I don't know."

## Files affected

- `backend/db.py` — `_HYBRID_SQL` (OR-BM25 swap + `img` CTE), `search_chunks` (merge/reserve logic), new `N_IMG_RESERVED` / `IMAGE_FLOOR` constants.
- `backend/query.py` — only if the parameters are threaded through; otherwise unchanged (defaults live in `db.py`).
- `tests/test_db.py` — the four new tests above; possibly an image-in-context assertion in `tests/test_query.py`.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Reserved images add noise to text-heavy answers | `IMAGE_FLOOR` gating; `SYSTEM_PROMPT` already constrains the model to answer only from context and to say "I don't know" if insufficient, so a marginal image cannot cause a hallucinated answer. |
| OR-BM25 lowers precision (more rows match) | Only affects the `sparse` candidate set (capped at `k*2`); `ts_rank_cd` ranking + RRF fusion + the `k` limit absorb it. Dense path unchanged. |
| Floor mis-set (too low → pollution; too high → misses) | Calibrated against measured visual vs textual similarity bands before merge; value + rationale documented in code. |
| `::tsquery` cast on malformed input | Input is the output of `plainto_tsquery` (already normalized/validated); the only degenerate case (empty) is handled. |

## Verification of done

1. New deterministic tests pass; full suite green.
2. Live `paintings.pdf`: "show me a painting with a red coat" returns a red-clothing image and a substantive answer (not "I don't know").
3. A textual query on the same doc returns **no** images (floor gating holds).
