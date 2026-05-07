import pytest
from backend.db import insert_chunks, search_chunks, clear_all_chunks, get_doc_info


# helpers — (idx, text, parent_text, embedding, page_number)
def _row(idx, text, emb):
    return (idx, text, None, emb, None)


async def test_insert_and_search(pool):
    await insert_chunks(pool, "doc1", [_row(0, "The cat sat on the mat.", [0.1] * 768)])
    results = await search_chunks(pool, "doc1", [0.1] * 768, k=1)
    assert len(results) == 1
    assert results[0]["text"] == "The cat sat on the mat."
    assert results[0]["chunk_index"] == 0


async def test_search_returns_most_similar_first(pool):
    await insert_chunks(
        pool,
        "doc2",
        [
            _row(0, "chunk zero", [1.0] + [0.0] * 767),
            _row(1, "chunk one", [0.0] + [1.0] + [0.0] * 766),
        ],
    )
    results = await search_chunks(pool, "doc2", [1.0] + [0.0] * 767, k=2)
    assert results[0]["chunk_index"] == 0


async def test_search_hybrid_returns_results(pool):
    await insert_chunks(
        pool, "doc5", [_row(0, "Quantum computing is fast.", [0.5] * 768)]
    )
    results = await search_chunks(
        pool, "doc5", [0.5] * 768, question="quantum computing", k=1
    )
    assert len(results) == 1
    assert "similarity" in results[0]


async def test_clear_all_removes_everything(pool):
    await insert_chunks(pool, "doc3", [_row(0, "text", [0.2] * 768)])
    await clear_all_chunks(pool)
    results = await search_chunks(pool, "doc3", [0.2] * 768, k=1)
    assert results == []


async def test_get_doc_info_returns_count(pool):
    await insert_chunks(
        pool, "doc4", [_row(0, "a", [0.3] * 768), _row(1, "b", [0.4] * 768)]
    )
    info = await get_doc_info(pool, "doc4")
    assert info is not None
    assert info["chunk_count"] == 2
    assert info["embedding_dim"] == 768


async def test_get_doc_info_returns_none_for_missing(pool):
    assert await get_doc_info(pool, "no_such_doc") is None
