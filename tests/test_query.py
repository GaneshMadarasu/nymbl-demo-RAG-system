from unittest.mock import AsyncMock, MagicMock, patch
from backend.query import build_prompt, run_query, SYSTEM_PROMPT


def test_build_prompt_includes_system_prompt():
    chunks = [{"chunk_index": 0, "text": "AI is transformative."}]
    prompt = build_prompt("What is AI?", chunks)
    assert SYSTEM_PROMPT in prompt


def test_build_prompt_formats_chunk_references():
    chunks = [
        {"chunk_index": 3, "text": "Deep learning uses neural networks."},
        {"chunk_index": 17, "text": "Transformers changed NLP."},
    ]
    prompt = build_prompt("What changed NLP?", chunks)
    assert "[Chunk 3]" in prompt
    assert "Deep learning uses neural networks." in prompt
    assert "[Chunk 17]" in prompt
    assert "What changed NLP?" in prompt


def test_build_prompt_includes_question():
    chunks = [{"chunk_index": 0, "text": "some text"}]
    prompt = build_prompt("What are the key findings?", chunks)
    assert "What are the key findings?" in prompt


async def test_run_query_yields_sources_and_done(pool):
    from backend.db import insert_chunks
    import backend.query as bq

    await insert_chunks(
        pool, "qdoc1", [(0, "AI is great for research.", None, [0.1] * 768)]
    )
    mock_chunk = MagicMock()
    mock_chunk.text = "AI is indeed great."

    async def _fake_stream(*args, **kwargs):
        async def _iter():
            yield mock_chunk

        return _iter()

    with (
        patch("backend.query.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch(
            "backend.query._embed_query",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
        patch.object(
            bq._client.aio.models, "generate_content_stream", side_effect=_fake_stream
        ),
    ):
        events = [e async for e in run_query("What is AI?", "qdoc1")]
    event_types = [e["type"] for e in events]
    assert "sources" in event_types
    assert "token" in event_types
    assert "done" in event_types


async def test_run_query_returns_i_dont_know_when_no_chunks(pool):
    with (
        patch("backend.query.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch(
            "backend.query._embed_query",
            new_callable=AsyncMock,
            return_value=[0.5] * 768,
        ),
    ):
        events = [
            e async for e in run_query("What is the meaning of life?", "empty_doc")
        ]
    token_events = [e for e in events if e["type"] == "token"]
    assert any("I don't know" in e["text"] for e in token_events)
