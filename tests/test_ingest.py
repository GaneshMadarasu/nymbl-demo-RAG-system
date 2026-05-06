from unittest.mock import AsyncMock, patch
from backend.ingest import _doc_id, run_ingest


def test_doc_id_is_deterministic():
    b = b"hello pdf content"
    assert _doc_id(b) == _doc_id(b)


def test_doc_id_differs_for_different_content():
    assert _doc_id(b"pdf one") != _doc_id(b"pdf two")


def test_doc_id_is_16_chars():
    assert len(_doc_id(b"anything")) == 16


async def test_run_ingest_yields_done_event(pool):
    fake_text = "This is a sentence about artificial intelligence. " * 20
    with (
        patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch(
            "backend.ingest._extract_text",
            new_callable=AsyncMock,
            return_value=fake_text,
        ),
        patch(
            "backend.ingest._embed_one",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
    ):
        events = [e async for e in run_ingest(b"fake-pdf-bytes")]
    statuses = [e["status"] for e in events]
    assert "done" in statuses


async def test_run_ingest_done_event_has_chunk_count(pool):
    fake_text = "Sentence about topic. " * 30
    with (
        patch("backend.ingest.db.get_pool", new_callable=AsyncMock, return_value=pool),
        patch(
            "backend.ingest._extract_text",
            new_callable=AsyncMock,
            return_value=fake_text,
        ),
        patch(
            "backend.ingest._embed_one",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
    ):
        events = [e async for e in run_ingest(b"fake-pdf-bytes")]
    done = next(e for e in events if e["status"] == "done")
    assert done["chunk_count"] > 0
    assert "doc_id" in done
