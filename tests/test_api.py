from httpx import AsyncClient, ASGITransport


async def test_health():
    from backend.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_query_without_doc_returns_400():
    from backend.main import app, _state

    _state["doc_id"] = None
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post("/query", json={"question": "what is this?"})
    assert r.status_code == 400


async def test_ingest_rejects_non_pdf():
    from backend.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.post(
            "/ingest",
            files={"file": ("notes.txt", b"hello world", "text/plain")},
        )
    assert r.status_code == 400


async def test_doc_info_no_doc():
    from backend.main import app, _state

    _state["doc_id"] = None
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/doc/info")
    assert r.status_code == 200
    assert r.json()["loaded"] is False


async def test_doc_info_with_doc():
    from backend.main import app, _state

    _state["doc_id"] = "abc123"
    _state["chunk_count"] = 42
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/doc/info")
    assert r.status_code == 200
    data = r.json()
    assert data["loaded"] is True
    assert data["chunk_count"] == 42


async def test_ingest_endpoint_accepts_ocr_scanned_form_field():
    from unittest.mock import AsyncMock, patch
    from httpx import AsyncClient, ASGITransport
    from backend.main import app

    async def fake_run_ingest(pdf_bytes, process_images=True, ocr_scanned=False):
        # Yield a minimal done event so the endpoint completes
        yield {
            "status": "done",
            "doc_id": "abc123",
            "chunk_count": 1,
            "k": 5,
            "message": "ok",
        }
        # Capture call args via the mock wrapper below

    with patch("backend.main.ingest.run_ingest", side_effect=fake_run_ingest) as m:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            resp = await ac.post(
                "/ingest",
                files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
                data={"process_images": "false", "ocr_scanned": "true"},
            )
    assert resp.status_code == 200
    # Verify ocr_scanned was passed through
    _, kwargs = m.call_args
    assert kwargs.get("ocr_scanned") is True
