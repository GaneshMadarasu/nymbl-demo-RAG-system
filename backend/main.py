import json
import logging
import logging.handlers
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend import db, ingest, query

_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3
)
_file_handler.setFormatter(_fmt)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)

FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"
VIEWER = Path(__file__).parent.parent / "frontend" / "viewer.html"
_PDF_PATH = Path(tempfile.gettempdir()) / "docrag_current.pdf"

_state: dict = {"doc_id": None, "chunk_count": 0, "k": 8}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pool = await db.get_pool()
        doc = await db.get_latest_doc(pool)
        if doc:
            _state["doc_id"] = doc["doc_id"]
            _state["chunk_count"] = doc["chunk_count"]
            _state["k"] = doc.get("k", 8)
            logger.info(
                "Restored doc_id=%s (%d chunks, k=%d)",
                doc["doc_id"],
                doc["chunk_count"],
                _state["k"],
            )
    except Exception:
        logger.warning("Could not restore document state from DB", exc_info=True)
    yield


app = FastAPI(title="DocRAG", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(
        "%s %s → %d %s", request.method, request.url.path, exc.status_code, exc.detail
    )
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.get("/")
async def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND)


@app.get("/viewer")
async def serve_viewer() -> FileResponse:
    return FileResponse(
        VIEWER, headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/doc/pdf")
async def serve_pdf() -> FileResponse:
    if not _state["doc_id"]:
        raise HTTPException(status_code=404, detail="No document loaded.")
    if not _PDF_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="PDF not available — re-upload the document.",
        )
    return FileResponse(
        _PDF_PATH, media_type="application/pdf", filename="document.pdf"
    )


@app.get("/doc/chunk/{chunk_index}")
async def get_chunk(chunk_index: int) -> dict:
    if not _state["doc_id"]:
        raise HTTPException(status_code=404, detail="No document loaded.")
    pool = await db.get_pool()
    result = await db.get_chunk_text(pool, _state["doc_id"], chunk_index)
    if not result:
        raise HTTPException(status_code=404, detail=f"Chunk {chunk_index} not found.")
    return result


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/doc/info")
async def doc_info() -> dict:
    if not _state["doc_id"]:
        return {"loaded": False}
    return {
        "loaded": True,
        "doc_id": _state["doc_id"],
        "chunk_count": _state["chunk_count"],
        "embedding_dim": 768,
    }


_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)) -> StreamingResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    pdf_bytes = await file.read()
    if len(pdf_bytes) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(pdf_bytes) // 1024 // 1024} MB). Maximum is 500 MB.",
        )

    async def stream():
        try:
            async for event in ingest.run_ingest(pdf_bytes):
                if event.get("status") == "done":
                    _state["doc_id"] = event["doc_id"]
                    _state["chunk_count"] = event["chunk_count"]
                    _state["k"] = event.get("k", 8)
                    _PDF_PATH.write_bytes(pdf_bytes)
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.exception("Ingest failed")
            yield f"data: {json.dumps({'status': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.delete("/doc")
async def clear_doc() -> dict:
    pool = await db.get_pool()
    await db.clear_all_chunks(pool)
    _state["doc_id"] = None
    _state["chunk_count"] = 0
    _state["k"] = 8
    _PDF_PATH.unlink(missing_ok=True)
    return {"cleared": True}


class QueryRequest(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/query")
async def query_doc(req: QueryRequest) -> StreamingResponse:
    if not _state["doc_id"]:
        raise HTTPException(
            status_code=400, detail="No document loaded. Upload a PDF first."
        )

    async def stream():
        try:
            async for event in query.run_query(
                req.question, _state["doc_id"], _state["k"], history=req.history
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.exception("Query stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
