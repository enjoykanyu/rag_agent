from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.agent.graph import create_rag_graph
from src.agent.streaming import StreamingRAGAgent
from src.config import AppConfig, load_config
from src.pipeline.rag_pipeline import RAGPipeline
from src.session_manager import SessionManager

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def create_app(config_path: str | None = None) -> FastAPI:
    config = load_config(config_path)

    app = FastAPI(title=config.ui.title, version="3.0.0")

    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    pipeline = RAGPipeline(config)
    graph = create_rag_graph(config, pipeline)
    session_manager = SessionManager(storage_dir=config.session.storage_dir)
    agent = StreamingRAGAgent(graph, config, session_manager)

    chunk_count = pipeline.index_local()
    logger.info("Initial index: %d chunks from local documents", chunk_count)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        template_path = BASE_DIR / "templates" / "index.html"
        return template_path.read_text(encoding="utf-8")

    @app.post("/api/ask")
    async def ask_question(question: str = Form(...), session_id: str = Form("default")):
        if not question.strip():
            return JSONResponse({"error": "问题不能为空"}, status_code=400)

        async def event_generator():
            async for event in agent.ask_stream(question, session_id=session_id):
                yield event

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/upload")
    async def upload_document(file: UploadFile = File(...)):
        ext = Path(file.filename or "").suffix.lower()
        if ext not in config.sources.upload.allowed_extensions:
            return JSONResponse(
                {"error": f"不支持的文件类型: {ext}，仅支持 {config.sources.upload.allowed_extensions}"},
                status_code=400,
            )
        content = await file.read()
        count = pipeline.index_upload(file.filename or "unknown", content)
        return JSONResponse({"message": f"已索引 {count} 个文本块", "chunks": count})

    @app.post("/api/reindex")
    async def reindex():
        pipeline.store.clear()
        pipeline._documents.clear()
        pipeline._all_chunks.clear()
        count = pipeline.index_local()
        return JSONResponse({"message": f"重新索引完成，共 {count} 个文本块", "chunks": count})

    @app.post("/api/reset")
    async def reset_conversation(session_id: str = "default"):
        agent.reset_conversation(session_id)
        return JSONResponse({"message": "对话已重置"})

    @app.get("/api/status")
    async def status():
        return JSONResponse({
            "documents": pipeline.document_count,
            "chunks": pipeline.chunk_count,
            "embedding": config.rag.embedding,
            "llm_model": config.agent.llm.model,
            "llm_configured": bool(config.agent.llm.api_key),
            "milvus_enabled": bool(config.rag.milvus_uri),
        })

    @app.post("/api/sessions")
    async def create_session(title: str = "新会话"):
        record = session_manager.create_session(title=title)
        return JSONResponse(record)

    @app.get("/api/sessions")
    async def list_sessions():
        sessions = session_manager.list_sessions()
        return JSONResponse(sessions)

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        info = session_manager.get_session_info(session_id)
        return JSONResponse(info)

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        session_manager.delete_session(session_id)
        agent.reset_conversation(session_id)
        return JSONResponse({"message": "会话已删除"})

    @app.put("/api/sessions/{session_id}/title")
    async def rename_session(session_id: str, title: str = Form(...)):
        record = session_manager.rename_session(session_id, title)
        return JSONResponse(record)

    return app


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = os.environ.get("RAG_CONFIG", None)
    app = create_app(config_path)

    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
