# -*- coding: utf-8 -*-
"""FastAPI 应用入口。"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import admin, chat, health, search
from app.config import settings
from app.logging_conf import configure_logging

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app() -> FastAPI:
    configure_logging()
    logging.getLogger(__name__).info(
        "启动 Agent 设计问答服务 v%s | milvus=%s | collection=%s | llm=%s",
        __version__,
        settings.milvus_uri,
        settings.milvus_collection,
        "on" if settings.llm_enabled else "off",
    )

    app = FastAPI(
        title="Agent 设计问答 Agent",
        description="基于《智能体设计模式》与《深入理解 AI Agent》的 RAG 问答服务",
        version=__version__,
    )
    app.include_router(health.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(admin.router, prefix="/api")

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
