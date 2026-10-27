# -*- coding: utf-8 -*-
"""健康检查：服务 / Milvus / Embedding / LLM 配置自检。"""
from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.config import settings
from app.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    milvus_ok = False
    collection_ready = False

    def _masked_uri(uri: str) -> str:
        """健康检查对外只回显“已配置”，不暴露 Milvus 地址。"""
        return "<configured>" if uri else "<unset>"

    detail: dict[str, object] = {
        "milvus_uri": _masked_uri(settings.milvus_uri),
        "collection": settings.milvus_collection,
    }
    try:
        from app.ingest.milvus_store import get_store

        store = get_store()
        milvus_ok = store.has_collection()
        if milvus_ok:
            detail["row_count"] = store.count()
            collection_ready = True
        else:
            detail["collection_error"] = "collection 尚未创建"
    except Exception as exc:
        detail["milvus_error"] = f"{type(exc).__name__}: {exc}"

    if settings.llm_enabled:
        detail["llm_model"] = settings.llm_model
    if settings.embedding_mock:
        detail["embedding_mode"] = "mock"
    if not settings.embedding_api_key and not settings.embedding_mock:
        detail["embedding_hint"] = "未配置 EMBEDDING_API_KEY，且未启用 EMBEDDING_MOCK"

    status = "ok" if milvus_ok and collection_ready else "degraded"
    return HealthResponse(
        status=status,
        version=__version__,
        milvus=milvus_ok,
        collection_ready=collection_ready,
        embedding_configured=settings.embedding_enabled,
        embedding_mock=settings.embedding_mock,
        llm_configured=settings.llm_enabled,
        detail=detail,
    )
