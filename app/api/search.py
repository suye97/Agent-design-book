# -*- coding: utf-8 -*-
"""检索接口：/api/search（只检索不生成，便于验证召回质量）。"""
from __future__ import annotations

import time

from fastapi import APIRouter

from app.api.deps import get_components, unavailable
from app.models import RouteKind, SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


def _books_for(req: SearchRequest) -> list[str]:
    if req.book:
        return [req.book.value]
    if req.route == RouteKind.DESIGN:
        return ["design"]
    if req.route == RouteKind.DEEP:
        return ["deep"]
    return ["design", "deep"]


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    started = time.perf_counter()
    try:
        from app.ingest.embed import get_embedder
        from app.ingest.milvus_store import get_store

        books = _books_for(req)
        embedder = get_embedder()
        if embedder.mock:
            raise RuntimeError("真实 Embedding 不可用，不能查询正式知识库")
        vector = embedder.encode_query(req.query)
        hits = get_store().search(req.query, vector, books, top_k=req.top_k)
    except Exception as exc:
        raise unavailable(exc, "请先执行 `python -m app.cli ingest`")

    return SearchResponse(
        query=req.query,
        hits=hits,
        latency_ms=int((time.perf_counter() - started) * 1000),
        filters={"books": books, "top_k": req.top_k},
    )
