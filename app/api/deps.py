# -*- coding: utf-8 -*-
"""API 依赖装配：懒加载流水线 / 检索 / Agent，避免应用启动时强依赖外部服务。

任何一层未就绪（例如尚未安装 rag 模块、Milvus 不可达），接口返回 503 并带
可读原因，而不是让整个服务起不来。
"""
from __future__ import annotations

import threading
from typing import Any

from fastapi import HTTPException

_lock = threading.Lock()
_cache: dict[str, Any] = {}


class Components:
    """进程级组件容器（惰性初始化）。"""

    def __init__(self) -> None:
        self._router = None
        self._retriever = None
        self._agents: dict[str, Any] = {}

    # ---- 路由 ----
    def route_question(self, question: str, book_override: str | None = None):
        if self._router is None:
            from app.rag.router import route_question  # 延迟导入

            self._router = route_question
        return self._router(question, book_override)

    # ---- 检索 ----
    def retriever(self):
        if self._retriever is None:
            from app.rag.retriever import get_retriever

            self._retriever = get_retriever()
        return self._retriever

    # ---- Agent ----
    def agent(self, route):
        key = getattr(route, "value", str(route))
        if key not in self._agents:
            from app.rag.agents import get_agent

            self._agents[key] = get_agent(route)
        return self._agents[key]

    # ---- 工具函数 ----
    @staticmethod
    def verify_citations(answer: str, citations) -> bool:
        from app.rag.agents import verify_citations

        return verify_citations(answer, citations)

    @staticmethod
    def citations_markdown(citations) -> str:
        from app.rag.agents import build_citations_markdown

        return build_citations_markdown(citations)


_components: Components | None = None


def get_components() -> Components:
    global _components
    if _components is None:
        with _lock:
            if _components is None:
                _components = Components()
    return _components


def unavailable(exc: Exception, hint: str = "") -> HTTPException:
    """把组件缺失/外部服务故障转成 503。"""
    detail = f"{type(exc).__name__}: {exc}"
    if hint:
        detail = f"{detail}. {hint}"
    return HTTPException(status_code=503, detail=detail)
