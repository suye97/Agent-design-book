# -*- coding: utf-8 -*-
"""在线问答链路：路由 / 检索 / 双库 Agent / 引用审计。

对外公共接口（API 层 / CLI 直接使用）：

- :func:`app.rag.router.route_question` → ``RouteDecision``
- :class:`app.rag.retriever.Retriever` / :func:`app.rag.retriever.get_retriever`
- :class:`app.rag.agents.AnswerAgent` / :func:`app.rag.agents.get_agent`
- :func:`app.rag.agents.verify_citations` / :func:`app.rag.agents.build_citations_markdown`
- :class:`app.rag.llm.LlmUnavailable` / :func:`app.rag.llm.get_llm`
"""
from __future__ import annotations

from app.rag.agents import (
    AnswerAgent,
    BothAgent,
    DeepUnderstandingAgent,
    DesignPatternAgent,
    build_citations_markdown,
    get_agent,
    verify_citations,
)
from app.rag.citations import (
    assign_indices,
    dedupe_citations,
    make_excerpt,
    parse_citation_indices,
)
from app.rag.llm import LlmClient, LlmUnavailable, get_llm, reset_llm
from app.rag.retriever import Retriever, get_retriever, reset_retriever
from app.rag.router import route_question

__all__ = [
    "AnswerAgent",
    "BothAgent",
    "DeepUnderstandingAgent",
    "DesignPatternAgent",
    "LlmClient",
    "LlmUnavailable",
    "Retriever",
    "assign_indices",
    "build_citations_markdown",
    "dedupe_citations",
    "get_agent",
    "get_llm",
    "get_retriever",
    "make_excerpt",
    "parse_citation_indices",
    "reset_llm",
    "reset_retriever",
    "route_question",
    "verify_citations",
]
