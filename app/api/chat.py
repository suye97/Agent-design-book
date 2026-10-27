# -*- coding: utf-8 -*-
"""问答接口：/api/chat 与 /api/chat/stream。"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import get_components, unavailable
from app.logging_conf import get_logger, log_answer
from app.models import ChatRequest, ChatResponse, RouteKind

router = APIRouter(tags=["chat"])
logger = get_logger("api.chat")


def _answer_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def _prepare(req: ChatRequest):
    """路由 + 检索 + 生成，返回 (answer_id, answer, citations, route, used_llm, ok)。"""
    comps = get_components()
    book_override = req.book.value if req.book else None
    route = comps.route_question(req.question, book_override)
    citations = comps.retriever().retrieve(route.rewritten_query or req.question, route.route, req.top_k)
    answer, used_llm = comps.agent(route.route).answer(req.question, citations)
    ok = comps.verify_citations(answer, citations)
    if citations and "引用出处" not in answer:
        answer = answer.rstrip() + "\n\n### 引用出处\n\n" + comps.citations_markdown(citations)
    return route, citations, answer, used_llm, ok


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    answer_id = _answer_id()
    try:
        route, citations, answer, used_llm, ok = _prepare(req)
    except Exception as exc:  # 组件/外部服务不可用
        logger.exception("chat 失败: %s", exc)
        raise unavailable(exc, "请先执行 `python -m app.cli ingest` 并确认 Milvus/Embedding 可用")

    latency = int((time.perf_counter() - started) * 1000)
    log_answer(
        answer_id=answer_id,
        question=req.question,
        route=route,
        citations=citations,
        answer=answer,
        latency_ms=latency,
        used_llm=used_llm,
        citation_violation=not ok,
    )
    from app.config import settings

    return ChatResponse(
        answer_id=answer_id,
        answer=answer,
        citations=citations,
        route=route,
        latency_ms=latency,
        model=settings.llm_model if used_llm else "",
        used_llm=used_llm,
        citation_ok=ok,
        warning=None if citations else "知识库中没有检索到相关片段，请先入库或换个问法",
    )


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE：检索后先下发元信息，再透传模型实际增量输出。"""
    answer_id = _answer_id()
    started = time.perf_counter()
    try:
        comps = get_components()
        book_override = req.book.value if req.book else None
        route = comps.route_question(req.question, book_override)
        citations = comps.retriever().retrieve(
            route.rewritten_query or req.question, route.route, req.top_k
        )
        agent = comps.agent(route.route)
    except Exception as exc:
        logger.exception("chat/stream 准备失败: %s", exc)
        raise unavailable(exc, "请先执行 `python -m app.cli ingest`")

    def gen():
        yield _sse("meta", {"answer_id": answer_id, "route": route.model_dump(mode="json")})
        yield _sse("sources", {"citations": [c.model_dump(mode="json") for c in citations]})
        parts: list[str] = []
        used_llm = False
        stream_answer = getattr(agent, "stream_answer", None)
        if callable(stream_answer):
            stream = iter(stream_answer(req.question, citations))
            while True:
                try:
                    text = next(stream)
                except StopIteration as stop:
                    used_llm = bool(stop.value)
                    break
                if text:
                    parts.append(text)
                    yield _sse("token", {"text": text})
        else:  # 测试/自定义 Agent 的兼容路径
            answer, used_llm = agent.answer(req.question, citations)
            parts.append(answer)
            yield _sse("token", {"text": answer})
        if citations:
            sources = "\n\n**引用出处**\n" + comps.citations_markdown(citations)
            parts.append(sources)
            yield _sse("token", {"text": sources})
        answer = "".join(parts)
        ok = comps.verify_citations(answer, citations)
        latency = int((time.perf_counter() - started) * 1000)
        log_answer(
            answer_id=answer_id, question=req.question, route=route,
            citations=citations, answer=answer, latency_ms=latency,
            used_llm=used_llm, citation_violation=not ok,
        )
        yield _sse("done", {
            "answer_id": answer_id, "latency_ms": latency,
            "used_llm": used_llm, "citation_ok": ok,
        })

    return StreamingResponse(gen(), media_type="text/event-stream")
