# -*- coding: utf-8 -*-
"""API 层契约测试：用假组件替换检索/Agent，不依赖 Milvus、Embedding、LLM。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.config import settings
from app.main import app
from app.models import (
    ChunkType,
    Citation,
    Intent,
    RouteDecision,
    RouteKind,
    RouterSource,
)


def make_citation(index: int = 1) -> Citation:
    return Citation(
        index=index,
        chunk_id=f"chunk-{index}",
        book="design",
        book_title="智能体设计模式：智能系统构建实战指南",
        chapter_no=5,
        chapter_title="工具使用（函数调用）",
        section_no="",
        section_title="",
        page=67,
        page_pdf=80,
        para_start=3,
        para_end=3,
        chunk_type=ChunkType.TEXT,
        excerpt="工具使用模式让模型在推理过程中调用外部函数，从而突破自身能力边界。",
        score=0.91,
        rank=1,
    )


class FakeComponents:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def route_question(self, question: str, book_override: str | None = None):
        if self.fail:
            raise RuntimeError("milvus down")
        return RouteDecision(
            route=RouteKind.DESIGN,
            intent=Intent.CONCEPT,
            confidence=0.86,
            source=RouterSource.LLM,
            rewritten_query=question,
        )

    def retriever(self):
        return self

    def retrieve(self, question: str, route, top_k=None):
        return [make_citation(1), make_citation(2)]

    def agent(self, route):
        return self

    def answer(self, question: str, citations):
        return "工具使用模式让模型调用外部函数完成实际动作 [1][2]。", True

    def verify_citations(self, answer: str, citations) -> bool:
        return True

    def citations_markdown(self, citations) -> str:
        return "\n".join(f"[{c.index}] {c.label()}" for c in citations)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(deps, "_components", FakeComponents())
    # 测试产生的审计日志写到临时目录，避免污染 logs/ 下的真实问答记录
    monkeypatch.setattr(settings, "logs_dir", tmp_path)
    with TestClient(app) as c:
        yield c


def test_admin_ingest_requires_token(client, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", "admin-test-key")
    assert client.post("/api/admin/ingest", json={"dry_run": True}).status_code == 401
    assert client.get("/api/admin/ingest/example").status_code == 401


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"]
    assert {"milvus", "collection_ready", "embedding_configured", "llm_configured"} <= set(body)


def test_chat_returns_answer_with_citations(client):
    resp = client.post("/api/chat", json={"question": "工具使用模式是什么？"})
    assert resp.status_code == 200
    body = resp.json()
    assert "工具使用模式" in body["answer"]
    assert body["route"]["route"] == "design"
    assert body["route"]["confidence"] == pytest.approx(0.86)
    assert len(body["citations"]) == 2
    assert body["citation_ok"] is True
    assert body["used_llm"] is True
    # 引用元数据必须齐备：书 / 章 / 页 / 段
    cit = body["citations"][0]
    assert cit["book_title"].startswith("智能体设计模式")
    assert cit["chapter_no"] == 5
    assert cit["page"] == 67
    assert cit["para_start"] == 3
    assert "引用出处" in body["answer"]


def test_chat_uses_rewritten_query_for_retrieval(client, monkeypatch):
    fake = FakeComponents()
    used_queries = []
    fake.route_question = lambda *_args: RouteDecision(
        route=RouteKind.DESIGN,
        rewritten_query="工具使用模式",
    )
    fake.retrieve = lambda query, *_args: used_queries.append(query) or [make_citation(1)]
    monkeypatch.setattr(deps, "_components", fake)

    response = client.post("/api/chat", json={"question": "设计模式那本书里工具使用模式是什么？"})

    assert response.status_code == 200
    assert used_queries == ["工具使用模式"]


def test_chat_stream_emits_actual_agent_deltas_then_sources(client, monkeypatch):
    class StreamingComponents(FakeComponents):
        def stream_answer(self, question: str, citations):
            yield "工具使用模式"
            yield "允许调用外部函数[1][2]。"
            return True

    monkeypatch.setattr(deps, "_components", StreamingComponents())

    response = client.post("/api/chat/stream", json={"question": "工具使用模式"})

    assert response.status_code == 200
    events = response.text.split("\n\n")
    assert events[0].startswith("event: meta\n")
    assert events[1].startswith("event: sources\n")
    token_events = [event for event in events if event.startswith("event: token\n")]
    assert len(token_events) == 3
    assert "工具使用模式" in token_events[0]
    assert "允许调用外部函数" in token_events[1]
    assert "引用出处" in token_events[2]
    assert "event: done" in events[-2]
    assert '"used_llm": true' in events[-2]
    assert '"citation_ok": true' in events[-2]


def test_chat_embeds_citations_markdown(client):
    body = client.post("/api/chat", json={"question": "q"}).json()
    assert "第 67 页" in body["answer"]


def test_chat_returns_503_when_pipeline_unavailable(monkeypatch):
    monkeypatch.setattr(deps, "_components", FakeComponents(fail=True))
    with TestClient(app) as c:
        resp = c.post("/api/chat", json={"question": "q"})
    assert resp.status_code == 503
    assert "milvus down" in resp.json()["detail"]


def test_chat_request_validation(client):
    resp = client.post("/api/chat", json={"question": ""})
    assert resp.status_code == 422


def test_index_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Agent 设计问答" in resp.text
