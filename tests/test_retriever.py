# -*- coding: utf-8 -*-
"""检索装配单测：同章去重 / 相邻块扩展 / 引用编号连续 / 路由过滤 / 降级。

全部使用 FakeEmbedder + FakeStore 注入，绝不连接真实 Milvus / Embedding 服务。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.config import settings
from app.models import (
    BOOK_TITLES,
    BookKey,
    Chunk,
    ChunkType,
    RouteKind,
    SearchHit,
)
from app.rag.retriever import Retriever, get_retriever, reset_retriever

DESIGN_TITLE = BOOK_TITLES[BookKey.DESIGN.value]
DEEP_TITLE = BOOK_TITLES[BookKey.DEEP.value]

LONG_CONTENT = (
    "工具使用模式让模型在推理过程中调用外部函数，从而把语言能力延伸到真实世界。"
    "实现时需要定义函数签名、控制调用时机，并把结果回填到对话上下文，同时注意幂等与错误处理。"
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    dim = 4

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.error: Exception | None = None

    def encode(self, texts):
        return [[0.1] * self.dim for _ in texts]

    def encode_query(self, text: str):
        if self.error is not None:
            raise self.error
        self.queries.append(text)
        return [0.1] * self.dim


class FakeStore:
    def __init__(self, hits, chunks=None, error: Exception | None = None) -> None:
        self.hits = list(hits)
        self.chunks = {chunk.chunk_id: chunk for chunk in (chunks or [])}
        self.error = error
        self.search_calls: list[dict] = []
        self.fetch_calls: list[list[str]] = []

    def search(self, query_text, query_vector, books, top_k=10):
        self.search_calls.append(
            {
                "query_text": query_text,
                "query_vector": list(query_vector),
                "books": list(books),
                "top_k": top_k,
            }
        )
        if self.error is not None:
            raise self.error
        filtered = [hit for hit in self.hits if hit.book in books]
        return filtered[:top_k]

    def fetch(self, ids):
        self.fetch_calls.append(list(ids))
        return {chunk_id: self.chunks[chunk_id] for chunk_id in ids if chunk_id in self.chunks}

    def count(self) -> int:
        return len(self.chunks)


def make_chunk(
    chunk_id: str,
    *,
    book: str = "design",
    chapter_no: int = 5,
    chapter_title: str = "工具使用（函数调用）",
    page_pdf: int = 80,
    page_print: int = 67,
    para: int = 3,
    content: str = LONG_CONTENT,
    prev_id: str | None = None,
    next_id: str | None = None,
    chunk_type: ChunkType = ChunkType.TEXT,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        book=book,
        book_title=BOOK_TITLES[book],
        chapter_no=chapter_no,
        chapter_title=chapter_title,
        section_no="5.2",
        section_title="工具调用流程",
        page_start=page_pdf,
        page_end=page_pdf,
        page_print_start=page_print,
        page_print_end=page_print,
        para_start=para,
        para_end=para,
        chunk_type=chunk_type,
        content=content,
        content_hash=f"hash-{chunk_id}",
        token_len=80,
        prev_id=prev_id,
        next_id=next_id,
    )


def make_hit(
    chunk_id: str,
    *,
    book: str = "design",
    chapter_no: int = 5,
    chapter_title: str = "工具使用（函数调用）",
    page: int = 67,
    para: int = 3,
    score: float = 0.9,
    content: str = LONG_CONTENT,
    chunk_type: ChunkType = ChunkType.TEXT,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        book=book,
        book_title=BOOK_TITLES[book],
        chapter_no=chapter_no,
        chapter_title=chapter_title,
        section_no="5.2",
        section_title="工具调用流程",
        page=page,
        para_start=para,
        para_end=para,
        chunk_type=chunk_type,
        excerpt=content[:40],
        content=content,
        score=score,
    )


def build_retriever(store: FakeStore, **kwargs) -> Retriever:
    return Retriever(embedder=FakeEmbedder(), store=store, **kwargs)


# --------------------------------------------------------------------------- #
# 基础装配
# --------------------------------------------------------------------------- #
def test_retrieve_returns_numbered_citations() -> None:
    hits = [
        make_hit("c5", chapter_no=5, page=67, score=0.9),
        make_hit("c6", chapter_no=6, chapter_title="MCP", page=122, para=1, score=0.6),
    ]
    chunks = [
        make_chunk("c5", chapter_no=5, page_pdf=80, page_print=67),
        make_chunk("c6", chapter_no=6, chapter_title="MCP", page_pdf=135, page_print=122, para=1),
    ]
    store = FakeStore(hits, chunks)
    retriever = build_retriever(store, neighbor_expand=0)

    citations = retriever.retrieve("工具使用模式怎么用", RouteKind.DESIGN)

    assert len(citations) == 2
    assert [c.index for c in citations] == [1, 2]
    assert [c.rank for c in citations] == [1, 2]
    assert [c.chunk_id for c in citations] == ["c5", "c6"]
    # 分数降序
    assert citations[0].score >= citations[1].score
    first = citations[0]
    assert first.book == "design"
    assert first.book_title == DESIGN_TITLE
    assert first.chapter_no == 5
    assert first.chapter_title == "工具使用（函数调用）"
    assert first.page == 67          # 印刷页码
    assert first.page_pdf == 80      # PDF 页码
    assert first.para_start == 3
    assert 20 <= len(first.excerpt) <= 40
    assert first.excerpt in LONG_CONTENT or LONG_CONTENT.startswith(first.excerpt.rstrip("…"))
    # 全文供 Agent 组 prompt；序列化（API 响应/log_answer）不含 content
    assert first.content == LONG_CONTENT
    assert "content" not in first.model_dump(mode="json")


def test_retrieve_uses_default_top_k() -> None:
    store = FakeStore([], [])
    retriever = build_retriever(store)

    retriever.retrieve("提示链", RouteKind.DESIGN)

    assert store.search_calls[-1]["top_k"] == settings.top_k
    assert store.search_calls[-1]["query_text"] == "提示链"
    assert store.search_calls[-1]["query_vector"] == [0.1, 0.1, 0.1, 0.1]


@pytest.mark.parametrize(
    "route,expected_books",
    [
        (RouteKind.DESIGN, ["design"]),
        (RouteKind.DEEP, ["deep"]),
        (RouteKind.BOTH, ["design", "deep"]),
    ],
)
def test_retrieve_filters_by_route(route: RouteKind, expected_books: list[str]) -> None:
    store = FakeStore([], [])
    retriever = build_retriever(store)

    retriever.retrieve("记忆管理", route)

    assert store.search_calls[-1]["books"] == expected_books


def test_retrieve_accepts_plain_string_route() -> None:
    store = FakeStore([], [])
    retriever = build_retriever(store)

    retriever.retrieve("记忆管理", "deep")

    assert store.search_calls[-1]["books"] == ["deep"]


def test_retrieve_passes_explicit_top_k() -> None:
    store = FakeStore([], [])
    retriever = build_retriever(store)

    retriever.retrieve("提示链", RouteKind.DESIGN, top_k=5)

    assert store.search_calls[-1]["top_k"] == 5


# --------------------------------------------------------------------------- #
# 同章去重
# --------------------------------------------------------------------------- #
def test_same_chapter_cap() -> None:
    hits = [
        make_hit(f"c{i}", chapter_no=5, para=i, score=0.9 - i * 0.05) for i in range(1, 6)
    ]
    chunks = [make_chunk(f"c{i}", chapter_no=5, para=i) for i in range(1, 6)]
    store = FakeStore(hits, chunks)
    retriever = build_retriever(store, max_per_chapter=3, neighbor_expand=0)

    citations = retriever.retrieve("工具使用模式", RouteKind.DESIGN)

    assert len(citations) == 3
    assert [c.chunk_id for c in citations] == ["c1", "c2", "c3"]
    assert [c.index for c in citations] == [1, 2, 3]


def test_cap_does_not_drop_other_chapters() -> None:
    hits = [
        make_hit("a1", chapter_no=1, para=1, score=0.9),
        make_hit("a2", chapter_no=1, para=2, score=0.8),
        make_hit("a3", chapter_no=1, para=3, score=0.7),
        make_hit("b1", chapter_no=2, chapter_title="路由", para=1, score=0.6),
    ]
    chunks = [
        make_chunk("a1", chapter_no=1, para=1),
        make_chunk("a2", chapter_no=1, para=2),
        make_chunk("a3", chapter_no=1, para=3),
        make_chunk("b1", chapter_no=2, chapter_title="路由", para=1),
    ]
    retriever = build_retriever(FakeStore(hits, chunks), max_per_chapter=2, neighbor_expand=0)

    citations = retriever.retrieve("工具使用模式", RouteKind.DESIGN)

    assert [c.chunk_id for c in citations] == ["a1", "a2", "b1"]


# --------------------------------------------------------------------------- #
# 相邻块扩展
# --------------------------------------------------------------------------- #
def test_neighbor_expansion_adds_prev_and_next() -> None:
    hits = [make_hit("c2", chapter_no=5, para=2, score=0.9)]
    chunks = [
        make_chunk("c1", chapter_no=5, para=1, prev_id=None, next_id="c2"),
        make_chunk("c2", chapter_no=5, para=2, prev_id="c1", next_id="c3"),
        make_chunk("c3", chapter_no=5, para=3, prev_id="c2", next_id=None),
    ]
    store = FakeStore(hits, chunks)
    retriever = build_retriever(store, neighbor_expand=1)

    citations = retriever.retrieve("工具使用模式", RouteKind.DESIGN)

    assert [c.chunk_id for c in citations] == ["c2", "c1", "c3"]
    assert [c.index for c in citations] == [1, 2, 3]
    assert citations[0].score == pytest.approx(0.9)
    assert citations[1].score == pytest.approx(0.9 * 0.9)
    assert citations[2].score == pytest.approx(0.9 * 0.9)


def test_neighbor_expansion_skips_references_chunk() -> None:
    """参考文献块即使与正文相邻，也不能通过邻接扩展进入上下文。"""
    hits = [make_hit("c2", chapter_no=5, para=2, score=0.9)]
    chunks = [
        make_chunk(
            "c1",
            chapter_no=5,
            para=1,
            next_id="c2",
            chunk_type=ChunkType.REFERENCES,
            content="参考文献 [1] Author. Title. 2025.",
        ),
        make_chunk("c2", chapter_no=5, para=2, prev_id="c1", next_id="c3"),
        make_chunk("c3", chapter_no=5, para=3, prev_id="c2"),
    ]
    store = FakeStore(hits, chunks)
    retriever = build_retriever(store, neighbor_expand=1)

    citations = retriever.retrieve("工具使用模式", RouteKind.DESIGN)

    ids = [c.chunk_id for c in citations]
    assert "c2" in ids
    assert "c1" not in ids, "参考文献块不应通过邻接扩展混入引用"
    assert "c3" in ids


def test_citation_content_carries_full_text_for_primaries_and_neighbors() -> None:
    primary_content = "主块全文开头。" + "主块正文内容" * 120
    neighbor_content = "相邻块全文开头。" + "邻接补全内容" * 120
    hits = [make_hit("c2", chapter_no=5, para=2, score=0.9, content=primary_content)]
    chunks = [
        make_chunk("c1", chapter_no=5, para=1, content=neighbor_content, next_id="c2"),
        make_chunk(
            "c2",
            chapter_no=5,
            para=2,
            content=primary_content,
            prev_id="c1",
            next_id=None,
        ),
    ]
    retriever = build_retriever(FakeStore(hits, chunks), neighbor_expand=1)

    citations = retriever.retrieve("工具使用模式", RouteKind.DESIGN)
    by_id = {c.chunk_id: c for c in citations}

    assert by_id["c2"].content == primary_content
    assert by_id["c2"].excerpt != primary_content  # excerpt 仍是 20–40 字展示用摘录
    assert 20 <= len(by_id["c2"].excerpt) <= 40
    assert by_id["c1"].content == neighbor_content  # 邻接块 fetch 回的全文同样带上
    assert "content" not in by_id["c2"].model_dump(mode="json")


def test_neighbor_expansion_disabled() -> None:
    hits = [make_hit("c2", chapter_no=5, para=2)]
    chunks = [
        make_chunk("c1", chapter_no=5, para=1, next_id="c2"),
        make_chunk("c2", chapter_no=5, para=2, prev_id="c1", next_id="c3"),
        make_chunk("c3", chapter_no=5, para=3, prev_id="c2"),
    ]
    retriever = build_retriever(FakeStore(hits, chunks), neighbor_expand=0)

    citations = retriever.retrieve("工具使用模式", RouteKind.DESIGN)

    assert [c.chunk_id for c in citations] == ["c2"]


def test_neighbor_expansion_does_not_duplicate_primary() -> None:
    hits = [
        make_hit("c2", chapter_no=5, para=2, score=0.9),
        make_hit("c3", chapter_no=5, para=3, score=0.8),
    ]
    chunks = [
        make_chunk("c2", chapter_no=5, para=2, prev_id="c1", next_id="c3"),
        make_chunk("c3", chapter_no=5, para=3, prev_id="c2", next_id="c4"),
        make_chunk("c1", chapter_no=5, para=1, next_id="c2"),
        make_chunk("c4", chapter_no=5, para=4, prev_id="c3"),
    ]
    retriever = build_retriever(FakeStore(hits, chunks), neighbor_expand=1)

    citations = retriever.retrieve("工具使用模式", RouteKind.DESIGN)

    chunk_ids = [c.chunk_id for c in citations]
    assert len(chunk_ids) == len(set(chunk_ids)) == 4
    assert set(chunk_ids) == {"c1", "c2", "c3", "c4"}
    assert [c.index for c in citations] == [1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# 去重 / 过滤 / 降级
# --------------------------------------------------------------------------- #
def test_duplicate_hits_are_deduped() -> None:
    hits = [
        make_hit("c1", score=0.9),
        make_hit("c1", score=0.7),
    ]
    chunks = [make_chunk("c1")]
    retriever = build_retriever(FakeStore(hits, chunks), neighbor_expand=0)

    citations = retriever.retrieve("工具使用模式", RouteKind.DESIGN)

    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"
    assert citations[0].score == pytest.approx(0.9)


def test_references_chunk_type_filtered() -> None:
    hits = [
        make_hit("ref", chunk_type=ChunkType.REFERENCES, score=0.95),
        make_hit("body", score=0.9),
    ]
    chunks = [make_chunk("ref", chunk_type=ChunkType.REFERENCES), make_chunk("body")]
    retriever = build_retriever(FakeStore(hits, chunks), neighbor_expand=0)

    citations = retriever.retrieve("工具使用模式", RouteKind.DESIGN)

    assert [c.chunk_id for c in citations] == ["body"]


def test_empty_search_returns_empty() -> None:
    retriever = build_retriever(FakeStore([], []))

    assert retriever.retrieve("不存在的主题", RouteKind.DESIGN) == []


def test_empty_question_returns_empty() -> None:
    store = FakeStore([], [])
    retriever = build_retriever(store)

    assert retriever.retrieve("   ", RouteKind.DESIGN) == []
    assert store.search_calls == []


def test_search_failure_degrades_to_empty() -> None:
    store = FakeStore([], [], error=RuntimeError("milvus down"))
    retriever = build_retriever(store)

    assert retriever.retrieve("工具使用模式", RouteKind.DESIGN) == []


def test_embedding_failure_degrades_to_empty() -> None:
    embedder = FakeEmbedder()
    embedder.error = RuntimeError("embedding down")
    store = FakeStore([], [])
    retriever = Retriever(embedder=embedder, store=store)

    assert retriever.retrieve("工具使用模式", RouteKind.DESIGN) == []


def test_missing_chunk_falls_back_to_hit_metadata() -> None:
    hits = [make_hit("c1", page=67, score=0.8)]
    store = FakeStore(hits, [])  # fetch 拿不到完整块
    retriever = build_retriever(store, neighbor_expand=1)

    citations = retriever.retrieve("工具使用模式", RouteKind.DESIGN)

    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"
    assert citations[0].page == 67
    assert citations[0].page_pdf == 67  # 无完整块时回退使用 hit.page
    assert citations[0].content == LONG_CONTENT  # content 回退用 hit.content


def test_both_route_mixes_books() -> None:
    hits = [
        make_hit("d1", book="design", chapter_no=5, score=0.9),
        make_hit("p1", book="deep", chapter_no=3, chapter_title="上下文工程", score=0.7),
    ]
    chunks = [
        make_chunk("d1", book="design", chapter_no=5),
        make_chunk("p1", book="deep", chapter_no=3, chapter_title="上下文工程"),
    ]
    store = FakeStore(hits, chunks)
    retriever = build_retriever(store, neighbor_expand=0)

    citations = retriever.retrieve("记忆管理", RouteKind.BOTH)

    assert [c.book for c in citations] == ["design", "deep"]
    assert citations[1].book_title == DEEP_TITLE
    assert [c.index for c in citations] == [1, 2]


# --------------------------------------------------------------------------- #
# 单例
# --------------------------------------------------------------------------- #
def test_get_retriever_returns_singleton() -> None:
    reset_retriever()
    try:
        first = get_retriever()
        second = get_retriever()
        assert first is second
    finally:
        reset_retriever()
