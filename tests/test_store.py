# -*- coding: utf-8 -*-
"""milvus_store.py 单测：schema 契约 / 行映射 / 过滤表达式 / 真实 Milvus 往返。

真实 Milvus 用例在连不上时自动 skip；它只操作临时 collection
``rea_smoke_ingest_test``，绝不触碰 agent_design_kb 与其它项目 collection。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from app.ingest import milvus_store as ms
from app.ingest.embed import mock_vector
from app.ingest.milvus_store import MilvusStore
from app.models import Chunk, ChunkType

DIM = 1024


def make_chunk(index: int = 1, **overrides) -> Chunk:
    data = dict(
        chunk_id=f"chunk{index:02d}" + "0" * 26,
        book="design",
        book_title="智能体设计模式：智能系统构建实战指南",
        chapter_no=5,
        chapter_title="工具使用（函数调用）",
        section_no="5.1",
        section_title="概述",
        page_start=64,
        page_end=65,
        page_print_start=51,
        page_print_end=52,
        para_start=3,
        para_end=4,
        chunk_type=ChunkType.TEXT,
        content="工具使用模式让模型在推理过程中调用外部函数。" * 5,
        content_hash="a" * 32,
        token_len=120,
        prev_id=None,
        next_id="chunk020" + "0" * 26,
    )
    data.update(overrides)
    return Chunk(**data)


# --------------------------------------------------------------------------- #
# schema 契约
# --------------------------------------------------------------------------- #
def test_scalar_field_specs_match_design() -> None:
    names = [name for name, _dtype, _extra in ms.SCALAR_FIELD_SPECS]
    assert names == [
        "id",
        "book",
        "book_title",
        "chapter_no",
        "chapter_title",
        "section_no",
        "section_title",
        "page_start",
        "page_end",
        "page_print_start",
        "page_print_end",
        "para_start",
        "para_end",
        "chunk_type",
        "content",
        "content_hash",
        "token_len",
        "prev_id",
        "next_id",
    ]
    spec = dict((name, extra) for name, _dtype, extra in ms.SCALAR_FIELD_SPECS)
    assert spec["id"]["is_primary"] is True
    assert spec["content"]["enable_analyzer"] is True
    assert spec["content"]["max_length"] == 65535


def test_output_fields_complete() -> None:
    # 检索回填需要的字段都应出现在 OUTPUT_FIELDS 中
    expected = set(name for name, _dtype, _extra in ms.SCALAR_FIELD_SPECS)
    assert expected == set(ms.OUTPUT_FIELDS)


# --------------------------------------------------------------------------- #
# 行映射 / 过滤
# --------------------------------------------------------------------------- #
def test_chunk_row_mapping() -> None:
    store = MilvusStore(dim=DIM)
    chunk = make_chunk()
    row = store._chunk_row(chunk, [0.1] * DIM)
    assert row["id"] == chunk.chunk_id
    assert row["chunk_type"] == "text"
    assert row["vector"] == [0.1] * DIM
    assert "sparse" not in row  # 稀疏向量由 BM25 Function 服务端生成
    assert row["next_id"] == chunk.next_id
    # 所有 schema 字段（除 sparse/vector）都在 row 中
    for name, _dtype, _extra in ms.SCALAR_FIELD_SPECS:
        assert name in row


def test_chunk_row_rejects_bad_dim() -> None:
    store = MilvusStore(dim=DIM)
    with pytest.raises(ValueError):
        store._chunk_row(make_chunk(), [0.1] * 8)


def test_row_to_chunk_roundtrip() -> None:
    chunk = make_chunk()
    row = {
        "id": chunk.chunk_id,
        "book": chunk.book,
        "book_title": chunk.book_title,
        "chapter_no": chunk.chapter_no,
        "chapter_title": chunk.chapter_title,
        "section_no": chunk.section_no,
        "section_title": chunk.section_title,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "page_print_start": chunk.page_print_start,
        "page_print_end": chunk.page_print_end,
        "para_start": chunk.para_start,
        "para_end": chunk.para_end,
        "chunk_type": "text",
        "content": chunk.content,
        "content_hash": chunk.content_hash,
        "token_len": chunk.token_len,
        "prev_id": "",
        "next_id": chunk.next_id,
    }
    restored = ms._row_to_chunk(row)
    assert restored.chunk_id == chunk.chunk_id
    assert restored.chunk_type == ChunkType.TEXT
    assert restored.prev_id is None
    assert restored.next_id == chunk.next_id
    assert restored.page_print_start == 51


@pytest.mark.parametrize(
    "books,expected",
    [
        (None, ""),
        ([], ""),
        (["design"], 'book in ["design"]'),
        (["deep"], 'book in ["deep"]'),
        (["design", "deep"], 'book in ["design", "deep"]'),
        (["unknown"], ""),
    ],
)
def test_filter_expr(books, expected) -> None:
    store = MilvusStore(dim=DIM)
    assert store._filter_expr(books) == expected


def test_make_excerpt_truncates() -> None:
    excerpt = ms._make_excerpt("甲" * 500)
    assert len(excerpt) == 161
    assert excerpt.endswith("…")


def test_drop_guard_refuses_other_collections() -> None:
    store = MilvusStore(collection="market_research_samples")
    with pytest.raises(ValueError):
        store.drop()


# --------------------------------------------------------------------------- #
# 真实 Milvus 往返（连不上自动 skip）
# --------------------------------------------------------------------------- #
def _connectable() -> tuple[bool, str]:
    try:
        store = MilvusStore(collection="rea_smoke_ingest_test", timeout=8)
        version = store.server_version()
        store.close()
        if version.startswith("unknown("):
            return False, version
        return True, ""
    except Exception as exc:  # pragma: no cover - 取决于网络
        return False, f"{type(exc).__name__}: {exc}"


_CONNECTED, _CONNECT_ERROR = _connectable()


@pytest.mark.skipif(not _CONNECTED, reason=f"Milvus 不可达：{_CONNECT_ERROR}")
def test_real_milvus_roundtrip() -> None:
    collection = f"rea_smoke_ingest_test_{uuid.uuid4().hex[:10]}"
    store = MilvusStore(collection=collection, dim=DIM)
    try:
        store.ensure_collection()
        assert store.count() == 0

        chunks = [make_chunk(1), make_chunk(2, page_start=66, page_end=66, para_start=5, para_end=5)]
        vectors = [mock_vector(c.content, DIM) for c in chunks]
        inserted = store.upsert_chunks(chunks, vectors)
        assert inserted == 2
        assert store.count() == 2

        # 混合检索（dense + BM25）；若服务端不支持会降级 dense-only
        hits = store.search("工具使用 函数调用", mock_vector("工具使用 函数调用", DIM), top_k=5)
        assert hits, "检索应至少命中一条"
        assert {h.chunk_id for h in hits} & {c.chunk_id for c in chunks}
        assert all(h.excerpt and h.content for h in hits)

        fetched = store.fetch([chunks[0].chunk_id])
        assert chunks[0].chunk_id in fetched
        restored = fetched[chunks[0].chunk_id]
        assert restored.chapter_no == 5
        assert restored.page_print_start == 51

        # 过滤表达式按 book 生效
        assert store.search(
            "工具", mock_vector("工具", DIM), books=["deep"], top_k=5
        ) == []
    finally:
        try:
            store.drop()
        except Exception:  # pragma: no cover - 取决于网络
            pass
        store.close()
