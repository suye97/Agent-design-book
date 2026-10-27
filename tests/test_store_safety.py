"""Milvus 重建与降级路径的离线测试，不创建网络连接。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.ingest.milvus_store import MilvusStore
from app.models import Chunk, ChunkType


def make_chunk(index: int = 1, **overrides) -> Chunk:
    data = dict(
        chunk_id=f"chunk{index:02d}" + "0" * 26,
        book="design", book_title="智能体设计模式", chapter_no=5,
        chapter_title="工具使用", page_start=64, page_end=64,
        page_print_start=51, page_print_end=51, para_start=3, para_end=4,
        chunk_type=ChunkType.TEXT, content="工具调用", content_hash="a" * 32,
        token_len=4,
    )
    data.update(overrides)
    return Chunk(**data)


def staged_store() -> tuple[MilvusStore, MagicMock]:
    store = MilvusStore(dim=2)
    client = MagicMock()
    client.has_collection.return_value = True
    client.query.return_value = [{"count(*)": 1}]
    store._client = client
    store._create_physical = MagicMock()
    store.validate_collection = MagicMock()
    return store, client


def test_rebuild_switches_alias_without_dropping_old_version() -> None:
    store, client = staged_store()
    client.describe_alias.return_value = {"collection_name": "agent_design_kb_v_old"}
    inserted = store.rebuild([make_chunk()], [[0.1, 0.2]], books=["design", "deep"])

    assert inserted == 1
    client.alter_alias.assert_called_once()
    client.drop_collection.assert_not_called()


def test_single_book_rebuild_copies_other_book_rows() -> None:
    store, client = staged_store()
    client.describe_alias.return_value = {"collection_name": "agent_design_kb_v_old"}
    untouched = make_chunk(2, book="deep")
    other_row = store._chunk_row(untouched, [0.3, 0.4])
    iterator = MagicMock()
    iterator.next.side_effect = [[other_row], []]
    client.query_iterator.return_value = iterator
    client.query.return_value = [{"count(*)": 2}]

    store.rebuild([make_chunk()], [[0.1, 0.2]], books=["design"])

    rows = client.insert.call_args.kwargs["data"]
    assert {row["book"] for row in rows} == {"design", "deep"}
    client.drop_collection.assert_not_called()


def test_legacy_rebuild_preserves_backup_and_rolls_back_failed_alias() -> None:
    store, client = staged_store()
    client.describe_alias.side_effect = RuntimeError("not an alias")
    client.create_alias.side_effect = RuntimeError("alias failed")

    with pytest.raises(RuntimeError, match="alias failed"):
        store.rebuild([make_chunk()], [[0.1, 0.2]], books=["design", "deep"])

    assert client.rename_collection.call_count == 2
    rollback = client.rename_collection.call_args_list[-1].kwargs
    assert rollback["new_name"] == "agent_design_kb"
    client.drop_collection.assert_not_called()


def test_rebuild_rejects_empty_or_unknown_book_before_staging() -> None:
    store, client = staged_store()
    client.describe_alias.return_value = {"collection_name": "agent_design_kb_v_old"}

    with pytest.raises(ValueError, match="重建数据为空"):
        store.rebuild([], [], books=["design", "deep"])
    with pytest.raises(ValueError, match="重建书目无效"):
        store.rebuild([make_chunk()], [[0.1, 0.2]], books=["other"])

    store._create_physical.assert_not_called()
    client.alter_alias.assert_not_called()


def test_drop_removes_alias_and_physical_target() -> None:
    store, client = staged_store()
    client.describe_alias.return_value = {"collection_name": "agent_design_kb_v_test"}

    store.drop()

    client.drop_alias.assert_called_once_with(alias="agent_design_kb")
    client.drop_collection.assert_called_once_with(collection_name="agent_design_kb_v_test")


def test_ensure_collection_guards_before_touching_other_project() -> None:
    store = MilvusStore(collection="market_research_samples")
    client = MagicMock()
    store._client = client

    with pytest.raises(ValueError):
        store.ensure_collection(recreate=True)
    client.assert_not_called()


def test_transient_hybrid_failure_does_not_disable_future_hybrid_search() -> None:
    store, client = staged_store()
    client.hybrid_search.side_effect = [RuntimeError("timeout"), []]
    client.search.return_value = []

    store.search("q", [0.1, 0.2])
    store.search("q", [0.1, 0.2])

    assert client.hybrid_search.call_count == 2
    assert store.degraded_reason is None
