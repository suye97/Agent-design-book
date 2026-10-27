"""入库提交顺序的离线测试，不读取 PDF、不访问外部服务。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ingest import pipeline
from app.models import Chunk, ChunkType


def make_run(chunks: list[Chunk]) -> pipeline.BookRun:
    return pipeline.BookRun(
        book="design", book_title="智能体设计模式", pdf_path="",
        pdf_pages=1, print_offset=0, print_offset_method="test",
        body_start_page=1, body_font_size=10.0, chapters=1, sections=1,
        chunks=chunks,
    )


def make_chunk() -> Chunk:
    return Chunk(
        chunk_id="chunk010" + "0" * 26,
        book="design", book_title="智能体设计模式", chapter_no=1,
        chapter_title="测试", page_start=1, page_end=1,
        page_print_start=1, page_print_end=1, para_start=1, para_end=1,
        chunk_type=ChunkType.TEXT, content="工具调用",
        content_hash="a" * 32, token_len=4,
    )


class FakeEmbedder:
    mock = False
    dim = 2
    model = "test-model"
    base_url = "https://embed.invalid/v1"

    def encode(self, texts, *, force=False):
        return [[0.1, 0.2] for _ in texts]


def configure_offline_run(monkeypatch, tmp_path, run: pipeline.BookRun) -> None:
    monkeypatch.setattr(type(pipeline.settings), "ensure_dirs", lambda _self: None)
    monkeypatch.setattr(pipeline, "process_book", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(pipeline, "verify_page_attribution", lambda *_args: None)
    monkeypatch.setattr(pipeline, "REPORT_PATH", tmp_path / "INGEST_REPORT.md")
    monkeypatch.setattr(pipeline, "CHUNKS_PATH", tmp_path / "chunks.jsonl")


def test_failed_milvus_update_preserves_local_chunk_snapshot(monkeypatch, tmp_path) -> None:
    configure_offline_run(monkeypatch, tmp_path, make_run([make_chunk()]))
    merge_calls = []
    monkeypatch.setattr(
        pipeline, "_merge_chunks_jsonl",
        lambda *_args: merge_calls.append("unexpected"),
    )

    def fail_write():
        raise ConnectionError("offline")

    store = SimpleNamespace(
        dim=2, model="test-model", provider="https://embed.invalid/v1",
        degraded_reason=None, ensure_collection=fail_write,
    )
    report = pipeline.run_ingest(
        book="design", embedder=FakeEmbedder(), store=store, write_files=True,
    )

    assert report.ok is False
    assert merge_calls == []
    assert "Milvus 写入失败" in report.warnings[-1]


def test_empty_parse_cannot_replace_index_or_snapshot(monkeypatch, tmp_path) -> None:
    configure_offline_run(monkeypatch, tmp_path, make_run([]))
    store = SimpleNamespace(rebuild=lambda *_args, **_kwargs: pytest.fail("不得重建"))

    with pytest.raises(ValueError, match="未产生任何切片"):
        pipeline.run_ingest(
            book="design", embedder=FakeEmbedder(), store=store, write_files=True,
        )

    assert not (tmp_path / "chunks.jsonl").exists()
