"""Embedding 缓存身份与跨线程行为，全部使用本地桩向量。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from app.ingest.embed import Embedder


def test_cache_key_includes_provider(tmp_path) -> None:
    first = Embedder(cache_path=tmp_path / "a.db", force_mock=True, base_url="https://a.example/v1")
    second = Embedder(cache_path=tmp_path / "b.db", force_mock=True, base_url="https://b.example/v1")
    assert first._cache_key("same") != second._cache_key("same")
    first.close()
    second.close()


def test_cache_reused_across_threads(tmp_path) -> None:
    embedder = Embedder(cache_path=tmp_path / "threads.db", force_mock=True, dim=8)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(embedder.encode_query, "跨线程缓存").result()
            second = pool.submit(embedder.encode_query, "跨线程缓存").result()
        assert second == pytest.approx(first)
        assert embedder.stats.hits >= 1
    finally:
        embedder.close()
