# -*- coding: utf-8 -*-
"""Embedding：DashScope / OpenAI 兼容 /embeddings，批量 + 重试 + sqlite 缓存。

- 有 API key：调用 ``{base_url}/embeddings``（text-embedding-v4，1024 维）；
- 无 key 或 ``EMBEDDING_MOCK=1``：用 sha256 播种的确定性 1024 维桩向量跑通链路
  （仅开发/测试，**检索语义无效**，流水线不会把桩向量写进正式 collection）；
- 缓存：``data/embed_cache.sqlite3``，key = sha256(model|text)，重跑不重复计费。

对外主入口：
    get_embedder() -> Embedder（单例）
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.logging_conf import get_logger

logger = get_logger("ingest.embed")

_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key  TEXT PRIMARY KEY,
    model      TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embedding_cache_hash ON embedding_cache(text_hash);
"""


@dataclass
class EmbedderStats:
    hits: int = 0
    misses: int = 0
    api_calls: int = 0
    api_texts: int = 0
    mock_texts: int = 0
    failed: int = 0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "api_calls": self.api_calls,
            "api_texts": self.api_texts,
            "mock_texts": self.mock_texts,
            "failed": self.failed,
        }


def _pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def mock_vector(text: str, dim: int) -> list[float]:
    """确定性桩向量：sha256 计数器扩展 → 均匀分布 → L2 归一化。"""
    seed = hashlib.sha256(text.encode("utf-8")).digest()
    need = dim * 4
    chunks: list[bytes] = []
    counter = 0
    while sum(len(c) for c in chunks) < need:
        chunks.append(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    raw = b"".join(chunks)[:need]
    values = [
        (int.from_bytes(raw[i : i + 4], "big") / 2147483648.0) - 1.0
        for i in range(0, need, 4)
    ]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class Embedder:
    """批量向量化器（带 sqlite 缓存与确定性 mock）。"""

    def __init__(
        self,
        *,
        cache_path: Path | None = None,
        model: str | None = None,
        dim: int | None = None,
        batch_size: int | None = None,
        force_mock: bool | None = None,
        base_url: str | None = None,
        enable_cache: bool = True,
    ) -> None:
        self.model = model or settings.embedding_model
        self.dim = int(dim or settings.embedding_dimension)
        self.batch_size = max(1, int(batch_size or settings.embedding_batch_size))
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.api_key = settings.embedding_api_key
        self.timeout = settings.embedding_timeout
        self.max_retries = max(1, int(settings.embedding_max_retries))
        self._force_mock = force_mock
        self._mock = bool(force_mock) if force_mock is not None else (
            settings.embedding_mock or not bool(self.api_key)
        )
        self._cache_path = Path(cache_path or (settings.data_dir / "embed_cache.sqlite3"))
        self._enable_cache = enable_cache
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._pending_commits = 0
        self.stats = EmbedderStats()
        self._client = None
        if not self._mock:
            try:
                from openai import OpenAI  # 延迟导入，mock 模式不需要
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("缺少 openai 依赖，无法调用真实 Embedding") from exc
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,  # 重试交给 tenacity，避免双重重试
            )

    # ---------------- 属性 ----------------
    @property
    def mock(self) -> bool:
        return self._mock

    @property
    def cache_path(self) -> Path:
        return self._cache_path

    # ---------------- 缓存 ----------------
    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            # The singleton serves FastAPI worker threads; encode() serializes cache access.
            self._conn = sqlite3.connect(str(self._cache_path), check_same_thread=False)
            self._conn.executescript(_CACHE_SCHEMA)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _cache_key(self, text: str) -> str:
        # 缓存 key 必须区分 mock / 真实 API，避免无 key 阶段写入的桩向量被真实入库复用
        mode = "mock" if self._mock else "api"
        return hashlib.sha256(f"{self.base_url}|{self.model}|{mode}|{text}".encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> list[float] | None:
        if not self._enable_cache:
            return None
        conn = self._connection()
        row = conn.execute(
            "SELECT vector FROM embedding_cache WHERE cache_key = ? AND dim = ?",
            (key, self.dim),
        ).fetchone()
        if not row:
            return None
        return _unpack(row[0])

    def _cache_put(self, key: str, text: str, vector: Sequence[float]) -> None:
        if not self._enable_cache:
            return
        conn = self._connection()
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache"
            " (cache_key, model, text_hash, dim, vector, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                key,
                self.model,
                hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                len(vector),
                _pack(vector),
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            ),
        )
        self._pending_commits += 1
        if self._pending_commits >= 200:
            conn.commit()
            self._pending_commits = 0

    def _commit(self) -> None:
        if self._conn is not None and self._pending_commits:
            self._conn.commit()
            self._pending_commits = 0

    # ---------------- 真实 API ----------------
    def _call_api(self, batch: Sequence[str]) -> list[list[float]]:
        assert self._client is not None

        @retry(
            retry=retry_if_exception_type(Exception),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=20.0),
            stop=stop_after_attempt(self.max_retries),
            reraise=True,
        )
        def _do() -> list[list[float]]:
            kwargs = {
                "model": self.model,
                "input": list(batch),
                "encoding_format": "float",
            }
            try:
                resp = self._client.embeddings.create(**kwargs)  # type: ignore[union-attr]
            except Exception as exc:  # 某些兼容端点不认 dimensions，降级重试一次
                if "dimension" in str(exc).lower() and "dimensions" not in kwargs:
                    kwargs["dimensions"] = self.dim
                    resp = self._client.embeddings.create(**kwargs)  # type: ignore[union-attr]
                else:
                    raise
            data = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
            return [list(d.embedding) for d in data]

        vectors = _do()
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedding 返回数量不一致：{len(vectors)} != {len(batch)}"
            )
        for vec in vectors:
            if len(vec) != self.dim:
                raise RuntimeError(
                    f"Embedding 维度不一致：{len(vec)} != {self.dim}"
                    f"（model={self.model}）"
                )
        return vectors

    # ---------------- 对外接口 ----------------
    def encode(self, texts: Sequence[str], *, force: bool = False) -> list[list[float]]:
        """批量向量化；命中缓存的不再调用 API。缓存操作可安全跨线程调用。"""
        with self._lock:
            return self._encode_locked(texts, force=force)

    def _encode_locked(self, texts: Sequence[str], *, force: bool = False) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        pending: list[int] = []

        if not force:
            for idx, text in enumerate(texts):
                key = self._cache_key(text)
                vec = self._cache_get(key)
                if vec is not None and len(vec) == self.dim:
                    results[idx] = vec
                    self.stats.hits += 1
                else:
                    pending.append(idx)
        else:
            pending = list(range(len(texts)))

        if pending:
            self.stats.misses += len(pending)
            if self._mock:
                for idx in pending:
                    vec = mock_vector(texts[idx], self.dim)
                    results[idx] = vec
                    self._cache_put(self._cache_key(texts[idx]), texts[idx], vec)
                self.stats.mock_texts += len(pending)
            else:
                for start in range(0, len(pending), self.batch_size):
                    idxs = pending[start : start + self.batch_size]
                    batch = [texts[i] for i in idxs]
                    try:
                        vectors = self._call_api(batch)
                    except Exception as exc:
                        self.stats.failed += len(idxs)
                        logger.error("Embedding 调用失败（%d 条）：%s", len(idxs), exc)
                        raise
                    self.stats.api_calls += 1
                    self.stats.api_texts += len(idxs)
                    for idx, vec in zip(idxs, vectors):
                        results[idx] = vec
                        self._cache_put(self._cache_key(texts[idx]), texts[idx], vec)

        missing = [i for i, vec in enumerate(results) if vec is None]
        if missing:
            raise RuntimeError(f"有 {len(missing)} 条文本未得到向量")
        self._commit()
        return [vec for vec in results if vec is not None]

    def encode_query(self, text: str) -> list[float]:
        """单条查询向量（同样走缓存）。"""
        return self.encode([text])[0]


_EMBEDDER: Embedder | None = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(**kwargs) -> Embedder:
    """进程级单例。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = Embedder(**kwargs)
    return _EMBEDDER


def reset_embedder() -> None:
    """测试辅助：重置单例。"""
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is not None:
            _EMBEDDER.close()
        _EMBEDDER = None


__all__ = ["Embedder", "EmbedderStats", "get_embedder", "mock_vector", "reset_embedder"]
