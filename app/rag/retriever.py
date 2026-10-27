# -*- coding: utf-8 -*-
"""检索装配：召回 → 章节多样性 → 相邻块扩展 → 去重 → 排序 → 引用。

真实依赖（延迟 import，允许测试注入 fake）：

- ``app.ingest.embed.get_embedder()``：``encode(texts)`` / ``encode_query(text)`` / ``dim``
- ``app.ingest.milvus_store.get_store()``：``search(query_text, query_vector, books, top_k)``
  / ``fetch(ids)`` / ``count()``

装配规则（docs/DESIGN.md 第 6 节）：

1. 路由决定检索范围：design → 设计模式书；deep → 深入理解书；both → 两本；
2. 同一章最多保留 ``settings.max_per_chapter`` 条（避免单章刷屏）；
3. 相邻块扩展 ``settings.neighbor_expand`` 步（补全上下文，分数按 0.9 衰减）；
4. 按 ``chunk_id`` 去重，按分数降序排序；
5. 输出 :class:`app.models.Citation`，``index`` 从 1 连续、``rank`` 与之一致，
   ``excerpt`` 取原文 20–40 字。
"""
from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.logging_conf import get_logger
from app.models import (
    BookKey,
    Chunk,
    ChunkType,
    Citation,
    RouteKind,
    SearchHit,
    book_title,
)
from app.rag.citations import clean_text, make_excerpt

logger = get_logger("rag.retriever")

__all__ = ["Retriever", "get_retriever", "reset_retriever"]

# 相邻块分数衰减：保证父块排在扩展块之前
_NEIGHBOR_SCORE_FACTOR = 0.9

_ROUTE_BOOKS: dict[RouteKind, tuple[str, ...]] = {
    RouteKind.DESIGN: (BookKey.DESIGN.value,),
    RouteKind.DEEP: (BookKey.DEEP.value,),
    RouteKind.BOTH: (BookKey.DESIGN.value, BookKey.DEEP.value),
}


# --------------------------------------------------------------------------- #
# 内部候选块（SearchHit / Chunk 的统一投影）
# --------------------------------------------------------------------------- #
@dataclass
class _Candidate:
    chunk_id: str
    book: str
    book_title: str
    chapter_no: int = 0
    chapter_title: str = ""
    section_no: str = ""
    section_title: str = ""
    page: int = 0          # 印刷页码（引用展示）
    page_pdf: int = 0      # PDF 页码（排查）
    para_start: int = 0
    para_end: int = 0
    chunk_type: ChunkType = ChunkType.TEXT
    content: str = ""
    score: float = 0.0


def _coerce_route(route: RouteKind | str) -> RouteKind:
    if isinstance(route, RouteKind):
        return route
    try:
        return RouteKind(str(route).strip().lower())
    except ValueError:
        logger.warning("未知 route=%r，按 both 检索", route)
        return RouteKind.BOTH


def _chunk_type_of(value: Any) -> ChunkType:
    if isinstance(value, ChunkType):
        return value
    try:
        return ChunkType(str(value))
    except ValueError:
        return ChunkType.TEXT


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """同时支持 pydantic 对象与 dict（兼容 store 实现差异）。"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_chunk(value: Any) -> Chunk | None:
    if isinstance(value, Chunk):
        return value
    if isinstance(value, dict):
        try:
            return Chunk.model_validate(value)
        except Exception:  # noqa: BLE001 - 上游结构异常时降级
            return None
    return None


def _candidate_from_hit(hit: SearchHit) -> _Candidate:
    content = _get(hit, "content") or _get(hit, "excerpt") or ""
    book = str(_get(hit, "book", "") or "")
    return _Candidate(
        chunk_id=str(_get(hit, "chunk_id", "") or ""),
        book=book,
        book_title=_get(hit, "book_title") or book_title(book),
        chapter_no=int(_get(hit, "chapter_no", 0) or 0),
        chapter_title=_get(hit, "chapter_title") or "",
        section_no=_get(hit, "section_no") or "",
        section_title=_get(hit, "section_title") or "",
        page=int(_get(hit, "page", 0) or 0),
        # SearchHit 不带 PDF 页码；真实 store 的 fetch 结果会覆盖该值
        page_pdf=int(_get(hit, "page", 0) or 0),
        para_start=int(_get(hit, "para_start", 0) or 0),
        para_end=int(_get(hit, "para_end", 0) or 0),
        chunk_type=_chunk_type_of(_get(hit, "chunk_type")),
        content=content,
        score=float(_get(hit, "score", 0.0) or 0.0),
    )


def _candidate_from_chunk(chunk: Chunk, score: float) -> _Candidate:
    return _Candidate(
        chunk_id=str(chunk.chunk_id),
        book=str(chunk.book),
        book_title=chunk.book_title or book_title(str(chunk.book)),
        chapter_no=int(chunk.chapter_no or 0),
        chapter_title=chunk.chapter_title or "",
        section_no=chunk.section_no or "",
        section_title=chunk.section_title or "",
        page=int(chunk.page_print_start or 0),
        page_pdf=int(chunk.page_start or 0),
        para_start=int(chunk.para_start or 0),
        para_end=int(chunk.para_end or 0),
        chunk_type=_chunk_type_of(chunk.chunk_type),
        content=chunk.content or "",
        score=float(score or 0.0),
    )


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """双库混合检索 + 装配。

    Args:
        embedder: 可注入的 embedder（默认延迟 import ``app.ingest.embed``）。
        store: 可注入的向量库（默认延迟 import ``app.ingest.milvus_store``）。
        max_per_chapter / neighbor_expand / top_k: 覆盖 settings 的同名参数（测试用）。
    """

    def __init__(
        self,
        embedder: Any | None = None,
        store: Any | None = None,
        *,
        max_per_chapter: int | None = None,
        neighbor_expand: int | None = None,
        top_k: int | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self.max_per_chapter = int(
            settings.max_per_chapter if max_per_chapter is None else max_per_chapter
        )
        self.neighbor_expand = int(
            settings.neighbor_expand if neighbor_expand is None else neighbor_expand
        )
        self.top_k = int(settings.top_k if top_k is None else top_k)

    # ------------------------------------------------------------------ #
    # 依赖（延迟加载 / 注入）
    # ------------------------------------------------------------------ #
    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from app.ingest.embed import get_embedder  # 延迟 import，避免测试依赖

            self._embedder = get_embedder()
        return self._embedder

    @property
    def store(self) -> Any:
        if self._store is None:
            from app.ingest.milvus_store import get_store  # 延迟 import

            self._store = get_store()
        return self._store

    # ------------------------------------------------------------------ #
    # 公共 API
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        question: str,
        route: RouteKind | str,
        top_k: int | None = None,
    ) -> list[Citation]:
        """按路由检索并装配为引用列表；任何检索故障都降级为空列表（不抛异常）。"""
        question = (question or "").strip()
        if not question:
            return []
        route_kind = _coerce_route(route)
        books = list(_ROUTE_BOOKS[route_kind])
        limit = self._resolve_top_k(top_k)

        if getattr(self.embedder, "mock", False):
            raise RuntimeError("真实 Embedding 不可用，不能用调试向量查询正式知识库")

        try:
            query_vector = self._encode_query(question)
        except Exception as exc:  # noqa: BLE001 - 检索失败要降级为“资料未覆盖”
            logger.warning("query embedding 失败，返回空检索结果: %s", exc)
            return []

        try:
            hits = list(
                self.store.search(
                    query_text=question,
                    query_vector=query_vector,
                    books=books,
                    top_k=limit,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("向量检索失败，返回空检索结果: %s", exc)
            return []

        if not hits:
            logger.info("检索无结果 route=%s books=%s", route_kind.value, books)
            return []

        try:
            candidates = self._assemble(hits, limit)
        except Exception as exc:  # noqa: BLE001 - 上游字段异常时降级为“未覆盖”
            logger.warning("检索结果装配失败，返回空检索结果: %s", exc)
            return []
        citations = self._build_citations(candidates)
        logger.info(
            "检索完成 route=%s books=%s hits=%d citations=%d",
            route_kind.value,
            books,
            len(hits),
            len(citations),
        )
        return citations

    # ------------------------------------------------------------------ #
    # 装配流程
    # ------------------------------------------------------------------ #
    def _assemble(self, hits: Sequence[SearchHit], limit: int) -> list[_Candidate]:
        # 1) 过滤参考文献块（设计文档 C9：默认降权/过滤）
        filtered = [
            h
            for h in hits
            if _chunk_type_of(_get(h, "chunk_type")) is not ChunkType.REFERENCES
        ]
        # 2) 分数降序 + chunk_id 去重（同一块可能被 dense/sparse 多路召回）
        filtered.sort(
            key=lambda h: (
                -float(_get(h, "score", 0.0) or 0.0),
                str(_get(h, "chunk_id", "")),
            )
        )
        primaries: list[SearchHit] = []
        seen_ids: set[str] = set()
        for hit in filtered:
            chunk_id = str(_get(hit, "chunk_id", "") or "")
            if not chunk_id or chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            primaries.append(hit)
        if not primaries:
            return []

        # 3) 同章多样性上限
        capped = self._cap_per_chapter(primaries)

        # 4) 拿完整 Chunk 元数据（印刷页码 / PDF 页码 / prev/next）
        primary_ids = [str(_get(h, "chunk_id", "")) for h in capped]
        primary_chunks = self._fetch(primary_ids)
        selected: dict[str, _Candidate] = {}
        for hit in capped:
            chunk_id = str(_get(hit, "chunk_id", ""))
            chunk = primary_chunks.get(chunk_id)
            if chunk is not None:
                selected[chunk_id] = _candidate_from_chunk(
                    chunk, float(_get(hit, "score", 0.0) or 0.0)
                )
            else:
                selected[chunk_id] = _candidate_from_hit(hit)

        # 5) 相邻块扩展（可多步），分数衰减
        if self.neighbor_expand > 0:
            self._expand_neighbors(selected, primary_chunks)

        # 6) 过滤空内容 + 排序 + 截断
        candidates = [c for c in selected.values() if clean_text(c.content)]
        candidates.sort(
            key=lambda c: (-c.score, c.book, c.chapter_no, c.para_start, c.chunk_id)
        )
        final_limit = limit + max(0, self.neighbor_expand) * 2
        return candidates[:final_limit]

    def _cap_per_chapter(self, hits: Sequence[SearchHit]) -> list[SearchHit]:
        cap = self.max_per_chapter
        if cap <= 0:
            return list(hits)
        counts: Counter[tuple[str, int]] = Counter()
        result: list[SearchHit] = []
        for hit in hits:
            key = (
                str(_get(hit, "book", "")),
                int(_get(hit, "chapter_no", 0) or 0),
            )
            if counts[key] >= cap:
                continue
            counts[key] += 1
            result.append(hit)
        return result

    def _expand_neighbors(
        self, selected: dict[str, _Candidate], primary_chunks: dict[str, Chunk]
    ) -> None:
        visited = set(selected)
        frontier: list[tuple[str, float, Chunk | None]] = [
            (chunk_id, candidate.score, primary_chunks.get(chunk_id))
            for chunk_id, candidate in selected.items()
        ]
        frontier = [item for item in frontier if item[2] is not None]
        for _depth in range(self.neighbor_expand):
            pending: list[tuple[str, float]] = []
            for _parent_id, parent_score, parent_chunk in frontier:
                assert parent_chunk is not None
                for attr in ("prev_id", "next_id"):
                    neighbor_id = getattr(parent_chunk, attr, None)
                    if not neighbor_id or str(neighbor_id) in visited:
                        continue
                    visited.add(str(neighbor_id))
                    pending.append(
                        (str(neighbor_id), parent_score * _NEIGHBOR_SCORE_FACTOR)
                    )
            if not pending:
                return
            fetched = self._fetch([neighbor_id for neighbor_id, _ in pending])
            next_frontier: list[tuple[str, float, Chunk | None]] = []
            for neighbor_id, neighbor_score in pending:
                chunk = fetched.get(neighbor_id)
                if chunk is None:
                    continue
                # 参考文献块只做检索噪声过滤，也不允许通过邻接扩展回到上下文
                if chunk.chunk_type is ChunkType.REFERENCES:
                    continue
                if neighbor_id not in selected:
                    selected[neighbor_id] = _candidate_from_chunk(chunk, neighbor_score)
                    next_frontier.append((neighbor_id, neighbor_score, chunk))
            if not next_frontier:
                return
            frontier = next_frontier

    # ------------------------------------------------------------------ #
    # 依赖调用
    # ------------------------------------------------------------------ #
    def _resolve_top_k(self, top_k: int | None) -> int:
        value = self.top_k if top_k is None else top_k
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = settings.top_k
        return max(1, min(value, 30))

    def _encode_query(self, question: str) -> Sequence[float]:
        embedder = self.embedder
        encode_query = getattr(embedder, "encode_query", None)
        if callable(encode_query):
            vector = encode_query(question)
        else:  # 兼容只有 encode() 的 embedder
            vectors = embedder.encode([question])
            vector = vectors[0] if vectors else []
        if not vector:
            raise RuntimeError("embedder 返回空向量")
        return vector

    def _fetch(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        ids = [str(chunk_id) for chunk_id in chunk_ids if chunk_id]
        if not ids:
            return {}
        try:
            data = self.store.fetch(ids)
        except Exception as exc:  # noqa: BLE001 - 扩展失败不影响主召回
            logger.warning("fetch 失败（忽略相邻块扩展）: %s", exc)
            return {}
        if not isinstance(data, dict):
            # 兼容 store 返回 list[Chunk] 的实现
            if isinstance(data, list):
                chunks: dict[str, Chunk] = {}
                for raw in data:
                    chunk = _as_chunk(raw)
                    if chunk is not None:
                        chunks[str(chunk.chunk_id)] = chunk
                return chunks
            return {}
        chunks: dict[str, Chunk] = {}
        for chunk_id, raw in data.items():
            chunk = _as_chunk(raw)
            if chunk is not None:
                chunks[str(chunk_id)] = chunk
        return chunks

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #
    def _build_citations(self, candidates: Sequence[_Candidate]) -> list[Citation]:
        citations: list[Citation] = []
        for rank, candidate in enumerate(candidates, 1):
            citations.append(
                Citation(
                    index=rank,
                    chunk_id=candidate.chunk_id,
                    book=candidate.book,
                    book_title=candidate.book_title or book_title(candidate.book),
                    chapter_no=candidate.chapter_no,
                    chapter_title=candidate.chapter_title,
                    section_no=candidate.section_no,
                    section_title=candidate.section_title,
                    page=candidate.page,
                    page_pdf=candidate.page_pdf,
                    para_start=candidate.para_start,
                    para_end=candidate.para_end,
                    chunk_type=candidate.chunk_type,
                    excerpt=make_excerpt(candidate.content),
                    # 全文仅供 Agent 组 prompt（Citation.content exclude=True，不进 API JSON）
                    content=candidate.content,
                    score=round(float(candidate.score), 6),
                    rank=rank,
                )
            )
        return citations


_retriever: Retriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> Retriever:
    """进程内单例（懒加载真实 embedder / store）。"""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = Retriever()
    return _retriever


def reset_retriever() -> None:
    """丢弃单例（测试 / 配置变化后使用）。"""
    global _retriever
    with _retriever_lock:
        _retriever = None
