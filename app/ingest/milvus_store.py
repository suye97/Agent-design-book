# -*- coding: utf-8 -*-
"""Milvus 存取：schema / 索引 / upsert / 混合检索（dense + 原生 BM25）。

collection 固定为 ``settings.milvus_collection``（默认 agent_design_kb），
**绝不读写**其他项目的 market_research_samples / trade_knowledge。

设计（docs/DESIGN.md 第 6 节）：
    - content 字段启用中文 analyzer，BM25 Function 自动生成 sparse 稀疏向量；
    - dense: FLOAT_VECTOR(1024) + HNSW(M=16, efConstruction=200, COSINE)；
    - sparse: SPARSE_INVERTED_INDEX(BM25)；
    - book / chapter_no / chunk_type 建 INVERTED 标量索引；
    - hybrid_search + RRFRanker(k=60)；hybrid 失败时降级为 dense-only 并记录 warning。

对外主入口：
    get_store() -> MilvusStore
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Sequence

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)

from app.config import settings
from app.logging_conf import get_logger
from app.models import Chunk, ChunkType, SearchHit, book_title

logger = get_logger("ingest.store")

#: 内容字段长度上限（Milvus VARCHAR 65535）
CONTENT_MAX_LEN = 65535
#: 写入批大小
INSERT_BATCH = 200
#: 输出字段（检索/回填）
OUTPUT_FIELDS: tuple[str, ...] = (
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
)

#: 字段定义（顺序即 schema 顺序；id 为 VARCHAR 主键，与 chunk_id 一致）
SCALAR_FIELD_SPECS: tuple[tuple[str, str, dict], ...] = (
    ("id", "VARCHAR", {"is_primary": True, "max_length": 64}),
    ("book", "VARCHAR", {"max_length": 32}),
    ("book_title", "VARCHAR", {"max_length": 256}),
    ("chapter_no", "INT64", {}),
    ("chapter_title", "VARCHAR", {"max_length": 512}),
    ("section_no", "VARCHAR", {"max_length": 64}),
    ("section_title", "VARCHAR", {"max_length": 512}),
    ("page_start", "INT64", {}),
    ("page_end", "INT64", {}),
    ("page_print_start", "INT64", {}),
    ("page_print_end", "INT64", {}),
    ("para_start", "INT64", {}),
    ("para_end", "INT64", {}),
    ("chunk_type", "VARCHAR", {"max_length": 32}),
    ("content", "VARCHAR", {"max_length": CONTENT_MAX_LEN, "enable_analyzer": True}),
    ("content_hash", "VARCHAR", {"max_length": 64}),
    ("token_len", "INT64", {}),
    ("prev_id", "VARCHAR", {"max_length": 64}),
    ("next_id", "VARCHAR", {"max_length": 64}),
)

_DTYPE_MAP = {
    "VARCHAR": DataType.VARCHAR,
    "INT64": DataType.INT64,
    "BOOL": DataType.BOOL,
    "FLOAT": DataType.FLOAT,
}


@dataclass
class MilvusStoreStats:
    created: bool = False
    degraded: bool = False
    upserted: int = 0
    insert_batches: int = 0
    warnings: list[str] = field(default_factory=list)


class MilvusStore:
    """Milvus collection 的薄封装（懒连接，便于无网环境导入/单测）。"""

    def __init__(
        self,
        *,
        uri: str | None = None,
        token: str | None = None,
        collection: str | None = None,
        dim: int | None = None,
        model: str | None = None,
        provider: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.uri = uri or settings.milvus_uri
        self._token = token or settings.milvus_token
        self.collection = collection or settings.milvus_collection
        self.dim = int(dim or settings.embedding_dimension)
        self.model = model or settings.embedding_model
        self.provider = (provider or settings.embedding_base_url).rstrip("/")
        self.timeout = float(timeout or settings.milvus_timeout)
        self._client: MilvusClient | None = None
        self._lock = threading.Lock()
        self.stats = MilvusStoreStats()
        #: 稀疏/BM25 不可用时的降级原因（None = 未降级）
        self.degraded_reason: str | None = None
        self.last_warning: str | None = None

    # ------------------------------------------------------------------ #
    # 连接
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = MilvusClient(
                        uri=self.uri, token=self._token, timeout=self.timeout
                    )
                    logger.info("已连接 Milvus：%s（collection=%s）", _safe_uri(self.uri), self.collection)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass
            self._client = None

    def server_version(self) -> str:
        try:
            return str(self.client.get_server_version())
        except Exception as exc:  # pragma: no cover
            return f"unknown({exc})"

    # ------------------------------------------------------------------ #
    # schema / index
    # ------------------------------------------------------------------ #
    def _build_schema(self, *, with_sparse: bool):
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        for name, dtype, extra in SCALAR_FIELD_SPECS:
            kwargs = dict(extra)
            if name == "content":
                kwargs["analyzer_params"] = {"type": "chinese"}
            schema.add_field(name, _DTYPE_MAP[dtype], **kwargs)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        if with_sparse:
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            schema.add_function(
                Function(
                    name="content_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=["content"],
                    output_field_names=["sparse"],
                )
            )
        return schema

    def _build_index_params(self, *, with_sparse: bool):
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        if with_sparse:
            index_params.add_index(
                field_name="sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )
        for field_name in ("book", "chapter_no", "chunk_type"):
            index_params.add_index(field_name=field_name, index_type="INVERTED")
        return index_params

    def _guard_collection(self) -> None:
        """Every path capable of creating, replacing, or deleting must pass this guard."""
        if self.collection != "agent_design_kb" and not self.collection.startswith(
            ("agent_design_kb_", "rea_smoke_")
        ):
            raise ValueError(f"拒绝修改非本项目 collection：{self.collection}")

    def _identity(self) -> dict[str, str]:
        return {"ingest_model": self.model, "ingest_provider": self.provider,
                "ingest_dimension": str(self.dim), "ingest_schema": "1"}

    def validate_collection(self, name: str | None = None, *, require_identity: bool = True) -> None:
        """Reject incompatible schemas before writes; legacy metadata is migration-only."""
        info = self.client.describe_collection(name or self.collection)
        fields = {field["name"]: field for field in info.get("fields", [])}
        expected = {key for key, _, _ in SCALAR_FIELD_SPECS} | {"vector"}
        if not expected <= fields.keys():
            raise ValueError(f"collection schema 缺少字段：{sorted(expected - fields.keys())}")
        for key, dtype, spec in SCALAR_FIELD_SPECS:
            field = fields[key]
            if field.get("type") not in (_DTYPE_MAP[dtype], int(_DTYPE_MAP[dtype])):
                raise ValueError(f"collection 字段类型不符：{key}")
            if key == "id" and not field.get("is_primary"):
                raise ValueError("collection id 不是主键")
            if "max_length" in spec:
                field_params = field.get("params") or field.get("type_params") or {}
                if int(field_params.get("max_length", 0)) != spec["max_length"]:
                    raise ValueError(f"collection 字段长度不符：{key}")
        vector = fields["vector"]
        if vector.get("type") not in (DataType.FLOAT_VECTOR, int(DataType.FLOAT_VECTOR)):
            raise ValueError("collection vector 字段类型不符")
        params = vector.get("params") or vector.get("type_params") or {}
        if int(params.get("dim", 0)) != self.dim:
            raise ValueError(f"collection 向量维度与 embedder 不符：{params}")
        if not require_identity:
            return
        actual = info.get("properties") or {}
        for key, value in self._identity().items():
            if actual.get(key) != value:
                raise ValueError(f"collection {key} 不匹配或缺失；拒绝写入")

    def _alias_target(self) -> str | None:
        try:
            info = self.client.describe_alias(self.collection)
        except Exception:
            return None
        return info.get("collection_name")

    def _new_physical_name(self) -> str:
        return f"{self.collection}_v_{uuid.uuid4().hex[:12]}"

    def _create_physical(self, name: str) -> None:
        """Build on a private name. A failed sparse build may only remove that private name."""
        self._guard_collection()
        try:
            self.client.create_collection(
                collection_name=name, schema=self._build_schema(with_sparse=True),
                index_params=self._build_index_params(with_sparse=True),
                properties=self._identity(),
            )
        except Exception as exc:
            reason = str(exc).lower()
            if not any(word in reason for word in ("bm25", "analyzer", "sparse", "function")):
                raise
            reason = f"BM25/中文 analyzer 建表失败，降级 dense-only：{type(exc).__name__}: {exc}"
            logger.warning(reason)
            self.degraded_reason = reason
            self._warn(reason)
            if self.client.has_collection(name):
                self.client.drop_collection(name)
            self.client.create_collection(
                collection_name=name, schema=self._build_schema(with_sparse=False),
                index_params=self._build_index_params(with_sparse=False),
                properties=self._identity(),
            )
        self.client.load_collection(name)
        self.validate_collection(name)

    def ensure_collection(self, recreate: bool = False) -> bool:
        """Create the alias if absent; existing recreation requires ``rebuild``.

        A legacy physical collection at the public name cannot be atomically swapped
        to an alias. Never drop or overwrite it from this method.
        """
        self._guard_collection()
        if self.client.has_collection(self.collection):
            if recreate:
                raise ValueError("已有 collection：请使用 staging rebuild 完整重建")
            self.validate_collection()
            self._load()
            return False
        name = self._new_physical_name()
        self._create_physical(name)
        self.client.create_alias(collection_name=name, alias=self.collection)
        self.stats.created = True
        return True

    def rebuild(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]],
                *, books: Sequence[str]) -> int:
        """Stage a complete collection, validate count, then atomically change alias.

        For a single book, copy other books with original vectors. The old version is
        retained for recovery. A legacy physical collection is renamed to a backup
        immediately before the public alias is created; failure restores its name.
        """
        self._guard_collection()
        selected = set(books)
        if not selected or not selected <= {"design", "deep"}:
            raise ValueError(f"重建书目无效：{sorted(selected)}")
        target = self._alias_target()
        legacy = bool(self.client.has_collection(self.collection) and not target)
        if target:
            self.validate_collection()
        elif legacy:
            # Existing physical installations can be migrated without deleting their data.
            self.validate_collection(require_identity=False)
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")
        rows = [self._chunk_row(c, v) for c, v in zip(chunks, vectors)]
        if target or legacy:
            self._load()  # Explicit readiness check before replacing the public target.
        if (target or legacy) and len(selected) < 2:
            iterator = self.client.query_iterator(
                collection_name=self.collection,
                filter=f'book not in [{", ".join(chr(34) + b + chr(34) for b in sorted(selected))}]',
                output_fields=list(OUTPUT_FIELDS) + ["vector"], batch_size=INSERT_BATCH,
            )
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    rows.extend(batch)
            finally:
                iterator.close()
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("重建数据包含重复 chunk ID")
        if not rows:
            raise ValueError("重建数据为空；拒绝切换为空索引")
        name = self._new_physical_name()
        self._create_physical(name)
        for start in range(0, len(rows), INSERT_BATCH):
            self.client.insert(collection_name=name, data=rows[start:start + INSERT_BATCH])
        self.client.flush(name)
        result = self.client.query(collection_name=name, filter="", output_fields=["count(*)"])
        if not result or int(result[0]["count(*)"]) != len(rows):
            raise RuntimeError("staging 行数校验失败，旧 alias 保持不变")
        if target:
            self.client.alter_alias(collection_name=name, alias=self.collection)
        elif legacy:
            backup = f"{self.collection}_legacy_{uuid.uuid4().hex[:12]}"
            self.client.rename_collection(old_name=self.collection, new_name=backup)
            try:
                self.client.create_alias(collection_name=name, alias=self.collection)
            except Exception:
                # Restore the original public name if migration cannot complete.
                self.client.rename_collection(old_name=backup, new_name=self.collection)
                raise
        else:
            self.client.create_alias(collection_name=name, alias=self.collection)
        self.stats.created = True
        self.stats.upserted += len(chunks)
        return len(chunks)

    def _load(self) -> None:
        self.client.load_collection(self.collection)

    def has_collection(self) -> bool:
        return bool(self.client.has_collection(self.collection))

    def count(self) -> int:
        """当前存活行数（排除 upsert 产生的软删除历史行）。"""
        try:
            rows = self.client.query(
                collection_name=self.collection, filter="", output_fields=["count(*)"]
            )
            if rows:
                row = rows[0]
                if isinstance(row, dict):
                    return int(row.get("count(*)", 0))
                return int(row)
        except Exception as exc:  # 兼容旧版服务端
            logger.debug("count(*) 查询失败，回退 collection stats：%s", exc)
        stats = self.client.get_collection_stats(self.collection)
        if isinstance(stats, dict):
            value = stats.get("row_count", stats.get("num_entities", 0))
            return int(value or 0)
        return int(stats or 0)

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def _chunk_row(self, chunk: Chunk, vector: Sequence[float]) -> dict:
        if len(vector) != self.dim:
            raise ValueError(f"向量维度 {len(vector)} != collection dim {self.dim}")
        return {
            "id": chunk.chunk_id,
            "book": chunk.book,
            "book_title": chunk.book_title,
            "chapter_no": int(chunk.chapter_no),
            "chapter_title": chunk.chapter_title,
            "section_no": chunk.section_no,
            "section_title": chunk.section_title,
            "page_start": int(chunk.page_start),
            "page_end": int(chunk.page_end),
            "page_print_start": int(chunk.page_print_start),
            "page_print_end": int(chunk.page_print_end),
            "para_start": int(chunk.para_start),
            "para_end": int(chunk.para_end),
            "chunk_type": chunk.chunk_type.value,
            "content": chunk.content[:CONTENT_MAX_LEN],
            "content_hash": chunk.content_hash,
            "token_len": int(chunk.token_len),
            "prev_id": chunk.prev_id or "",
            "next_id": chunk.next_id or "",
            "vector": list(vector),
        }

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """写入/覆盖切片向量；返回写入行数。写完后 flush，保证立即可查。"""
        self._guard_collection()
        self.validate_collection()
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量不一致")
        if not chunks:
            return 0
        rows = [self._chunk_row(chunk, vec) for chunk, vec in zip(chunks, vectors)]
        inserted = 0
        for start in range(0, len(rows), INSERT_BATCH):
            batch = rows[start : start + INSERT_BATCH]
            result = self.client.upsert(collection_name=self.collection, data=batch)
            self.stats.insert_batches += 1
            inserted += len(batch)
            logger.info(
                "upsert %s 批次 %d/%d（%d 行）-> %s",
                self.collection,
                self.stats.insert_batches,
                (len(rows) + INSERT_BATCH - 1) // INSERT_BATCH,
                len(batch),
                result.get("upsert_count", len(batch)) if isinstance(result, dict) else "ok",
            )
        self.client.flush(self.collection)
        self.stats.upserted += inserted
        return inserted

    def prune_stale(self, books: Sequence[str], current_ids: Sequence[str]) -> int:
        """Remove obsolete IDs for the selected books after all new batches succeeded."""
        self._guard_collection()
        self.validate_collection()
        keep = set(current_ids)
        stale: list[str] = []
        for book in books:
            if book not in ("design", "deep"):
                raise ValueError(f"未知书目：{book}")
            iterator = self.client.query_iterator(
                collection_name=self.collection, filter=f'book == "{book}"',
                output_fields=["id"], batch_size=INSERT_BATCH,
            )
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    stale.extend(row["id"] for row in batch if row["id"] not in keep)
            finally:
                iterator.close()
        for start in range(0, len(stale), INSERT_BATCH):
            self.client.delete(collection_name=self.collection, ids=stale[start:start + INSERT_BATCH])
        if stale:
            self.client.flush(self.collection)
        return len(stale)

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def _filter_expr(self, books: Sequence[str] | None) -> str:
        if not books:
            return ""
        safe = [b for b in books if b in ("design", "deep")]
        if not safe:
            return ""
        joined = ", ".join(f'"{b}"' for b in safe)
        return f"book in [{joined}]"

    def search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        books: Sequence[str] | None = None,
        top_k: int = 10,
    ) -> list[SearchHit]:
        """混合检索：dense + BM25 → RRF；失败自动降级 dense-only。"""
        if len(query_vector) != self.dim:
            raise ValueError(f"查询向量维度 {len(query_vector)} != {self.dim}")
        expr = self._filter_expr(books)
        self.last_warning = None
        if not self.degraded_reason:
            try:
                return self._hybrid_search(query_text, query_vector, expr, top_k)
            except Exception as exc:
                reason = f"hybrid_search 失败，降级 dense-only：{type(exc).__name__}: {exc}"
                logger.warning(reason)
                self._warn(reason)
        return self._dense_search(query_vector, expr, top_k)

    def _hybrid_search(
        self, query_text: str, query_vector: Sequence[float], expr: str, top_k: int
    ) -> list[SearchHit]:
        dense_req = AnnSearchRequest(
            data=[list(query_vector)],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=max(top_k, 20),
            expr=expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],  # 稀疏分支直接传原始查询文本，服务端用 chinese analyzer 生成
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=max(top_k, 20),
            expr=expr or None,
        )
        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(settings.rrf_k),
            limit=top_k,
            filter=expr,
            output_fields=list(OUTPUT_FIELDS),
        )
        return self._parse_results(results)

    def _dense_search(
        self, query_vector: Sequence[float], expr: str, top_k: int
    ) -> list[SearchHit]:
        results = self.client.search(
            collection_name=self.collection,
            data=[list(query_vector)],
            anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            filter=expr,
            output_fields=list(OUTPUT_FIELDS),
        )
        return self._parse_results(results)

    def _parse_results(self, results) -> list[SearchHit]:
        hits: list[SearchHit] = []
        if not results:
            return hits
        for rank, hit in enumerate(results[0], start=1):
            entity = _hit_get(hit, "entity") or {}
            content = _field(entity, "content") or ""
            chunk_id = _hit_get(hit, "id") or _field(entity, "id") or ""
            if not chunk_id:
                continue
            hits.append(
                SearchHit(
                    chunk_id=str(chunk_id),
                    book=str(_field(entity, "book") or ""),
                    book_title=str(
                        _field(entity, "book_title")
                        or book_title(str(_field(entity, "book") or ""))
                    ),
                    chapter_no=int(_field(entity, "chapter_no") or 0),
                    chapter_title=str(_field(entity, "chapter_title") or ""),
                    section_no=str(_field(entity, "section_no") or ""),
                    section_title=str(_field(entity, "section_title") or ""),
                    page=int(_field(entity, "page_print_start") or 0),
                    para_start=int(_field(entity, "para_start") or 0),
                    para_end=int(_field(entity, "para_end") or 0),
                    chunk_type=ChunkType(str(_field(entity, "chunk_type") or "text")),
                    excerpt=_make_excerpt(content),
                    content=content,
                    score=float(_hit_get(hit, "distance") or _hit_get(hit, "score") or 0.0),
                )
            )
            del rank
        return hits

    def fetch(self, ids: Sequence[str]) -> dict[str, Chunk]:
        """按 chunk_id 回查完整 Chunk（检索侧邻接扩展用）。"""
        if not ids:
            return {}
        rows = self.client.get(
            collection_name=self.collection,
            ids=list(ids),
            output_fields=list(OUTPUT_FIELDS),
        )
        out: dict[str, Chunk] = {}
        for row in rows or []:
            try:
                chunk = _row_to_chunk(row)
            except Exception as exc:  # pragma: no cover - 数据损坏时跳过
                logger.warning("fetch 解析失败：%s", exc)
                continue
            out[chunk.chunk_id] = chunk
        return out

    def drop(self) -> None:
        """删除本项目 collection（仅用于测试）；alias 与其物理目标一并清理。"""
        self._guard_collection()
        target = self._alias_target()
        self.validate_collection(target or self.collection)
        if target:
            self.client.drop_alias(alias=self.collection)
            self.client.drop_collection(collection_name=target)
        else:
            self.client.drop_collection(collection_name=self.collection)

    # ------------------------------------------------------------------ #
    def _warn(self, message: str) -> None:
        self.last_warning = message
        if message not in self.stats.warnings:
            self.stats.warnings.append(message)


# --------------------------------------------------------------------------- #
# 结果解析辅助
# --------------------------------------------------------------------------- #
def _hit_get(hit, key: str):
    try:
        return hit[key]  # type: ignore[index]
    except Exception:
        return getattr(hit, key, None)


def _field(entity, key: str):
    if entity is None:
        return None
    if isinstance(entity, dict):
        return entity.get(key)
    try:
        return entity[key]
    except Exception:
        return getattr(entity, key, None)


def _make_excerpt(content: str, limit: int = 160) -> str:
    text = (content or "").strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


def _row_to_chunk(row: dict) -> Chunk:
    return Chunk(
        chunk_id=str(row["id"]),
        book=str(row.get("book") or ""),
        book_title=str(row.get("book_title") or ""),
        chapter_no=int(row.get("chapter_no") or 0),
        chapter_title=str(row.get("chapter_title") or ""),
        section_no=str(row.get("section_no") or ""),
        section_title=str(row.get("section_title") or ""),
        page_start=int(row.get("page_start") or 0),
        page_end=int(row.get("page_end") or 0),
        page_print_start=int(row.get("page_print_start") or 0),
        page_print_end=int(row.get("page_print_end") or 0),
        para_start=int(row.get("para_start") or 0),
        para_end=int(row.get("para_end") or 0),
        chunk_type=ChunkType(str(row.get("chunk_type") or "text")),
        content=str(row.get("content") or ""),
        content_hash=str(row.get("content_hash") or ""),
        token_len=int(row.get("token_len") or 0),
        prev_id=(str(row["prev_id"]) if row.get("prev_id") else None),
        next_id=(str(row["next_id"]) if row.get("next_id") else None),
    )


def _safe_uri(uri: str) -> str:
    """日志/报告中不输出 host 与 token，只保留 scheme 与端口。"""
    clean = (uri or "").split("@")[-1]
    scheme, sep, rest = clean.partition("://")
    if not sep:
        return "<milvus>"
    hostport, _slash, _path = rest.partition("/")
    _host, _colon, port = hostport.partition(":")
    suffix = f":{port}" if port else ""
    return f"{scheme}://<milvus-host>{suffix}"


_STORE: MilvusStore | None = None
_STORE_LOCK = threading.Lock()


def get_store(**kwargs) -> MilvusStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = MilvusStore(**kwargs)
    return _STORE


def reset_store() -> None:
    global _STORE
    with _STORE_LOCK:
        if _STORE is not None:
            _STORE.close()
        _STORE = None


__all__ = [
    "CONTENT_MAX_LEN",
    "OUTPUT_FIELDS",
    "SCALAR_FIELD_SPECS",
    "MilvusStore",
    "MilvusStoreStats",
    "get_store",
    "reset_store",
]
