# -*- coding: utf-8 -*-
"""运维接口：/api/stats、/api/books、/api/admin/ingest。"""
from __future__ import annotations

import hmac
import json
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException

from app.config import settings
from app.models import (
    BookChapter,
    BookInfo,
    Chunk,
    IngestReport,
    IngestRequest,
    StatsResponse,
)

router = APIRouter(tags=["admin"])

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _chunks_path() -> Path:
    return settings.data_dir / "chunks.jsonl"


def load_chunks() -> list[Chunk]:
    path = _chunks_path()
    if not path.exists():
        return []
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.model_validate(json.loads(line)))
            except Exception:
                continue
    return chunks


def _book_infos(chunks: list[Chunk]) -> list[BookInfo]:
    by_book: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_book[c.book].append(c)

    infos: list[BookInfo] = []
    for book, items in by_book.items():
        # 无编号章节（前言/附录/后记等）按标题分组，避免全部塌缩成 chapter_no=0
        chapter_map: dict[tuple[int, str], list[Chunk]] = defaultdict(list)
        for c in items:
            chapter_map[(c.chapter_no, c.chapter_title)].append(c)
        chapters: list[BookChapter] = []
        for (no, chapter_title), group in sorted(chapter_map.items()):
            sections: list[str] = []
            for c in group:
                label = f"{c.section_no} {c.section_title}".strip()
                if label and label not in sections:
                    sections.append(label)
            chapters.append(
                BookChapter(
                    chapter_no=no,
                    title=chapter_title,
                    page_start=min(c.page_print_start for c in group),
                    page_end=max(c.page_print_end for c in group),
                    sections=sections[:40],
                    chunk_count=len(group),
                )
            )
        infos.append(
            BookInfo(
                book=book,
                title=items[0].book_title,
                pages=max(c.page_end for c in items),
                chunks=len(items),
                chapters=chapters,
            )
        )
    return infos


@router.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    chunks = load_chunks()
    row_count = 0
    try:
        from app.ingest.milvus_store import get_store

        row_count = get_store().count()
    except Exception:
        row_count = 0
    return StatsResponse(
        collection=settings.milvus_collection,
        row_count=row_count,
        embedding_dim=settings.embedding_dimension,
        books=_book_infos(chunks),
    )


@router.get("/books", response_model=list[BookInfo])
def books() -> list[BookInfo]:
    return _book_infos(load_chunks())


def _require_admin(authorization: str | None = Header(default=None)) -> None:
    token = settings.admin_api_key
    if not token:
        raise HTTPException(status_code=503, detail="后台入库未启用：请配置 ADMIN_API_KEY")
    scheme, _, credential = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(credential, token):
        raise HTTPException(status_code=401, detail="后台入库需要有效的 Bearer token")


@router.post("/admin/ingest", dependencies=[Depends(_require_admin)])
def start_ingest(req: IngestRequest) -> dict:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "started_at": time.time()}

    def worker() -> None:
        try:
            from app.ingest.pipeline import run_ingest

            report: IngestReport = run_ingest(
                book=req.book.value if req.book else None,
                dry_run=req.dry_run,
                force_reembed=req.force_reembed,
                recreate=req.recreate_collection,
            )
            with _jobs_lock:
                _jobs[job_id] = {
                    "status": "completed" if report.ok else "failed",
                    "elapsed_ms": report.elapsed_ms,
                    "report": report.model_dump(mode="json"),
                }
        except Exception as exc:  # noqa: BLE001 - 需要把失败原因回传
            with _jobs_lock:
                _jobs[job_id] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    threading.Thread(target=worker, name=f"ingest-{job_id}", daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@router.get("/admin/ingest/{job_id}", dependencies=[Depends(_require_admin)])
def ingest_status(job_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"未知任务 {job_id}")
    return {"job_id": job_id, **job}
