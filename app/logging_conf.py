# -*- coding: utf-8 -*-
"""日志：控制台可读日志 + 按日 JSONL 引用审计日志。

需求 R3：每次回复必须打印"哪本书 / 第几章 / 第几页 / 第几段原文"。
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import settings
from app.models import Citation, RouteDecision

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    settings.ensure_dirs()
    root = logging.getLogger()
    root.setLevel(level.upper())
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
    # 降噪
    for noisy in ("httpx", "httpcore", "urllib3", "pymilvus", "milvus_lite", "openai", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def _audit_path(day: datetime | None = None) -> Path:
    day = day or datetime.now()
    return settings.logs_dir / f"answers-{day:%Y%m%d}.jsonl"


def log_answer(
    *,
    answer_id: str,
    question: str,
    route: RouteDecision,
    citations: list[Citation],
    answer: str,
    latency_ms: int = 0,
    model: str = "",
    used_llm: bool = False,
    citation_violation: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    """打印一轮问答的引用出处，并落盘审计记录。"""
    logger = get_logger("answer")
    now = datetime.now()

    logger.info(
        "[ANSWER] id=%s route=%s intent=%s conf=%.2f src=%s ms=%d model=%s",
        answer_id,
        route.route.value,
        route.intent.value,
        route.confidence,
        route.source.value,
        latency_ms,
        model or "-",
    )
    if route.clarify_hint:
        logger.info("[澄清] %s", route.clarify_hint)

    for cit in citations:
        logger.info(cit.log_line())

    if not citations:
        logger.warning("[引用] 本轮无引用（资料未覆盖或检索为空）")
    if citation_violation:
        logger.warning("[引用校验] 回答中的引用标记或陈述覆盖不足")

    from app.rag.citations import cited_indices

    cited = cited_indices(answer)
    record: dict[str, Any] = {
        "answer_id": answer_id,
        "ts": now.isoformat(timespec="seconds"),
        "question": question,
        "route": route.model_dump(mode="json"),
        "answer": answer,
        "latency_ms": latency_ms,
        "model": model,
        "used_llm": used_llm,
        "citation_violation": citation_violation,
        "cited_ids": [c.chunk_id for c in citations if c.index in cited],
        "sources": [
            {
                "index": c.index,
                "chunk_id": c.chunk_id,
                "book": c.book,
                "book_title": c.book_title,
                "chapter_no": c.chapter_no,
                "chapter_title": c.chapter_title,
                "section_no": c.section_no,
                "section_title": c.section_title,
                "page": c.page,
                "page_pdf": c.page_pdf,
                "para_start": c.para_start,
                "para_end": c.para_end,
                "chunk_type": c.chunk_type.value,
                "excerpt": c.excerpt,
                "score": round(c.score, 6),
                "rank": c.rank,
                "label": c.label(),
            }
            for c in citations
        ],
    }
    if extra:
        record["extra"] = extra

    path = _audit_path(now)
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:  # 日志失败不能影响主流程
        logger.error("写入审计日志失败: %s", exc)
