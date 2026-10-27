# -*- coding: utf-8 -*-
"""命令行入口。

用法：
    python -m app.cli check
    python -m app.cli ingest [--book design|deep] [--dry-run] [--force-reembed] [--recreate]
    python -m app.cli ask "什么是提示链模式？"
    python -m app.cli search "提示链" [--book design]
    python -m app.cli stats
    python -m app.cli serve [--host 127.0.0.1] [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime

from app.config import settings
from app.logging_conf import configure_logging, get_logger, log_answer

logger = get_logger("cli")


def _mask(secret: str | None) -> str:
    if not secret:
        return "（未配置）"
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}…{secret[-4:]}（len={len(secret)}）"


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def cmd_check(args: argparse.Namespace) -> int:
    print("=" * 72)
    print("环境自检  Agent 设计问答 Agent")
    print("=" * 72)

    problems: list[str] = []
    warnings: list[str] = []
    require_pdfs = bool(getattr(args, "require_pdfs", False))

    print(f"[路径] 工作目录        : {settings.data_dir.parent}")
    print(f"[路径] 数据目录        : {settings.data_dir}")
    print(f"[路径] 日志目录        : {settings.logs_dir}")
    for book, pdf in settings.pdf_paths.items():
        ok = pdf.exists()
        print(f"[语料] {book:<7}      : {pdf.name}  {'OK' if ok else '缺失（运行时不需要）'}")
        if not ok:
            if require_pdfs:
                problems.append(f"缺少 PDF: {pdf}")
            else:
                warnings.append(f"缺少 PDF: {pdf}（仅影响在服务器上重新入库）")

    print(f"[Milvus] URI           : {'<已配置>' if settings.milvus_uri else '<未配置>'}")
    print(f"[Milvus] collection    : {settings.milvus_collection}")
    milvus_ok = False
    row_count = 0
    try:
        from app.ingest.milvus_store import get_store

        store = get_store()
        if not store.has_collection():
            problems.append("Milvus collection 尚未创建")
            print("[Milvus] collection    : 尚未创建")
        else:
            row_count = store.count()
            milvus_ok = True
    except Exception as exc:
        print(f"[Milvus] 连接失败      : {type(exc).__name__}: {exc}")
        problems.append("Milvus 不可用")
    if milvus_ok:
        print(f"[Milvus] 连接          : OK  行数={row_count}")

    print(f"[Embedding] model      : {settings.embedding_model}  dim={settings.embedding_dimension}")
    print(f"[Embedding] api-key    : {_mask(settings.embedding_api_key)}")
    print(f"[Embedding] mock 模式  : {'是' if settings.embedding_mock else '否'}")
    if not settings.embedding_enabled:
        problems.append("Embedding 未配置（EMBEDDING_API_KEY 为空且未开启 EMBEDDING_MOCK）")

    print(f"[LLM] base-url         : {settings.llm_base_url}")
    print(f"[LLM] model            : {settings.llm_model}")
    print(f"[LLM] api-key          : {_mask(settings.llm_api_key)}")
    if not settings.llm_enabled:
        print("[LLM] 状态             : 未配置 → 路由走规则、答案走检索摘要降级")
        warnings.append("LLM 未配置：可运行，但回答退化为检索摘要")

    chunks_path = settings.data_dir / "chunks.jsonl"
    if chunks_path.exists():
        n = sum(1 for _ in chunks_path.open(encoding="utf-8"))
        print(f"[数据] chunks.jsonl    : {n} 条")
    else:
        print("[数据] chunks.jsonl    : 尚未生成（/api/stats、/api/books 会为空）")
        warnings.append("未找到 data/chunks.jsonl：问答可用，统计/目录接口为空")

    print("-" * 72)
    if problems:
        print("结论：存在必须处理的问题")
        for p in problems:
            print(f"  ✗ {p}")
        if warnings:
            print("另有警告：")
            for w in warnings:
                print(f"  ! {w}")
        return 1
    if warnings:
        print("结论：核心检查通过（有非阻塞警告）")
        for w in warnings:
            print(f"  ! {w}")
        return 0
    print("结论：全部检查通过")
    return 0


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #
def cmd_ingest(args: argparse.Namespace) -> int:
    from app.ingest.pipeline import run_ingest

    started = time.perf_counter()
    report = run_ingest(
        book=args.book,
        dry_run=args.dry_run,
        force_reembed=args.force_reembed,
        recreate=args.recreate,
    )
    elapsed = int((time.perf_counter() - started) * 1000)
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print(f"\n完成，用时 {elapsed / 1000:.1f}s；详细报告见 docs/INGEST_REPORT.md")
    return 0 if report.ok else 1


# --------------------------------------------------------------------------- #
# ask / search / stats
# --------------------------------------------------------------------------- #
def cmd_ask(args: argparse.Namespace) -> int:
    from app.rag.agents import get_agent, verify_citations
    from app.rag.retriever import get_retriever
    from app.rag.router import route_question

    started = time.perf_counter()
    answer_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"

    try:
        route = route_question(args.question, args.book)
        citations = get_retriever().retrieve(route.rewritten_query or args.question, route.route, args.top_k)
        answer, used_llm = get_agent(route.route).answer(args.question, citations)
        ok = verify_citations(answer, citations)
    except Exception as exc:
        print(f"失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("提示：先执行 `python -m app.cli ingest`，并用 `check` 确认配置", file=sys.stderr)
        return 1

    latency = int((time.perf_counter() - started) * 1000)
    log_answer(
        answer_id=answer_id,
        question=args.question,
        route=route,
        citations=citations,
        answer=answer,
        latency_ms=latency,
        model=settings.llm_model,
        used_llm=used_llm,
        citation_violation=not ok,
    )

    if args.json:
        print(json.dumps(
            {
                "answer_id": answer_id,
                "answer": answer,
                "route": route.model_dump(mode="json"),
                "citation_ok": ok,
                "used_llm": used_llm,
                "latency_ms": latency,
                "citations": [c.model_dump(mode="json") for c in citations],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    print()
    print(answer)
    print()
    print("-" * 72)
    for c in citations:
        print(c.label())
        print(f"    {c.excerpt}")
    if not citations:
        print("（无引用：知识库无相关内容或尚未入库）")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from app.ingest.embed import get_embedder
    from app.ingest.milvus_store import get_store
    from app.models import BOOK_SHORT_TITLES

    books = [args.book] if args.book else ["design", "deep"]
    try:
        embedder = get_embedder()
        if embedder.mock:
            raise RuntimeError("真实 Embedding 不可用，不能查询正式知识库")
        vector = embedder.encode_query(args.query)
        hits = get_store().search(args.query, vector, books, top_k=args.top_k)
    except Exception as exc:
        print(f"失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    for i, hit in enumerate(hits, 1):
        name = BOOK_SHORT_TITLES.get(hit.book, hit.book)
        print(f"[{i}] {name} 第{hit.chapter_no}章 {hit.chapter_title} · 第{hit.page}页 · 第{hit.para_start}段  score={hit.score:.4f}")
        print(f"    {hit.excerpt}")
    if not hits:
        print("（无命中）")
    return 0


def cmd_stats(_args: argparse.Namespace) -> int:
    from app.api.admin import load_chunks

    chunks = load_chunks()
    by_book: dict[str, int] = {}
    for c in chunks:
        by_book[c.book] = by_book.get(c.book, 0) + 1
    print(f"collection : {settings.milvus_collection}")
    try:
        from app.ingest.milvus_store import get_store

        print(f"milvus rows: {get_store().count()}")
    except Exception as exc:
        print(f"milvus rows: 不可用 ({type(exc).__name__})")
    print(f"chunks.jsonl: {len(chunks)} 条")
    for book, n in by_book.items():
        print(f"  - {book}: {n}")
    return 0


def cmd_books(_args: argparse.Namespace) -> int:
    from app.api.admin import load_chunks
    from app.models import BOOK_SHORT_TITLES

    chunks = load_chunks()
    if not chunks:
        print("尚未入库（先执行 `python -m app.cli ingest`）")
        return 1

    by_book: dict[str, dict[int, dict]] = {}
    for c in chunks:
        info = by_book.setdefault(c.book, {}).setdefault(
            c.chapter_no,
            {
                "title": c.chapter_title,
                "pages": [c.page_print_start, c.page_print_end],
                "chunks": 0,
            },
        )
        info["pages"][0] = min(info["pages"][0], c.page_print_start)
        info["pages"][1] = max(info["pages"][1], c.page_print_end)
        info["chunks"] += 1

    for book, chapters in by_book.items():
        total = sum(v["chunks"] for v in chapters.values())
        print(f"\n《{BOOK_SHORT_TITLES.get(book, book)}》 共 {len(chapters)} 章 / {total} 块")
        for no in sorted(chapters):
            info = chapters[no]
            label = f"第 {no} 章" if no else "前言/附录"
            print(
                f"  {label:<6} {info['title'][:36]:<38} "
                f"p{info['pages'][0]}–{info['pages'][1]}  {info['chunks']} 块"
            )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=args.host or settings.host,
        port=args.port or settings.port,
        reload=args.reload,
    )
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description="Agent 设计问答 Agent 命令行")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="环境与连通性自检")
    p.add_argument(
        "--require-pdfs",
        action="store_true",
        help="把 PDF 缺失视为错误（在服务器上执行 ingest 时使用）",
    )
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("ingest", help="离线入库：解析 → 清洗 → 切片 → 向量化 → Milvus")
    p.add_argument("--book", choices=["design", "deep"], default=None)
    p.add_argument("--dry-run", action="store_true", help="只生成 chunks 与报告，不写 Milvus")
    p.add_argument("--force-reembed", action="store_true", help="忽略 embedding 缓存")
    p.add_argument("--recreate", action="store_true", help="删除并重建 collection")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="问一个问题")
    p.add_argument("question")
    p.add_argument("--book", choices=["design", "deep"], default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("search", help="只检索，不生成答案")
    p.add_argument("query")
    p.add_argument("--book", choices=["design", "deep"], default=None)
    p.add_argument("--top-k", type=int, default=8)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", help="查看入库统计")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("books", help="查看两本书章节目录")
    p.set_defaults(func=cmd_books)

    p = sub.add_parser("serve", help="启动 HTTP 服务")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
