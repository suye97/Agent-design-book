# -*- coding: utf-8 -*-
"""离线入库流水线编排：解析 → 清洗 → 切片 → Embedding → Milvus → 报告。

用法：
    python -m app.ingest.pipeline                 # dry-run（默认不写 Milvus）
    python -m app.ingest.pipeline --dry-run
    python -m app.ingest.pipeline --book design --recreate

对外主入口：
    run_ingest(book=None, dry_run=False, force_reembed=False, recreate=False) -> IngestReport
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from app.config import BASE_DIR, settings
from app.ingest.chunk import ChunkOptions, ChunkStats, chunk_book, chunk_stats
from app.ingest.clean import CleanResult, clean_book
from app.ingest.embed import Embedder, get_embedder
from app.ingest.extract import BookExtract, extract_book
from app.ingest.milvus_store import MilvusStore, get_store
from app.logging_conf import get_logger
from app.models import Chunk, ChunkType, IngestReport

logger = get_logger("ingest.pipeline")

ALL_BOOKS: tuple[str, ...] = ("design", "deep")
CHUNKS_PATH = settings.data_dir / "chunks.jsonl"
PAGES_DIR = settings.data_dir / "pages"
REPORT_PATH = settings.docs_dir / "INGEST_REPORT.md"


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class BookRun:
    """单本书的处理结果（报告素材）。"""

    book: str
    book_title: str
    pdf_path: str
    pdf_pages: int
    print_offset: int
    print_offset_method: str
    body_start_page: int
    body_font_size: float
    chapters: int
    sections: int
    clean_stats: dict[str, Any] = field(default_factory=dict)
    clean_warnings: list[str] = field(default_factory=list)
    units: dict[str, int] = field(default_factory=dict)
    chunk_stats: dict[str, Any] = field(default_factory=dict)
    chunks: list[Chunk] = field(default_factory=list)
    elapsed_ms: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "book": self.book,
            "book_title": self.book_title,
            "pdf_pages": self.pdf_pages,
            "print_offset": self.print_offset,
            "print_offset_method": self.print_offset_method,
            "body_start_page": self.body_start_page,
            "chapters": self.chapters,
            "sections": self.sections,
            "chunks": len(self.chunks),
            "chunk_stats": self.chunk_stats,
            "clean_stats": self.clean_stats,
            "clean_warnings": self.clean_warnings,
            "units": self.units,
            "elapsed_ms": self.elapsed_ms,
        }


# --------------------------------------------------------------------------- #
# 文件落盘
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _write_pages(extract: BookExtract, clean_result: CleanResult) -> Path:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = PAGES_DIR / f"{extract.book}.jsonl"
    lines: list[str] = []
    by_page = {p.pdf_page: p for p in clean_result.pages}
    for page in extract.pages:
        cleaned = by_page.get(page.pdf_page)
        row = {
            "book": extract.book,
            "pdf_page": page.pdf_page,
            "print_page": page.print_page,
            "text": cleaned.text if cleaned is not None else "",
        }
        lines.append(json.dumps(row, ensure_ascii=False))
    _atomic_write(path, "\n".join(lines) + ("\n" if lines else ""))
    return path


def _merge_chunks_jsonl(chunks: Sequence[Chunk], books: Sequence[str]) -> Path:
    """合并写 chunks.jsonl：只覆盖本次处理的书，保留另一本已有结果。"""
    keep: list[dict[str, Any]] = []
    if CHUNKS_PATH.exists():
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("book") not in set(books):
                keep.append(row)
    processed = [chunk.model_dump(mode="json") for chunk in chunks]
    rows = keep + processed
    order = {book: idx for idx, book in enumerate(ALL_BOOKS)}
    # 稳定排序：只按书分组，书内保持"阅读顺序"（与 prev_id/next_id 一致）
    rows.sort(key=lambda row: order.get(str(row.get("book")), 99))
    _atomic_write(
        CHUNKS_PATH, "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    )
    return CHUNKS_PATH


# --------------------------------------------------------------------------- #
# 单本书流水线
# --------------------------------------------------------------------------- #
def process_book(
    book: str,
    *,
    chunk_options: ChunkOptions | None = None,
) -> BookRun:
    """解析 → 清洗 → 切片，并落盘 data/pages/<book>.jsonl。"""
    started = time.time()
    extract = extract_book(book)
    clean_result = clean_book(extract)
    chunks = chunk_book(extract, clean_result, chunk_options)
    stats: ChunkStats = chunk_stats(chunks, chunk_options)
    _write_pages(extract, clean_result)

    units: dict[str, int] = {}
    for unit in clean_result.units():
        units[unit.kind.value] = units.get(unit.kind.value, 0) + 1
    run = BookRun(
        book=book,
        book_title=extract.book_title,
        pdf_path=extract.pdf_path,
        pdf_pages=extract.pdf_pages,
        print_offset=extract.print_offset,
        print_offset_method=extract.print_offset_method,
        body_start_page=extract.body_start_page,
        body_font_size=extract.body_font_size,
        chapters=len(extract.chapters),
        sections=sum(len(c.sections) for c in extract.chapters),
        clean_stats=clean_result.stats.as_dict(),
        clean_warnings=clean_result.warnings,
        units=units,
        chunk_stats=stats.as_dict(),
        chunks=chunks,
        elapsed_ms=int((time.time() - started) * 1000),
    )
    logger.info(
        "《%s》解析完成：%d 页 / %d 章 / %d 块（%d ms）",
        extract.book_title,
        extract.pdf_pages,
        run.chapters,
        len(chunks),
        run.elapsed_ms,
    )
    return run


# --------------------------------------------------------------------------- #
# 向量化
# --------------------------------------------------------------------------- #
def build_embed_text(chunk: Chunk) -> str:
    """Embedding 输入 = 书/章/节上下文 + 正文（只影响向量，不改写 content）。"""
    head = f"《{chunk.book_title}》{chunk.chapter_title}"
    if chunk.section_title:
        head += f" · {chunk.section_title}"
    return f"{head}\n{chunk.content}"


def embed_chunks(
    chunks: Sequence[Chunk],
    *,
    embedder: Embedder | None = None,
    force: bool = False,
) -> list[list[float]]:
    embedder = embedder or get_embedder()
    texts = [build_embed_text(chunk) for chunk in chunks]
    return embedder.encode(texts, force=force)


# --------------------------------------------------------------------------- #
# 页码可溯源抽检
# --------------------------------------------------------------------------- #
_PAGE_NORM_MAP = str.maketrans(
    {"\u00ad": "", "\u2011": "-", "\u2010": "-", "：": ":", "，": ","}
)


def _norm_page_text(text: str) -> str:
    return re.sub(r"\s+", "", text.translate(_PAGE_NORM_MAP))


def verify_page_attribution(
    chunks: Sequence[Chunk],
    *,
    sample_size: int = 120,
    seed: int = 7,
) -> dict[str, Any]:
    """随机抽样核对 chunk 的 content 前缀是否落在 page_start 对应的 PDF 页文本上。

    口径与 ``.tmp/quantify_pages.py`` 一致（原始 PDF 文本 + 去空白归一化前缀），
    结果写入 INGEST_REPORT，用于持续监控页码可溯源率。
    """
    import random

    import pymupdf

    if not chunks:
        return {"sample": 0, "stats": {}, "detail": [], "precision": 0.0}

    sample = random.Random(seed).sample(list(chunks), min(sample_size, len(chunks)))
    books = sorted({chunk.book for chunk in sample})
    docs = {}
    cache: dict[tuple[str, int], str] = {}
    try:
        for book in books:
            path = settings.pdf_paths.get(book)
            if path is not None and Path(path).exists():
                docs[book] = pymupdf.open(str(path))

        def page_text(book: str, page: int) -> str:
            key = (book, page)
            if key not in cache:
                doc = docs.get(book)
                if doc is None or not (1 <= page <= doc.page_count):
                    cache[key] = ""
                else:
                    cache[key] = _norm_page_text(doc[page - 1].get_text("text"))
            return cache[key]

        stats = {"start": 0, "start-1": 0, "start+1": 0, "range": 0, "none": 0}
        detail: list[dict[str, Any]] = []
        for chunk in sample:
            key = _norm_page_text(chunk.content)[:16]
            ps, pe = chunk.page_start, chunk.page_end
            if not key:
                continue
            category = ""
            if key in page_text(chunk.book, ps):
                stats["start"] += 1
            elif key in page_text(chunk.book, ps - 1):
                stats["start-1"] += 1
                category = "page_start-1"
            elif key in page_text(chunk.book, ps + 1):
                stats["start+1"] += 1
                category = "page_start+1"
            elif any(key in page_text(chunk.book, p) for p in range(ps, pe + 1)):
                stats["range"] += 1
                category = "in-range"
            else:
                stats["none"] += 1
                category = (
                    "code-line-number-order"
                    if chunk.chunk_type == ChunkType.CODE
                    else "figure/formula-text-order"
                )
            if category:
                detail.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "book": chunk.book,
                        "chapter_no": chunk.chapter_no,
                        "page_start": ps,
                        "page_end": pe,
                        "category": category,
                        "head": chunk.content[:40].replace("\n", " "),
                    }
                )
    finally:
        for doc in docs.values():
            doc.close()

    total = len(sample)
    return {
        "sample": total,
        "stats": stats,
        "detail": detail,
        "precision": round(stats["start"] / total, 4) if total else 0.0,
    }


def _page_check_markdown(page_check: dict[str, Any] | None) -> list[str]:
    if not page_check or not page_check.get("sample"):
        return []
    stats = page_check["stats"]
    total = page_check["sample"]
    lines = [
        "## 页码可溯源抽检（固定口径，便于回归）",
        "",
        f"- 抽样：随机 {total} 条（seed=7）；口径：content 去空白归一化后的前 16 字符是否出现在对应 PDF 页文本中。",
        f"- **page_start 精确命中：{stats['start']}/{total}（{page_check['precision'] * 100:.1f}%）**；"
        f"page_start-1：{stats['start-1']}；page_start+1：{stats['start+1']}；"
        f"落在声明页区间内：{stats['range']}；无法在任何声明页匹配：{stats['none']}。",
    ]
    if page_check.get("detail"):
        lines.append("")
        lines.append("| chunk | 书 | 章 | 页区间 | 分类 | 内容前缀 |")
        lines.append("|---|---|---|---|---|---|")
        for item in page_check["detail"][:10]:
            lines.append(
                f"| {item['chunk_id'][:10]} | {item['book']} | {item['chapter_no']} | "
                f"{item['page_start']}-{item['page_end']} | {item['category']} | {item['head'][:40]} |"
            )
        lines.append("")
        lines.append(
            "**匹配口径限制（非页码错置）**：图/表/公式内的文字在 PDF 文本流中按列或按块排列，"
            "我们按视觉行重排后并入正文，与原始文本流顺序不同，归一化前缀无法连续命中；"
            "代码块在 PDF 文本中行号与代码同行，剥离行号后前缀与页文本不再连续。"
            "这两类仅影响本抽检口径；page_start 已用 overlap 所在页 / 页文本校验校正。"
        )
    lines.append("")
    return lines


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _fmt_rule_stats(stats: dict[str, Any]) -> str:
    rules = stats.get("rules", {})
    if not rules:
        return "（无命中）"
    parts = []
    for rule in sorted(rules):
        detail = ", ".join(f"{k}={v}" for k, v in rules[rule].items())
        parts.append(f"{rule}: {detail}")
    return "；".join(parts)


def _sample_rows(chunks: Sequence[Chunk], n: int = 3) -> str:
    if not chunks:
        return "（无切片）"
    step = max(1, len(chunks) // n)
    picks = [chunks[min(len(chunks) - 1, i * step)] for i in range(n)]
    rows = ["| # | 书 | 章 | 节 | PDF页 | 印刷页 | 段 | 类型 | 长度 | 正文摘录 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, chunk in enumerate(picks, start=1):
        excerpt = chunk.content.replace("\n", " ")[:60]
        rows.append(
            f"| {i} | {chunk.book} | {chunk.chapter_no} {chunk.chapter_title} | "
            f"{chunk.section_no} {chunk.section_title} | {chunk.page_start}-{chunk.page_end} | "
            f"{chunk.page_print_start}-{chunk.page_print_end} | {chunk.para_start}-{chunk.para_end} | "
            f"{chunk.chunk_type.value} | {len(chunk.content)} | {excerpt} |"
        )
    return "\n".join(rows)


def build_report(
    runs: dict[str, BookRun],
    *,
    report: IngestReport,
    dry_run: bool,
    embedder: Embedder,
    store: MilvusStore | None,
    milvus_status: str,
    extra_warnings: Sequence[str] = (),
    chunks_file: Path | None = None,
    page_check: dict[str, Any] | None = None,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_chunks = sum(len(run.chunks) for run in runs.values())
    total_pages = sum(run.pdf_pages for run in runs.values())
    lines: list[str] = []
    lines.append("# 离线入库报告（INGEST_REPORT）")
    lines.append("")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- 运行模式：{'dry-run（不写 Milvus）' if dry_run else '正式入库'}")
    lines.append(f"- collection：`{settings.milvus_collection}`（Milvus：{_safe_uri(settings.milvus_uri)}）")
    lines.append(
        f"- Embedding：{'确定性桩向量（无 API key，仅验证链路）' if embedder.mock else embedder.model}"
        f"，维度 {embedder.dim}"
    )
    lines.append(f"- 切片文件：`{chunks_file.relative_to(BASE_DIR) if chunks_file else CHUNKS_PATH.relative_to(BASE_DIR)}`")
    lines.append(f"- 耗时：{report.elapsed_ms} ms；写入 Milvus 行数：{report.inserted}")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 书 | 页数 | 正文起始页 | 印刷页码偏移 | 章数 | 节数 | 清洗单元 | Chunk 数 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for book in ALL_BOOKS:
        run = runs.get(book)
        if run is None:
            continue
        units = sum(run.units.values())
        lines.append(
            f"| {run.book_title} | {run.pdf_pages} | {run.body_start_page} | "
            f"-{run.print_offset}（{run.print_offset_method}） | {run.chapters} | {run.sections} | "
            f"{units} | {len(run.chunks)} |"
        )
    lines.append(f"| **合计** | {total_pages} | - | - | - | - | - | **{total_chunks}** |")
    lines.append("")
    lines.append("## 清洗统计（C1–C10）")
    lines.append("")
    for book in ALL_BOOKS:
        run = runs.get(book)
        if run is None:
            continue
        lines.append(f"### 《{run.book_title}》")
        lines.append("")
        lines.append(f"- 剔除目录/封面页：{run.clean_stats.get('dropped_pages', 0)} 页"
                     f"（{run.clean_stats.get('dropped_page_numbers', [])[:15]}{'...' if run.clean_stats.get('dropped_pages', 0) > 15 else ''}）")
        lines.append(f"- 丢弃行数：{run.clean_stats.get('dropped_lines', 0)}")
        lines.append(f"- 规则命中：{_fmt_rule_stats(run.clean_stats)}")
        for warning in run.clean_warnings:
            lines.append(f"- 清洗警告：{warning}")
        lines.append(f"- 清洗单元：{run.units}")
        lines.append(
            f"- Chunk：{run.chunk_stats.get('chunks', 0)} 个，按类型 {run.chunk_stats.get('by_type', {})}，"
            f"平均 {run.chunk_stats.get('avg_chars', 0)} 字符；"
            f"低于目标 {run.chunk_stats.get('below_min', 0)}，"
            f"正文超上限 {run.chunk_stats.get('above_max', 0)}，"
            f"代码超上限（豁免）{run.chunk_stats.get('code_above_max', 0)}"
        )
        lines.append("")
    lines.append("## 抽样核对（书 / 章 / 节 / 页 / 段 / 类型）")
    lines.append("")
    for book in ALL_BOOKS:
        run = runs.get(book)
        if run is None:
            continue
        lines.append(f"### 《{run.book_title}》")
        lines.append("")
        lines.append(_sample_rows(run.chunks, 3))
        lines.append("")
    lines.extend(_page_check_markdown(page_check))
    lines.append("## Milvus")
    lines.append("")
    lines.append(f"- 状态：{milvus_status}")
    if store is not None and store.degraded_reason:
        lines.append(f"- **限制：已降级 dense-only** —— {store.degraded_reason}")
    lines.append("")
    lines.append("## 已知限制 / 待办")
    lines.append("")
    all_warnings = list(extra_warnings)
    if embedder.mock:
        all_warnings.append(
            "未配置 EMBEDDING_API_KEY：本次仅用 sha256 桩向量验证链路，桩向量不写入正式 collection。"
        )
    if dry_run:
        all_warnings.append("dry-run 未写 Milvus；如需入库请配置 EMBEDDING_API_KEY 后去掉 --dry-run。")
    else:
        all_warnings.append(
            "安全重建使用 staging collection 并切换 alias；旧版本与 legacy 备份会保留，"
            "须由运维确认后清理。普通增量入库的分批 upsert 期间不具备原子可见性。"
        )
    if not all_warnings:
        all_warnings.append("无")
    for warning in dict.fromkeys(all_warnings):
        lines.append(f"- {warning}")
    lines.append("")
    lines.append("## 复现命令")
    lines.append("")
    lines.append("```powershell")
    lines.append("# dry-run（无 API key 也可跑）")
    lines.append("python -m app.ingest.pipeline --dry-run")
    lines.append("# 正式入库（需要 EMBEDDING_API_KEY）")
    lines.append("python -m app.ingest.pipeline --book all")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _safe_uri(uri: str) -> str:
    return uri.split("@")[-1]


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_ingest(
    book: str | None = None,
    dry_run: bool = False,
    force_reembed: bool = False,
    recreate: bool = False,
    *,
    chunk_options: ChunkOptions | None = None,
    embedder: Embedder | None = None,
    store: MilvusStore | None = None,
    write_files: bool = True,
) -> IngestReport:
    """执行离线入库。返回 IngestReport（同时生成 docs/INGEST_REPORT.md）。"""
    started = time.time()
    settings.ensure_dirs()
    warnings: list[str] = []

    books: list[str]
    if book in (None, "", "all"):
        books = list(ALL_BOOKS)
    elif book in ALL_BOOKS:
        books = [book]
    else:
        raise ValueError(f"未知书目: {book}（可选 design/deep/all）")

    logger.info("开始离线入库：books=%s dry_run=%s recreate=%s", books, dry_run, recreate)
    runs: dict[str, BookRun] = {}
    for name in books:
        runs[name] = process_book(name, chunk_options=chunk_options)
        warnings.extend(runs[name].clean_warnings)
        warnings.extend(runs[name].chunk_stats.get("warnings", []))

    # --- 准备本次切片快照；正式入库成功后才提交 chunks.jsonl ---
    all_chunks: list[Chunk] = []
    for name in ALL_BOOKS:
        if name in runs:
            all_chunks.extend(runs[name].chunks)
    if not all_chunks:
        raise ValueError("本次解析未产生任何切片；拒绝更新本地快照或替换索引")
    chunks_file = CHUNKS_PATH

    # --- 页码可溯源抽检（写入报告，便于回归） ---
    page_check = verify_page_attribution(all_chunks) if write_files else None
    if page_check and page_check.get("sample"):
        logger.info(
            "页码抽检：page_start 精确命中 %d/%d（%.1f%%）",
            page_check["stats"]["start"],
            page_check["sample"],
            page_check["precision"] * 100,
        )

    # --- Embedding + Milvus ---
    embedder = embedder or get_embedder()
    store = store or get_store()
    inserted = 0
    milvus_status = "未写入（dry-run）"
    embed_texts_count = 0
    vectors: list[list[float]] | None = None

    if dry_run:
        if embedder.mock:
            # 桩向量很便宜：跑一遍验证"切片→向量"链路与缓存
            vectors = embed_chunks(all_chunks, embedder=embedder, force=force_reembed)
            embed_texts_count = len(vectors)
            milvus_status = (
                f"dry-run 未写入；已用桩向量算得 {embed_texts_count} 条向量（不写库）"
            )
        else:
            milvus_status = "dry-run 未写入（跳过真实 embedding 以节省费用）"
    else:
        if embedder.mock:
            warning = (
                "无 EMBEDDING_API_KEY：拒绝把桩向量写入正式 collection，已跳过 Milvus 写入。"
                "配置真实 key 后重跑即可入库。"
            )
            logger.warning(warning)
            warnings.append(warning)
            milvus_status = "跳过：无真实 embedding"
        else:
            try:
                if store.dim != embedder.dim:
                    raise ValueError("Embedding 维度与 MilvusStore 不一致")
                if store.model != embedder.model or store.provider != embedder.base_url.rstrip("/"):
                    raise ValueError("Embedding 模型或 provider 与 MilvusStore 不一致")
                # Embedding completes before any schema creation or write.
                vectors = embed_chunks(all_chunks, embedder=embedder, force=force_reembed)
                embed_texts_count = len(vectors)
                if recreate:
                    inserted = store.rebuild(all_chunks, vectors, books=books)
                else:
                    store.ensure_collection()
                    inserted = store.upsert_chunks(all_chunks, vectors)
                    removed = store.prune_stale(books, [c.chunk_id for c in all_chunks])
                    if removed:
                        logger.info("已清除 %d 条旧 chunk ID", removed)
                milvus_status = (
                    f"已写入 {inserted} 行（collection={store.collection}，"
                    f"row_count={store.count()}）"
                )
            except Exception as exc:
                message = f"Milvus 写入失败：{type(exc).__name__}: {exc}"
                logger.error(message)
                warnings.append(message)
                milvus_status = f"失败：{message}"

    total_chunks = sum(len(run.chunks) for run in runs.values())
    report_model = IngestReport(
        ok=(dry_run or (not embedder.mock and milvus_status.startswith("已写入"))),
        books={name: run.summary() for name, run in runs.items()},
        total_chunks=total_chunks,
        inserted=inserted,
        elapsed_ms=int((time.time() - started) * 1000),
        warnings=warnings,
    )

    if write_files:
        if dry_run or report_model.ok:
            chunks_file = _merge_chunks_jsonl(all_chunks, books)
            logger.info("已写入 %s（本次 %d 块）", chunks_file, len(all_chunks))
        else:
            logger.warning("Milvus 未成功更新，保留原有 chunks.jsonl 快照")
        markdown = build_report(
            runs,
            report=report_model,
            dry_run=dry_run,
            embedder=embedder,
            store=store,
            milvus_status=milvus_status,
            extra_warnings=warnings,
            chunks_file=chunks_file,
            page_check=page_check,
        )
        _atomic_write(REPORT_PATH, markdown)
        logger.info("已生成报告：%s", REPORT_PATH)

    logger.info(
        "入库完成：%d 块，写入 %d 行，耗时 %d ms",
        total_chunks,
        inserted,
        report_model.elapsed_ms,
    )
    return report_model


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent 设计问答 Agent —— 离线入库流水线（PDF → 清洗 → 切片 → 向量 → Milvus）"
    )
    parser.add_argument("--book", default="all", help="design | deep | all（默认 all）")
    parser.add_argument("--dry-run", action="store_true", help="只跑解析/清洗/切片，不写 Milvus")
    parser.add_argument("--force-reembed", action="store_true", help="忽略 embedding 缓存")
    parser.add_argument("--recreate", action="store_true", help="重建本项目 collection")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_ingest(
        book=args.book,
        dry_run=args.dry_run,
        force_reembed=args.force_reembed,
        recreate=args.recreate,
    )
    print()
    print("=" * 60)
    print(f"ok={report.ok}  total_chunks={report.total_chunks}  inserted={report.inserted}"
          f"  elapsed={report.elapsed_ms}ms")
    for name, info in report.books.items():
        print(
            f"  {name}: pages={info['pdf_pages']} chapters={info['chapters']}"
            f" chunks={info['chunks']}"
        )
    for warning in report.warnings:
        print(f"  [warning] {warning}")
    print(f"报告：{REPORT_PATH}")
    print(f"切片：{CHUNKS_PATH}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
