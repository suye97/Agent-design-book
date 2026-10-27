# -*- coding: utf-8 -*-
"""共享数据契约：书目录、切片、引用、路由、API 请求/响应。

所有模块（离线流水线 / 检索 Agent / API / 前端）都只依赖本文件的定义，
避免跨模块隐式约定。
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# 书籍
# --------------------------------------------------------------------------- #
class BookKey(str, Enum):
    DESIGN = "design"
    DEEP = "deep"


BOOK_TITLES: dict[str, str] = {
    BookKey.DESIGN.value: "智能体设计模式：智能系统构建实战指南",
    BookKey.DEEP.value: "深入理解 AI Agent：设计原理与工程实践",
}

BOOK_SHORT_TITLES: dict[str, str] = {
    BookKey.DESIGN.value: "Agent设计模式",
    BookKey.DEEP.value: "深入理解 AI Agent",
}

BOOK_FILES: dict[str, str] = {
    BookKey.DESIGN.value: "Agent设计模式-zh-CN.pdf",
    BookKey.DEEP.value: "深入理解 AI Agent-zh-CN.pdf",
}


def book_title(book: str) -> str:
    return BOOK_TITLES.get(book, book)


# --------------------------------------------------------------------------- #
# 切片
# --------------------------------------------------------------------------- #
class ChunkType(str, Enum):
    TEXT = "text"
    CODE = "code"
    LIST = "list"
    CAPTION = "caption"
    REFERENCES = "references"


class Chunk(BaseModel):
    """一个可检索、可引用的最小单元。"""

    chunk_id: str
    book: str
    book_title: str
    chapter_no: int = 0                      # 0 = 前言/后记等无编号章节
    chapter_title: str = ""
    section_no: str = ""                     # 如 "1.1.1"
    section_title: str = ""
    page_start: int                          # PDF 页码（1 基）
    page_end: int
    page_print_start: int                    # 印刷页码（引用用）
    page_print_end: int
    para_start: int                          # 章内自然段序号（1 基）
    para_end: int
    chunk_type: ChunkType = ChunkType.TEXT
    content: str
    content_hash: str
    token_len: int = 0
    prev_id: str | None = None
    next_id: str | None = None

    def location_label(self) -> str:
        chapter = f"第 {self.chapter_no} 章" if self.chapter_no else "前言/附录"
        head = f"{chapter} {self.chapter_title}".strip()
        if self.section_title and self.section_no:
            head = f"{head} · {self.section_no} {self.section_title}"
        return f"《{self.book_title}》{head} · 第 {self.page_print_start} 页 · 第 {self.para_start} 段"


class DocPage(BaseModel):
    """解析后的单页原文（清洗前/后）。"""

    book: str
    pdf_page: int
    print_page: int
    text: str


# --------------------------------------------------------------------------- #
# 引用 / 路由
# --------------------------------------------------------------------------- #
class Citation(BaseModel):
    index: int                               # 正文里的 [index]
    chunk_id: str
    book: str
    book_title: str
    chapter_no: int
    chapter_title: str
    section_no: str = ""
    section_title: str = ""
    page: int                                # 印刷页码（引用展示）
    page_pdf: int                            # PDF 页码（排查用）
    para_start: int
    para_end: int
    chunk_type: ChunkType = ChunkType.TEXT
    excerpt: str
    score: float = 0.0
    rank: int = 0
    # 切片全文：仅供 Agent 组 prompt 使用，不参与 API 序列化（避免响应体过大）
    content: str = Field(default="", exclude=True)

    def label(self) -> str:
        title = (self.chapter_title or "").strip()
        if self.chapter_no:
            seg = f"第 {self.chapter_no} 章 {title}".strip()
        else:
            seg = title or "前言/附录"
        para = (
            f"第 {self.para_start} 段"
            if self.para_start == self.para_end
            else f"第 {self.para_start}–{self.para_end} 段"
        )
        return f"《{self.book_title}》{seg} · 第 {self.page} 页 · {para}"

    def log_line(self) -> str:
        return f"[引用 {self.index}] {self.label()}\n         原文：“{self.excerpt}”"


class RouteKind(str, Enum):
    DESIGN = "design"
    DEEP = "deep"
    BOTH = "both"


class Intent(str, Enum):
    CONCEPT = "concept"          # 想了解模式/用法 → 设计模式书
    EXPLAIN = "explain"          # 想深入理解原理 → 深入理解书
    AMBIGUOUS = "ambiguous"


class RouterSource(str, Enum):
    OVERRIDE = "override"        # 用户显式指定
    LLM = "llm"
    RULE = "rule"


class RouteDecision(BaseModel):
    route: RouteKind = RouteKind.BOTH
    intent: Intent = Intent.AMBIGUOUS
    confidence: float = 0.0
    source: RouterSource = RouterSource.RULE
    reason: str = ""
    rewritten_query: str = ""
    book_hint: str | None = None
    clarify_hint: str | None = None


# --------------------------------------------------------------------------- #
# API 契约
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    book: BookKey | None = None              # 显式指定书目，覆盖路由
    top_k: int | None = Field(default=None, ge=1, le=30)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: RouteDecision
    latency_ms: int = 0
    model: str = ""
    used_llm: bool = False
    citation_ok: bool = True
    warning: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    book: BookKey | None = None
    route: RouteKind | None = None
    top_k: int = Field(default=8, ge=1, le=30)


class SearchHit(BaseModel):
    chunk_id: str
    book: str
    book_title: str
    chapter_no: int
    chapter_title: str
    section_no: str = ""
    section_title: str = ""
    page: int
    para_start: int
    para_end: int
    chunk_type: ChunkType = ChunkType.TEXT
    excerpt: str
    content: str
    score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    latency_ms: int = 0
    filters: dict[str, Any] = Field(default_factory=dict)


class BookChapter(BaseModel):
    chapter_no: int
    title: str
    page_start: int
    page_end: int
    sections: list[str] = Field(default_factory=list)
    chunk_count: int = 0


class BookInfo(BaseModel):
    book: str
    title: str
    pages: int
    chunks: int
    chapters: list[BookChapter] = Field(default_factory=list)


class StatsResponse(BaseModel):
    collection: str
    row_count: int
    embedding_dim: int
    books: list[BookInfo] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    version: str
    milvus: bool
    collection_ready: bool
    embedding_configured: bool
    embedding_mock: bool
    llm_configured: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    book: BookKey | None = None
    dry_run: bool = False
    force_reembed: bool = False
    recreate_collection: bool = False


class IngestReport(BaseModel):
    ok: bool
    books: dict[str, Any] = Field(default_factory=dict)
    total_chunks: int = 0
    inserted: int = 0
    elapsed_ms: int = 0
    warnings: list[str] = Field(default_factory=list)
