# -*- coding: utf-8 -*-
"""extract.py 单测：编号解析 / 大纲结构 / 阅读顺序 / 真实 PDF 冒烟。"""
from __future__ import annotations

import pytest

from app.ingest import extract as ex
from app.ingest.extract import TextLine


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "title,expected",
    [
        ("第 12 章：记忆管理", 12),
        ("第1 章AI Agent 入门", 1),
        ("第 三 章 规划", 3),
        ("第 二十一 章 评估", 21),
        ("引言", 0),
        ("附录 A - 高级提示工程技术", 0),
        ("后记：回到 Agent = LLM + 上下文 + 工具", 0),
    ],
)
def test_parse_chapter_no(title: str, expected: int) -> None:
    assert ex.parse_chapter_no(title) == expected


@pytest.mark.parametrize(
    "title,expected",
    [
        ("第 5 章：工具使用（函数调用）", "工具使用（函数调用）"),
        ("第1 章AI Agent 入门", "AI Agent 入门"),
        ("引言", "引言"),
    ],
)
def test_strip_chapter_prefix(title: str, expected: str) -> None:
    assert ex.strip_chapter_prefix(title) == expected


@pytest.mark.parametrize(
    "title,no,name",
    [
        ("1.1 现代Agent = LLM + 上下文 + 工具", "1.1", "现代Agent = LLM + 上下文 + 工具"),
        ("1.1.1 观察空间与动作空间", "1.1.1", "观察空间与动作空间"),
        ("提示链模式概述", "", "提示链模式概述"),
        ("第 1 章 AI Agent 入门", "", "第 1 章 AI Agent 入门"),
    ],
)
def test_parse_section_no(title: str, no: str, name: str) -> None:
    assert ex.parse_section_no(title) == (no, name)


def test_normalize_title_ignores_space_and_hyphen_variants() -> None:
    a = ex.normalize_title("第 1 章：提示链（Prompt Chaining）")
    b = ex.normalize_title("第1 章：提示链（Prompt Chaining）")
    c = ex.normalize_title("附录A ‑ 高级提示工程技术")
    d = ex.normalize_title("附录 A - 高级提示工程技术")
    assert a == b
    assert c == d


def test_detect_body_start_skips_toc_entry() -> None:
    toc = [
        (1, "目录", 2),
        (1, "简介", 14),
        (2, "前言", 14),
        (1, "第 1 章 提示链", 25),
    ]
    assert ex.detect_body_start(toc, 305) == 14


def test_build_chapters_assigns_sections_and_ranges() -> None:
    toc = [
        (1, "目录", 2),
        (1, "第 1 章 提示链", 25),
        (2, "1.1 概述", 25),
        (2, "1.2 示例", 29),
        (1, "第 2 章 路由", 34),
        (2, "2.1 概述", 34),
        (1, "结论", 238),
    ]
    chapters = ex.build_chapters(toc, body_start_page=14, pdf_pages=305)
    assert [c.chapter_no for c in chapters] == [1, 2, 0]
    assert chapters[0].title == "提示链"
    assert (chapters[0].page_start, chapters[0].page_end) == (25, 33)
    assert chapters[1].title == "路由"
    assert [s.no for s in chapters[0].sections] == ["1.1", "1.2"]
    assert chapters[2].title == "结论"


def test_reading_order_merges_row_fragments() -> None:
    def line(text: str, x0: float, x1: float, y0: float) -> TextLine:
        return TextLine(
            text=text, page=1, x0=x0, y0=y0, x1=x1, y1=y0 + 12, size=12.0
        )

    lines = [
        line("第一行正文", 70, 520, 100),
        line("第二行", 70, 300, 120),
        line("MemoryService 和", 190, 520, 80),
        line("SessionService", 80, 180, 80),
    ]
    ordered = ex._reading_order(lines)
    assert [ln.text for ln in ordered] == [
        "SessionService MemoryService 和",
        "第一行正文",
        "第二行",
    ]
    # 同一视觉行合并后保留最大字号（行内代码字号偏小）
    assert ordered[0].size == 12.0


def test_extract_page_lines_merges_inline_code() -> None:
    """真实 PDF 冒烟：行内代码碎片应合并回整行。"""
    book = ex.extract_book("design", page_numbers=[100])
    page = book.pages[0]
    first = page.lines[0]
    assert first.y0 < 80  # 页眉行
    assert "Session" in page.text


# --------------------------------------------------------------------------- #
# 真实 PDF（整本解析，结构断言）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def design_book() -> ex.BookExtract:
    return ex.extract_book("design")


@pytest.fixture(scope="module")
def deep_book() -> ex.BookExtract:
    return ex.extract_book("deep")


def test_design_book_structure(design_book: ex.BookExtract) -> None:
    assert design_book.pdf_pages == 305
    assert design_book.print_offset == 13
    assert design_book.print_offset_method == "running-head-number"
    assert design_book.body_start_page == 14
    assert design_book.body_font_size == pytest.approx(12.0, abs=0.1)
    assert len(design_book.outline) == 336
    assert len(design_book.chapters) == 33

    first = design_book.chapters[0]
    assert first.title == "简介"
    assert first.page_start == 14

    chapter1 = next(c for c in design_book.chapters if c.chapter_no == 1)
    assert chapter1.title == "提示链（Prompt Chaining）"
    assert chapter1.page_start == 25
    assert chapter1.heading_found
    assert chapter1.sections, "第 1 章应有节"
    assert any(s.title == "实战代码示例" for s in chapter1.sections)

    page14 = next(p for p in design_book.pages if p.pdf_page == 14)
    assert page14.print_page == 1
    # 目录页被识别在正文起点之前
    assert all(p.pdf_page < 14 for p in design_book.pages if p.pdf_page < 14)


def test_deep_book_structure(deep_book: ex.BookExtract) -> None:
    assert deep_book.pdf_pages == 313
    assert deep_book.print_offset == 8
    assert deep_book.print_offset_method == "page-label"
    assert deep_book.body_start_page == 9
    assert len(deep_book.chapters) == 12

    preface = deep_book.chapters[0]
    assert preface.title == "引言"
    assert preface.chapter_no == 0

    chapter1 = next(c for c in deep_book.chapters if c.chapter_no == 1)
    assert chapter1.title == "AI Agent 入门"
    assert chapter1.page_start == 15
    assert chapter1.heading_found
    assert chapter1.sections
    assert chapter1.sections[0].no == "1.1"

    page9 = next(p for p in deep_book.pages if p.pdf_page == 9)
    assert page9.print_page == 1


def test_to_doc_pages_contract(design_book: ex.BookExtract) -> None:
    docs = ex.to_doc_pages(design_book)
    assert len(docs) == design_book.pdf_pages
    assert docs[0].book == "design"
    assert docs[13].pdf_page == 14
    assert docs[13].print_page == 1
