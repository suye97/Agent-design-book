# -*- coding: utf-8 -*-
"""clean.py 单测：C1–C10 规则、代码行号、段落合并、真实 PDF 清洗冒烟。"""
from __future__ import annotations

import pytest

from app.ingest import clean as cl
from app.ingest import extract as ex
from app.ingest.extract import Chapter, PageText, Section, TextLine


# --------------------------------------------------------------------------- #
# 构造工具
# --------------------------------------------------------------------------- #
def line(
    text: str,
    y0: float,
    x0: float = 71.0,
    x1: float = 524.0,
    size: float = 12.0,
    page: int = 14,
    mono: bool = False,
) -> TextLine:
    fonts = {"SourceHanMonoSC-Regular": max(1, len(text))} if mono else {"SourceHanSansSC-Regular": len(text)}
    return TextLine(
        text=text,
        page=page,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y0 + 19,
        size=size,
        fonts=fonts,
        sizes={size: len(text)},
    )


def page(pdf_page: int, lines: list[TextLine]) -> PageText:
    for item in lines:
        item.page = pdf_page
    return PageText(
        book="design",
        pdf_page=pdf_page,
        print_page=pdf_page - 13,
        lines=lines,
        width=595.0,
        height=842.0,
    )


def make_extract(pages: list[PageText]) -> ex.BookExtract:
    chapter = Chapter(
        chapter_no=1,
        title="提示链",
        page_start=14,
        page_end=15,
        anchor_title="第 1 章 提示链",
        heading_y=69.0,
        heading_found=True,
        sections=[Section(no="1.1", title="概述", page=14, level=2, anchor_title="1.1 概述")],
    )
    return ex.BookExtract(
        book="design",
        book_title="智能体设计模式：智能系统构建实战指南",
        pdf_path="dummy.pdf",
        pdf_pages=305,
        pages=pages,
        outline=[],
        chapters=[chapter],
        print_offset=13,
        print_offset_method="test",
        body_start_page=14,
        body_font_size=12.0,
        body_top=75.0,
    )


def synthetic_pages() -> list[PageText]:
    toc_page = page(1, [line("目录 . . . 1", 100)])
    p14 = page(
        14,
        [
            line("5 第 1 章 提示链", 39),                     # C2 页眉
            line("第1 章 提示链", 69, size=24.8),             # heading
            line("这是第一段的第一行文字，", 100),
            line("这是第一段结尾。", 119, x1=300),
            line("第二段第一行，", 138),
            line("第二段结尾。", 157, x1=200),
            line("图1：这是图注说明。", 400, x1=450),
            line("5", 770, x0=294, x1=301),                   # C3 页脚页码
        ],
    )
    p15 = page(
        15,
        [
            line("1", 100, x0=80, x1=84, size=6.4, mono=True),
            line("import os", 100, x0=90, x1=160, size=6.4, mono=True),
            line("2", 112, x0=80, x1=84, size=6.4, mono=True),
            line("print(1)", 112, x0=90, x1=180, size=6.4, mono=True),
            line("3", 124, x0=80, x1=84, size=6.4, mono=True),
            line("x = 1", 124, x0=90, x1=160, size=6.4, mono=True),
            line("参考文献", 160, size=17.2, page=15),
            line("・参考资料一", 190, x1=400),
        ],
    )
    return [toc_page, p14, p15]


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #
def test_join_wrapped_cjk_ascii_hyphen() -> None:
    assert cl.join_wrapped("中文行尾", "继续下一行")[0] == "中文行尾继续下一行"
    assert cl.join_wrapped("LangChain", "and LangGraph")[0] == "LangChain and LangGraph"
    joined, fixed = cl.join_wrapped("inter-", "national")
    assert joined == "international" and fixed is True
    # 产品名里的 U+2011 不是行尾断词，保留
    assert cl.join_wrapped("langchain\u2011", "community")[0] == "langchaincommunity"


def test_normalize_for_search() -> None:
    assert cl.normalize_for_search("ＡＢＣ　 全角") == "abc 全角"
    assert cl.normalize_for_search("A\u2011B") == "a-b"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("图1：说明", True),
        ("表 2-3 数据", True),
        ("图 12-1 架构", True),
        ("示意图", False),
        ("Figure 3: pipeline", True),
    ],
)
def test_looks_like_caption(text: str, expected: bool) -> None:
    assert cl.looks_like_caption(text) is expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("12", True),
        ("iv", True),
        ("第 12 页", False),
        ("12 第 1 章", False),
        ("- 12 -", True),
    ],
)
def test_looks_like_page_number(text: str, expected: bool) -> None:
    assert cl.looks_like_page_number(text) is expected


def test_code_line_numbers_only_strip_verified_prefix() -> None:
    lines = [
        line(text, 100 + idx * 12, size=6.4, mono=True)
        for idx, text in enumerate(["1 a", "2 b", "3 c", "4 d", "2024 = 42"])
    ]
    rendered, stripped = cl._strip_code_line_numbers(lines)
    assert stripped == 4
    assert [text for text, _x in rendered] == ["a", "b", "c", "d", "2024 = 42"]


def test_same_page_repeated_paragraph_is_not_cross_page_duplicate() -> None:
    unit = cl.CleanUnit(
        text="需要保留的同页重复段落，内容长度超过三十个字符。" * 2,
        kind=cl.UnitKind.BODY, page_start=14, page_end=14, y0=100, y1=119,
    )
    seen = {}
    assert cl._maybe_dup(unit, seen) is False
    assert cl._maybe_dup(unit, seen) is False
    unit.page_start = 15
    assert cl._maybe_dup(unit, seen) is True


def test_strip_code_line_numbers_inline_pattern() -> None:
    lines = [
        line("1 import os", 100, size=6.4, mono=True),
        line("2 print(1)", 112, size=6.4, mono=True),
        line("3 x = 1", 124, size=6.4, mono=True),
    ]
    texts, stripped = cl._strip_code_line_numbers(lines)
    assert stripped == 3
    assert [t for t, _x in texts] == ["import os", "print(1)", "x = 1"]


def test_strip_code_line_numbers_separate_rows() -> None:
    lines = [
        line("1", 100, size=6.4, mono=True),
        line("import os", 100, x0=90, size=6.4, mono=True),
        line("2", 112, size=6.4, mono=True),
        line("print(1)", 112, x0=90, size=6.4, mono=True),
        line("3", 124, size=6.4, mono=True),
        line("x = 1", 124, x0=90, size=6.4, mono=True),
    ]
    texts, stripped = cl._strip_code_line_numbers(lines)
    assert stripped == 3
    assert [t for t, _x in texts] == ["import os", "print(1)", "x = 1"]


def test_does_not_strip_non_sequential_numbers() -> None:
    lines = [
        line("2024 年数据", 100, size=6.4, mono=True),
        line("2025 年数据", 112, size=6.4, mono=True),
        line("同比 +20%", 124, size=6.4, mono=True),
    ]
    texts, stripped = cl._strip_code_line_numbers(lines)
    assert stripped == 0
    assert [t for t, _x in texts][0] == "2024 年数据"


def test_render_code_restores_indent() -> None:
    lines = [
        line("1", 100, x0=80, size=6.4, mono=True),
        line("def f():", 100, x0=90, size=6.4, mono=True),
        line("2", 112, x0=80, size=6.4, mono=True),
        line("return 1", 112, x0=110, size=6.4, mono=True),
        line("3", 124, x0=80, size=6.4, mono=True),
        line("f()", 124, x0=90, size=6.4, mono=True),
    ]
    text, stripped = cl._render_code(lines)
    assert stripped == 3
    assert "def f():" in text
    assert "\n  return 1" in text


# --------------------------------------------------------------------------- #
# 整本清洗
# --------------------------------------------------------------------------- #
def test_clean_book_rules() -> None:
    book = make_extract(synthetic_pages())
    result = cl.clean_book(book)
    rules = result.stats.rules

    # C1：目录页整页剔除
    assert rules["C1"]["pages"] == 1
    assert result.pages[0].blank
    # C2 / C3
    assert rules["C2"]["lines"] >= 1
    assert rules["C3"]["lines"] >= 1
    # C4：代码行号剥离
    assert rules["C4"]["stripped_line_numbers"] == 3
    # C5：硬换行合并
    assert rules["C5"]["merged_line_joins"] >= 2
    # C8：图注
    assert rules["C8"]["captions"] == 1
    # C9：参考文献
    assert rules["C9"]["reference_units"] >= 1

    units = result.units()
    body_texts = [u.text for u in units if u.kind == cl.UnitKind.BODY]
    assert any("这是第一段的第一行文字，这是第一段结尾。" == t for t in body_texts)
    # 页眉不进入正文
    assert all("第 1 章 提示链" != t for t in body_texts)

    code_units = [u for u in units if u.kind == cl.UnitKind.CODE]
    assert code_units, "应识别出代码块"
    code_text = code_units[0].text
    assert "import os" in code_text
    assert not code_text.startswith("1 ")

    caption_units = [u for u in units if u.kind == cl.UnitKind.CAPTION]
    assert caption_units and caption_units[0].text.startswith("图1：")

    reference_units = [u for u in units if u.kind == cl.UnitKind.REFERENCE]
    assert reference_units and "参考资料一" in reference_units[0].text

    # 页眉/页脚行不产生正文单元
    assert result.stats.dropped_lines >= 2


def test_clean_book_caption_continuation() -> None:
    pages = [
        page(14, [
            line("图2：图注第一行，", 300),
            line("图注第二行。", 319, x1=120),
            line("正文在此开始。", 360),
        ])
    ]
    book = make_extract(pages)
    result = cl.clean_book(book)
    captions = [u for u in result.units() if u.kind == cl.UnitKind.CAPTION]
    assert captions and captions[0].text == "图2：图注第一行，图注第二行。"


# --------------------------------------------------------------------------- #
# 真实 PDF 冒烟
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def design_clean():
    book = ex.extract_book("design")
    return book, cl.clean_book(book)


def test_real_design_clean_stats(design_clean) -> None:
    book, result = design_clean
    rules = result.stats.rules
    assert rules["C1"]["pages"] == 13
    assert rules["C2"]["lines"] > 100
    assert rules["C3"]["lines"] > 100
    assert rules["C4"]["stripped_line_numbers"] > 500
    assert rules["C5"]["merged_line_joins"] > 1000
    assert rules["C8"]["captions"] > 20
    assert rules["C9"]["reference_units"] > 20

    units = result.units()
    assert len(units) > 2000
    kinds = {u.kind for u in units}
    assert cl.UnitKind.BODY in kinds
    assert cl.UnitKind.CODE in kinds
    assert cl.UnitKind.CAPTION in kinds
    assert cl.UnitKind.REFERENCE in kinds

    # 清洗后的正文里不应残留设计模式书页眉「16 第1 章：xxx」
    assert not any(
        u.text.startswith("16 第1 章：") for u in units if u.kind == cl.UnitKind.BODY
    )
    # 代码块内部不应带行号
    first_code = next(u for u in units if u.kind == cl.UnitKind.CODE and "import" in u.text)
    assert not first_code.text.lstrip().startswith("1 import")
