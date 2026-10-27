# -*- coding: utf-8 -*-
"""引用工具单测：摘录 / 去重 / 编号重排 / 正文 [n] 校验 / Markdown 出处列表。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.models import BOOK_TITLES, BookKey, ChunkType, Citation
from app.rag import citations as cit

DESIGN_TITLE = BOOK_TITLES[BookKey.DESIGN.value]
DEEP_TITLE = BOOK_TITLES[BookKey.DEEP.value]

LONG_TEXT = (
    "工具使用模式让模型在推理过程中调用外部函数，从而把语言能力延伸到真实世界。"
    "这一模式的关键是函数签名、调用时机与结果回填，需要注意幂等与错误处理。"
)


def make_citation(
    index: int = 1,
    chunk_id: str = "c1",
    book: str = "design",
    book_title: str | None = None,
    chapter_no: int = 5,
    chapter_title: str = "工具使用（函数调用）",
    section_no: str = "5.2",
    section_title: str = "工具调用流程",
    page: int = 67,
    page_pdf: int = 80,
    para_start: int = 3,
    para_end: int = 3,
    chunk_type: ChunkType = ChunkType.TEXT,
    excerpt: str = LONG_TEXT[:40],
    score: float = 0.9,
    rank: int = 1,
) -> Citation:
    return Citation(
        index=index,
        chunk_id=chunk_id,
        book=book,
        book_title=book_title or (DESIGN_TITLE if book == "design" else DEEP_TITLE),
        chapter_no=chapter_no,
        chapter_title=chapter_title,
        section_no=section_no,
        section_title=section_title,
        page=page,
        page_pdf=page_pdf,
        para_start=para_start,
        para_end=para_end,
        chunk_type=chunk_type,
        excerpt=excerpt,
        score=score,
        rank=rank,
    )


# --------------------------------------------------------------------------- #
# 摘录
# --------------------------------------------------------------------------- #
def test_make_excerpt_short_text_unchanged() -> None:
    assert cit.make_excerpt("短文本") == "短文本"
    assert cit.make_excerpt("") == ""
    assert cit.make_excerpt(None) == ""


def test_make_excerpt_length_in_20_40() -> None:
    excerpt = cit.make_excerpt(LONG_TEXT)
    assert 20 <= len(excerpt) <= 40
    assert excerpt.endswith("…")
    assert LONG_TEXT.startswith(excerpt[:-1])


def test_make_excerpt_prefers_sentence_boundary() -> None:
    text = "甲" * 25 + "。" + "乙" * 100
    excerpt = cit.make_excerpt(text)
    assert excerpt.endswith("。…")
    assert 20 <= len(excerpt) <= 40


def test_make_excerpt_normalizes_whitespace() -> None:
    assert cit.make_excerpt("  多  行\n文本\t片段  ") == "多 行 文本 片段"


# --------------------------------------------------------------------------- #
# 去重 / 编号
# --------------------------------------------------------------------------- #
def test_dedupe_keeps_highest_score_and_order() -> None:
    low = make_citation(index=1, chunk_id="c1", score=0.3)
    other = make_citation(index=2, chunk_id="c2", score=0.5)
    high = make_citation(index=1, chunk_id="c1", score=0.95)

    result = cit.dedupe_citations([low, other, high])

    assert [c.chunk_id for c in result] == ["c1", "c2"]
    assert result[0].score == pytest.approx(0.95)
    assert len(result) == 2


def test_assign_indices_renumbers_continuously() -> None:
    citations = [
        make_citation(index=0, chunk_id="a", rank=0),
        make_citation(index=0, chunk_id="b", rank=0),
        make_citation(index=9, chunk_id="c", rank=9),
    ]
    renumbered = cit.assign_indices(citations)

    assert [c.index for c in renumbered] == [1, 2, 3]
    assert [c.rank for c in renumbered] == [1, 2, 3]
    # 原对象不被原地修改
    assert [c.index for c in citations] == [0, 0, 9]


# --------------------------------------------------------------------------- #
# [n] 解析 / 校验
# --------------------------------------------------------------------------- #
def test_parse_citation_indices() -> None:
    assert cit.parse_citation_indices("[1] 内容 [2][3]") == [1, 2, 3]
    assert cit.parse_citation_indices("全角［4］也行") == [4]
    assert cit.parse_citation_indices("没有引用标记") == []
    assert cit.parse_citation_indices(None) == []


def test_verify_citations_all_valid() -> None:
    citations = [
        make_citation(index=1, chunk_id="a"),
        make_citation(index=2, chunk_id="b"),
    ]
    answer = "工具使用模式让模型调用外部函数[1]，需要注意错误处理[2]。"
    assert cit.verify_citations(answer, citations) is True


def test_verify_citations_out_of_range_is_false() -> None:
    citations = [make_citation(index=1, chunk_id="a")]
    assert cit.verify_citations("只有一个来源[1]，这里越界[9]。", citations) is False


def test_verify_citations_requires_each_paragraph_or_list_item_to_cite() -> None:
    citations = [make_citation(index=1, chunk_id="a")]
    # 同一段的多个句子可以共用段末引用，无需机械重复编号。
    assert cit.verify_citations("第一项有出处。第二项也有出处[1]。", citations) is True
    assert cit.verify_citations("第一项有出处[1]。\n\n第二段没有出处。", citations) is False
    assert cit.verify_citations("- 第一项有出处[1]。\n- 第二项没有出处。", citations) is False
    assert cit.cited_indices("回答[1][1]。\n\n**引用出处**\n[2] 来源") == [1]


def test_verify_citations_handles_headings_continuations_and_decimals() -> None:
    citations = [make_citation(index=1, chunk_id="a")]
    answer = (
        "**处理方案**\n"
        "压缩率可达到4.3%，需要保留任务状态。\n"
        "后续句子继续说明同一观点[1]。"
    )
    assert cit.verify_citations(answer, citations) is True
    assert cit.verify_citations("实施步骤：\n先压缩工具结果，再保留最近消息[1]。", citations) is True


def test_citation_issues_reports_actionable_reason() -> None:
    citations = [make_citation(index=1, chunk_id="a")]
    issues = cit.citation_issues("有效引用[1]。\n\n这一段没有来源。", citations)
    assert any("第二段" in issue or "这一段" in issue for issue in issues)
    assert "段落或列表项缺少引用" in issues[0]
    assert "超出资料范围" in cit.citation_issues("错误编号[9]。", citations)[0]


def test_verify_citations_without_marks_is_false() -> None:
    citations = [make_citation(index=1, chunk_id="a")]
    assert cit.verify_citations("这段回答没有任何行内引用。", citations) is False


def test_verify_citations_without_citations_is_false() -> None:
    assert cit.verify_citations("有标记但没有引用列表[1]。", []) is False
    assert cit.verify_citations("[1]", None) is False


def test_verify_ignores_sources_section_marks() -> None:
    citations = [make_citation(index=1, chunk_id="a")]
    # 正文没有 [n]，只有末尾出处列表有 [1] → 视为未引用
    answer = f"正文没有标注引用。\n\n**{cit.SOURCES_HEADING}**\n[1] {citations[0].label()}"
    assert cit.verify_citations(answer, citations) is False


def test_verify_supports_fullwidth_brackets() -> None:
    citations = [make_citation(index=1, chunk_id="a")]
    assert cit.verify_citations("要点在这里［1］。", citations) is True


# --------------------------------------------------------------------------- #
# Markdown 出处列表
# --------------------------------------------------------------------------- #
def test_build_citations_markdown_format() -> None:
    citations = [
        make_citation(index=1, chunk_id="a", page=67, para_start=3),
        make_citation(
            index=2,
            chunk_id="b",
            book="deep",
            chapter_no=3,
            chapter_title="上下文工程",
            page=55,
            para_start=2,
            para_end=4,
        ),
    ]
    markdown = cit.build_citations_markdown(citations)

    lines = markdown.splitlines()
    assert lines[0] == f"**{cit.SOURCES_HEADING}**"
    assert lines[1] == f"[1] 《{DESIGN_TITLE}》第 5 章 工具使用（函数调用） · 第 67 页 · 第 3 段"
    assert lines[2] == f"[2] 《{DEEP_TITLE}》第 3 章 上下文工程 · 第 55 页 · 第 2–4 段"


def test_build_citations_markdown_empty() -> None:
    assert cit.build_citations_markdown([]) == ""
    assert cit.build_citations_markdown(None) == ""


def test_build_citations_markdown_uses_fallback_index() -> None:
    citation = make_citation(index=0, chunk_id="a")
    markdown = cit.build_citations_markdown([citation])
    assert markdown.splitlines()[1].startswith("[1] 《")
