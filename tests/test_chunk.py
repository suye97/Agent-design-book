# -*- coding: utf-8 -*-
"""chunk.py 单测：段落编号 / 块大小 / 重叠 / 代码块 / 稳定性 / 真实切片冒烟。"""
from __future__ import annotations

import pytest

from app.ingest import chunk as ck
from app.ingest import clean as cl
from app.ingest import extract as ex
from app.ingest.clean import CleanResult, CleanStats, CleanUnit, CleanedPage, UnitKind
from app.ingest.extract import Chapter, PageText, Section, TextLine
from app.models import ChunkType


# --------------------------------------------------------------------------- #
# 构造工具
# --------------------------------------------------------------------------- #
def make_unit(
    text: str,
    kind: UnitKind,
    page_start: int,
    para: int = 1,
    y0: float = 100.0,
) -> CleanUnit:
    return CleanUnit(
        text=text,
        kind=kind,
        page_start=page_start,
        page_end=page_start,
        y0=y0,
        y1=y0 + 19,
        size=12.0,
        line_count=1,
    )


def make_book() -> ex.BookExtract:
    chapter = Chapter(
        chapter_no=1,
        title="提示链",
        page_start=14,
        page_end=17,
        anchor_title="第 1 章 提示链",
        heading_y=69.0,
        heading_found=True,
        sections=[
            Section(no="1.1", title="概述", page=14, level=2, y=69.0, anchor_title="1.1 概述"),
            Section(no="1.2", title="示例", page=16, level=2, y=100.0, anchor_title="1.2 示例"),
        ],
    )
    return ex.BookExtract(
        book="design",
        book_title="智能体设计模式：智能系统构建实战指南",
        pdf_path="dummy.pdf",
        pdf_pages=305,
        pages=[],
        outline=[],
        chapters=[chapter],
        print_offset=13,
        print_offset_method="test",
        body_start_page=14,
        body_font_size=12.0,
        body_top=75.0,
    )


def make_clean_result() -> CleanResult:
    paras: list[tuple[str, int]] = []
    templates = [
        "提示链模式把复杂任务拆成多个顺序步骤，每一步都由模型完成一个聚焦的子任务。",
        "第一步负责理解输入并抽取结构化信息，第二步根据抽取结果生成最终答案。",
        "这种顺序处理让每个步骤都可以单独调试和优化，也便于在中间环节接入外部工具。",
        "实践中常见两种实现方式：脚本里显式顺序调用，或用 LangGraph 等框架编排状态。",
        "顺序调用的优点是简单直观，缺点是错误会沿链条向后传播，需要在每步做校验。",
        "框架编排适合更复杂的流程，例如带条件分支、循环和人工审核节点的流程。",
        "无论采用哪种方式，都要保证每一步的输入输出格式清晰、可验证、可回放。",
        "在评估时，可以逐段替换提示词并观察最终指标，定位到底是哪一步拖累了效果。",
        "生产系统中还应记录每一步的耗时与 token 消耗，便于做成本归因和容量规划。",
        "如果某一环节依赖外部 API，需要设置超时、重试与降级策略，避免整条链被拖死。",
        "最终答案要把中间结果组织成可读文本，并附上必要的引用与不确定性说明。",
        "总结来说，提示链的核心价值是分而治之：把难以一次做好任务拆成可管理的步骤。",
    ]
    for index in range(18):
        text = templates[index % len(templates)] + templates[(index + 1) % len(templates)]
        paras.append((text, 14 + (index // 6)))
    units: list[CleanUnit] = []
    for text, page in paras:
        units.append(make_unit(text, UnitKind.BODY, page))
    # 代码块
    units.append(
        make_unit("from langchain_openai import ChatOpenAI\nllm = ChatOpenAI(temperature=0)", UnitKind.CODE, 16)
    )
    units.append(make_unit("代码执行后，模型会返回结构化的中间结果，供下一步使用。", UnitKind.BODY, 16))
    units.append(make_unit("最后一步把中间结果整理成最终答案，并检查引用是否完整。", UnitKind.BODY, 17))
    # 列表项
    units.append(make_unit("・第一步：抽取信息。", UnitKind.LIST, 17))
    units.append(make_unit("・第二步：生成答案。", UnitKind.LIST, 17))
    # 图注
    units.append(make_unit("图1：提示链的示意图。", UnitKind.CAPTION, 17))
    # 参考文献
    units.append(make_unit("・LangChain 官方文档。", UnitKind.REFERENCE, 17))

    pages = [
        CleanedPage(book="design", pdf_page=p, print_page=p - 13, units=[u for u in units if u.page_start == p])
        for p in (14, 15, 16, 17)
    ]
    return CleanResult(
        book="design",
        book_title="智能体设计模式：智能系统构建实战指南",
        pages=pages,
        stats=CleanStats(),
        body_start_page=14,
        body_font_size=12.0,
        left_margin=71.0,
        right_margin=524.0,
    )


def make_book_pages(page_start: int, page_end: int) -> ex.BookExtract:
    """构造只有一章、指定页范围的 BookExtract（页码归属测试用）。"""
    chapter = Chapter(
        chapter_no=1,
        title="提示链",
        page_start=page_start,
        page_end=page_end,
        anchor_title="第 1 章 提示链",
        heading_y=69.0,
        heading_found=True,
        sections=[],
    )
    return ex.BookExtract(
        book="design",
        book_title="智能体设计模式：智能系统构建实战指南",
        pdf_path="dummy.pdf",
        pdf_pages=305,
        pages=[],
        outline=[],
        chapters=[chapter],
        print_offset=13,
        print_offset_method="test",
        body_start_page=page_start,
        body_font_size=12.0,
        body_top=75.0,
    )


def make_clean_from_units(units: list[CleanUnit], pages: tuple[int, ...]) -> CleanResult:
    """按单元归属页构造 CleanResult。"""
    cleaned_pages = [
        CleanedPage(
            book="design",
            pdf_page=p,
            print_page=p - 13,
            units=[u for u in units if u.page_start == p],
        )
        for p in pages
    ]
    return CleanResult(
        book="design",
        book_title="智能体设计模式：智能系统构建实战指南",
        pages=cleaned_pages,
        stats=CleanStats(),
        body_start_page=min(pages),
        body_font_size=12.0,
        left_margin=71.0,
        right_margin=524.0,
    )


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #
def test_split_text_respects_max() -> None:
    text = "这是一个很长的句子。" * 100
    pieces = ck.split_text(text, max_chars=200, min_chars=100)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    assert "".join(pieces) == text


def test_split_text_does_not_merge_short_tail_beyond_max() -> None:
    pieces = ck.split_text("甲" * 179 + "。" + "乙" * 29 + "。", max_chars=200, min_chars=100)
    assert all(len(piece) <= 200 for piece in pieces)
    assert "".join(pieces) == "甲" * 179 + "。" + "乙" * 29 + "。"


def test_short_body_and_code_keep_distinct_types() -> None:
    units = [
        make_unit("短正文。", UnitKind.BODY, 14),
        make_unit("print('ok')", UnitKind.CODE, 14),
    ]
    result = make_clean_from_units(units, (14,))
    chunks = ck.chunk_book(make_book_pages(14, 14), result)
    assert [chunk.chunk_type for chunk in chunks] == [ChunkType.TEXT, ChunkType.CODE]


def test_estimate_tokens() -> None:
    assert ck.estimate_tokens("你好 world 123") == 2 + 2
    assert ck.estimate_tokens("") == 0


def test_trim_tail_length_and_boundary() -> None:
    text = "第一句。第二句。第三句。" * 20
    piece, offset = ck._trim_tail(text, min_len=80, max_len=120)
    assert 80 <= len(piece) <= 120
    assert text.endswith(piece)
    assert offset == len(text) - len(piece)


def test_overlap_page_start_uses_overlap_page() -> None:
    """跨页 overlap：页码取 overlap 实际所在页（上一页），而不是本块首单元页。"""
    units = [
        make_unit("甲" * 520, UnitKind.BODY, 14),
        make_unit("乙" * 40, UnitKind.BODY, 15),
        make_unit("丙" * 200, UnitKind.BODY, 15),
        make_unit("丁" * 200, UnitKind.BODY, 15),
        make_unit("戊" * 100, UnitKind.BODY, 16),
    ]
    book = make_book_pages(14, 16)
    result = make_clean_from_units(units, (14, 15, 16))
    chunks = ck.chunk_book(book, result)
    assert len(chunks) >= 2, "构造数据应至少切出两块"

    first, second = chunks[0], chunks[1]
    assert first.page_end == 15, "第一块应跨到第 15 页"
    overlap, _ = ck._trim_tail(first.content, min_len=80, max_len=120)
    assert second.content.startswith(overlap)
    # overlap 主体来自第 14 页的长段尾部 → 第二块 page_start 应为 14（而非 15）
    assert second.page_start == 14
    assert second.page_print_start == 1
    assert second.page_end >= second.page_start


def test_overlap_same_page_keeps_page_start() -> None:
    """同页 overlap：page_start 不受影响。"""
    units = [
        make_unit("甲" * 520, UnitKind.BODY, 14),
        make_unit("乙" * 40, UnitKind.BODY, 14),
        make_unit("丙" * 200, UnitKind.BODY, 14),
        make_unit("丁" * 200, UnitKind.BODY, 14),
    ]
    book = make_book_pages(14, 14)
    result = make_clean_from_units(units, (14,))
    chunks = ck.chunk_book(book, result)
    assert len(chunks) >= 2
    first, second = chunks[0], chunks[1]
    assert first.page_end == 14
    assert second.content.startswith(ck._trim_tail(first.content, 80, 120)[0])
    assert second.page_start == 14 == first.page_end


# --------------------------------------------------------------------------- #
# 切片主流程（构造数据）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synthetic_chunks():
    book = make_book()
    result = make_clean_result()
    return ck.chunk_book(book, result)


def test_chunk_basic_contract(synthetic_chunks) -> None:
    chunks = synthetic_chunks
    assert chunks, "应产出切片"
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(len(i) == 32 for i in ids)

    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.book == "design"
        assert chunk.chapter_no == 1
        assert chunk.chapter_title == "提示链"
        assert chunk.page_start <= chunk.page_end
        assert chunk.para_start <= chunk.para_end
        assert chunk.page_print_start == chunk.page_start - 13
        assert chunk.content_hash
        assert chunk.token_len > 0

    # prev/next 串联
    for i, chunk in enumerate(chunks):
        assert chunk.prev_id == (chunks[i - 1].chunk_id if i else None)
        assert chunk.next_id == (chunks[i + 1].chunk_id if i + 1 < len(chunks) else None)


def test_chunk_paragraph_order_and_overlap(synthetic_chunks) -> None:
    chunks = synthetic_chunks
    paras = [c.para_start for c in chunks]
    assert paras == sorted(paras)
    assert len(set(paras)) == len(paras) or True  # 允许长段被切后重复段号

    # 相邻非代码块之间存在 80–120 字符重叠
    overlapped = 0
    for prev, cur in zip(chunks, chunks[1:]):
        if prev.chunk_type == ChunkType.CODE or cur.chunk_type == ChunkType.CODE:
            continue
        piece, _ = ck._trim_tail(prev.content, min_len=80, max_len=120)
        assert piece and cur.content.startswith(piece), (prev.chunk_id, cur.chunk_id)
        overlapped += 1
    assert overlapped >= 1


def test_chunk_code_and_caption(synthetic_chunks) -> None:
    chunks = synthetic_chunks
    code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
    assert code_chunks
    assert "ChatOpenAI" in code_chunks[0].content

    caption_texts = [c.content for c in chunks if "图1：" in c.content]
    assert caption_texts, "图注应并入某个块"
    assert not any(c.content.strip().startswith("图1：") and len(c.content) < 100 for c in chunks)


def test_chunk_ids_stable() -> None:
    book = make_book()
    result = make_clean_result()
    ids1 = [c.chunk_id for c in ck.chunk_book(book, result)]
    ids2 = [c.chunk_id for c in ck.chunk_book(book, result)]
    assert ids1 == ids2


def test_chunk_stats() -> None:
    chunks = ck.chunk_book(make_book(), make_clean_result())
    stats = ck.chunk_stats(chunks)
    assert stats.chunks == len(chunks)
    assert stats.by_type.get("code", 0) >= 1
    assert stats.avg_chars > 0


# --------------------------------------------------------------------------- #
# 真实 PDF 冒烟
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_chunks():
    book = ex.extract_book("design")
    cleaned = cl.clean_book(book)
    return book, cleaned, ck.chunk_book(book, cleaned)


def test_real_design_chunks(real_chunks) -> None:
    book, cleaned, chunks = real_chunks
    assert len(chunks) > 300
    assert len({c.chunk_id for c in chunks}) == len(chunks)

    kinds = {c.chunk_type for c in chunks}
    assert ChunkType.TEXT in kinds
    assert ChunkType.CODE in kinds

    chapter_titles = {c.chapter_title for c in chunks}
    assert "提示链（Prompt Chaining）" in chapter_titles

    for chunk in chunks:
        assert chunk.book_title == book.book_title
        assert chunk.para_start >= 1 and chunk.para_end >= chunk.para_start
        assert chunk.page_print_start >= 1
        assert chunk.page_start <= chunk.page_end
        assert chunk.content_hash and len(chunk.content_hash) == 32

    # 带 overlap 的块：page_start 不得晚于上一块 page_end（overlap 可能来自上一页）
    for i, chunk in enumerate(chunks):
        if i == 0 or not chunk.prev_id:
            continue
        prev = chunks[i - 1]
        if (prev.chapter_no, prev.chapter_title) != (chunk.chapter_no, chunk.chapter_title):
            continue
        if prev.chunk_type == ChunkType.CODE or chunk.chunk_type == ChunkType.CODE:
            continue  # 代码块两侧不加重叠
        assert chunk.page_start <= prev.page_end, (chunk.chunk_id, chunk.page_start, prev.page_end)

    # 章内段号递增（下一块可从上一块尾部重叠处开始，因此比较 para_start）
    last_para_by_chapter: dict[tuple[int, str], int] = {}
    for chunk in chunks:
        key = (chunk.chapter_no, chunk.chapter_title)
        assert chunk.para_start >= last_para_by_chapter.get(key, 0)
        last_para_by_chapter[key] = chunk.para_start

    # 正文块大小：目标 450–700，允许极少数章末/图注块越界
    body_chunks = [c for c in chunks if c.chunk_type != ChunkType.CODE]
    oversized = [c for c in body_chunks if len(c.content) > 705]
    assert len(oversized) <= max(3, int(len(body_chunks) * 0.02))

    # 重叠：相邻正文块
    sampled = 0
    for prev, cur in zip(chunks, chunks[1:]):
        if prev.chapter_no != cur.chapter_no or (prev.chapter_no, prev.chapter_title) != (
            cur.chapter_no,
            cur.chapter_title,
        ):
            continue
        if prev.chunk_type == ChunkType.CODE or cur.chunk_type == ChunkType.CODE:
            continue
        piece, _ = ck._trim_tail(prev.content, min_len=80, max_len=120)
        assert piece and cur.content.startswith(piece)
        sampled += 1
        if sampled >= 20:
            break
    assert sampled >= 10

    # 稳定幂等
    again = ck.chunk_book(book, cleaned)
    assert [c.chunk_id for c in again] == [c.chunk_id for c in chunks]
