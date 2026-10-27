# -*- coding: utf-8 -*-
"""引用构造 / 去重 / 编号重排 / 校验 / Markdown 出处列表。

引用是「可溯源、不臆造」的落地点（需求 R3）：正文用 ``[n]`` 指向
:class:`app.models.Citation`，末尾给出「书名 · 章 · 页 · 段」。
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from app.models import Citation

# 摘录长度：设计文档要求 20–40 字原文；内容本身更短时原样返回（不编造）。
EXCERPT_MIN = 20
EXCERPT_MAX = 40

# 引用出处小标题；正文与出处列表通过它切分，避免把出处列表里的 [n] 当成正文引用。
SOURCES_HEADING = "引用出处"

_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_END = "。！？；…，、.!?;,:："
_CITATION_MARK_RE = re.compile(r"[\[［]\s*(\d{1,3})\s*[\]］]")

__all__ = [
    "EXCERPT_MAX",
    "EXCERPT_MIN",
    "SOURCES_HEADING",
    "assign_indices",
    "build_citations_markdown",
    "citation_issues",
    "cited_indices",
    "dedupe_citations",
    "format_citation_line",
    "make_excerpt",
    "parse_citation_indices",
    "verify_citations",
]


# --------------------------------------------------------------------------- #
# 文本处理
# --------------------------------------------------------------------------- #
def clean_text(text: str | None) -> str:
    """压缩空白、去首尾空格（保留单空格，代码/英文不受影响）。"""
    return _WHITESPACE_RE.sub(" ", (text or "")).strip()


def make_excerpt(content: str | None, max_len: int = EXCERPT_MAX) -> str:
    """取原文开头的一段作为摘录（20–40 字，优先在标点处截断）。"""
    text = clean_text(content)
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    # 为省略号预留 1 个字符
    hard = max(EXCERPT_MIN, max_len - 1)
    window = text[:hard]
    cut = 0
    for idx in range(len(window) - 1, EXCERPT_MIN - 1, -1):
        if window[idx] in _SENTENCE_END:
            cut = idx + 1
            break
    if cut <= 0:
        cut = hard
    return text[:cut].rstrip() + "…"


# --------------------------------------------------------------------------- #
# 去重 / 编号
# --------------------------------------------------------------------------- #
def dedupe_citations(citations: Iterable[Citation]) -> list[Citation]:
    """按 ``chunk_id`` 去重：保留分数最高的一条，顺序跟随首次出现位置。"""
    items = list(citations)
    order: list[str] = []
    best: dict[str, Citation] = {}
    for cit in items:
        current = best.get(cit.chunk_id)
        if current is None:
            order.append(cit.chunk_id)
            best[cit.chunk_id] = cit
        elif cit.score > current.score:
            best[cit.chunk_id] = cit
    return [best[chunk_id] for chunk_id in order]


def assign_indices(citations: Sequence[Citation], start: int = 1) -> list[Citation]:
    """按当前顺序重排编号（index / rank 从 ``start`` 起连续）。"""
    renumbered: list[Citation] = []
    for offset, cit in enumerate(citations):
        idx = start + offset
        renumbered.append(cit.model_copy(update={"index": idx, "rank": idx}))
    return renumbered


# --------------------------------------------------------------------------- #
# 正文引用标记解析 / 校验
# --------------------------------------------------------------------------- #
def parse_citation_indices(text: str | None) -> list[int]:
    """解析文本里出现的所有 ``[n]``（含全角括号），按出现顺序返回。"""
    if not text:
        return []
    return [int(match) for match in _CITATION_MARK_RE.findall(text)]


def _answer_body(answer: str | None) -> str:
    """去掉末尾「引用出处」列表后的正文（出处列表里的 [n] 不参与校验）。"""
    text = answer or ""
    marker = SOURCES_HEADING
    if marker in text:
        text = text.split(marker, 1)[0]
    # 兼容 "**引用出处**" 之外的自定义小标题写法
    text = re.sub(r"\*{1,2}\s*$", "", text)
    return text.strip()


def cited_indices(answer: str | None) -> list[int]:
    """按出现顺序提取正文实际使用的引用编号（不含出处清单）。"""
    return list(dict.fromkeys(parse_citation_indices(_answer_body(answer))))


def _statement_blocks(body: str) -> list[str]:
    """按自然段和列表项分组，保留同一段内换行及多句话的引用上下文。"""
    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append(" ".join(current))
            current.clear()

    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        is_item = bool(re.match(r"^(?:[-*+]\s+|\d+[.)、]\s*)", line))
        is_heading = line.startswith(("#", ">")) or bool(
            re.fullmatch(r"\*\*[^*]+\*\*[:：]?", line)
        )
        if (is_item or is_heading) and current:
            flush()
        current.append(line)
        if is_heading:
            flush()
    flush()
    return blocks


def _is_structural_block(text: str) -> bool:
    """标题、引导语和降级说明不是需要单独标注来源的事实段落。"""
    if text.startswith((
        "#", ">", "当前大模型不可用", "生成答案未通过引用校验", "以下为", "以上片段",
        "两本书均有相关内容", "本轮仅检索到",
    )):
        return True
    if re.fullmatch(r"\*\*[^*]+\*\*[:：]?", text):
        return True
    return bool(re.fullmatch(r"[^。！？!?；;\n]{1,24}[:：]", text))


def citation_issues(
    answer: str | None, citations: Sequence[Citation] | None
) -> list[str]:
    """返回可诊断的引用问题；每个自然段或列表项至少需要一个有效编号。"""
    cits = list(citations or [])
    if not cits:
        return ["没有可用的引用片段"]
    body = _answer_body(answer)
    if not body:
        return ["回答正文为空"]
    marks = parse_citation_indices(body)
    if not marks:
        return ["正文没有行内引用编号"]
    valid = {cit.index for cit in cits if cit.index > 0}
    if not valid:
        valid = set(range(1, len(cits) + 1))
    issues: list[str] = []
    unknown = sorted(set(marks) - valid)
    if unknown:
        issues.append(f"引用编号超出资料范围：{unknown}")
    for block in _statement_blocks(body):
        text = re.sub(r"^(?:[-*+]\s+|\d+[.)、]\s*)", "", block).strip()
        if (
            text
            and not _is_structural_block(text)
            and re.search(r"[一-鿿A-Za-z]", text)
            and not parse_citation_indices(text)
        ):
            issues.append(f"段落或列表项缺少引用：{clean_text(text)[:40]}")
    return issues


def verify_citations(answer: str | None, citations: Sequence[Citation] | None) -> bool:
    """校验引用编号及自然段/列表项覆盖，而非机械要求每句话重复编号。

    这是格式与覆盖检查，无法证明被引用原文确实支持陈述的事实。
    """
    return not citation_issues(answer, citations)


# --------------------------------------------------------------------------- #
# Markdown 出处列表
# --------------------------------------------------------------------------- #
def format_citation_line(citation: Citation, index: int | None = None) -> str:
    """单行出处：``[1] 《书名》第 X 章 标题 · 第 Y 页 · 第 Z 段``。"""
    idx = citation.index if index is None else index
    if idx <= 0:
        idx = 1
    return f"[{idx}] {citation.label()}"


def build_citations_markdown(citations: Sequence[Citation] | None) -> str:
    """生成「引用出处」Markdown 列表；无引用时返回空串。"""
    cits = list(citations or [])
    if not cits:
        return ""
    lines = [f"**{SOURCES_HEADING}**"]
    for offset, cit in enumerate(cits, 1):
        idx = cit.index if cit.index > 0 else offset
        lines.append(format_citation_line(cit, index=idx))
    return "\n".join(lines)
