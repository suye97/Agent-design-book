# -*- coding: utf-8 -*-
"""数据清洗：C1–C10 规则，先统计再删除，每一步可单独单测。

规则（对应 docs/DESIGN.md 5.1）：
    C1 目录/版权整页剔除（依据大纲定位正文起点）
    C2 页眉（按"正文顶线 + 字号"判定，不用全局黑名单）
    C3 页脚孤立页码（纯数字/罗马数字、位于页面底部）
    C4 代码行号剥离（先识别代码块，再按行号连续递增校验）
    C5 硬换行合并（行尾未满行 → 与下一行合并；中英文分别处理空格）
    C6 连字符/软连字符（跨行断词修复；U+00AD 直接删除）
    C7 LaTeX 连写（保留原文，只提供 normalize_for_search 供检索侧归一化）
    C8 图注/表注标记（kind=caption，不单独成块）
    C9 参考文献标记（kind=reference，默认降权/过滤）
    C10 空白/重复行（不产出空行；跨页重复段落按归一化文本去重）

对外主入口：
    clean_book(book_extract) -> CleanResult
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from app.ingest.extract import (
    REFERENCE_TITLES,
    BookExtract,
    PageText,
    TextLine,
    normalize_title,
)
from app.logging_conf import get_logger

logger = get_logger("ingest.clean")


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
class UnitKind(str, Enum):
    BODY = "body"
    LIST = "list"
    CODE = "code"
    CAPTION = "caption"
    REFERENCE = "reference"
    HEADING = "heading"


@dataclass(slots=True)
class CleanUnit:
    """清洗后的最小文本单元：自然段 / 代码块 / 图注 / 标题。"""

    text: str
    kind: UnitKind
    page_start: int
    page_end: int
    y0: float
    y1: float
    size: float = 0.0
    line_count: int = 1

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "kind": self.kind.value,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "y0": round(self.y0, 2),
            "y1": round(self.y1, 2),
            "size": self.size,
            "line_count": self.line_count,
        }


@dataclass(slots=True)
class CleanedPage:
    book: str
    pdf_page: int
    print_page: int
    units: list[CleanUnit] = field(default_factory=list)
    dropped_lines: int = 0
    blank: bool = False

    @property
    def text(self) -> str:
        return "\n".join(u.text for u in self.units if u.text.strip())


@dataclass(slots=True)
class CleanStats:
    """每规则命中统计（报告与测试都依赖它）。"""

    rules: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_pages: list[int] = field(default_factory=list)
    dropped_lines: int = 0
    kept_units: int = 0
    body_units: int = 0
    code_blocks: int = 0

    def hit(self, rule: str, **counts: int) -> None:
        bucket = self.rules.setdefault(rule, {})
        for key, value in counts.items():
            bucket[key] = bucket.get(key, 0) + int(value)

    def as_dict(self) -> dict:
        return {
            "rules": self.rules,
            "dropped_pages": len(self.dropped_pages),
            "dropped_page_numbers": self.dropped_pages,
            "dropped_lines": self.dropped_lines,
            "kept_units": self.kept_units,
            "body_units": self.body_units,
            "code_blocks": self.code_blocks,
        }


@dataclass(slots=True)
class CleanResult:
    book: str
    book_title: str
    pages: list[CleanedPage]
    stats: CleanStats
    body_start_page: int
    body_font_size: float
    left_margin: float
    right_margin: float
    warnings: list[str] = field(default_factory=list)

    def units(self) -> list[CleanUnit]:
        """按页序展开全部非标题单元（chunk.py 的输入）。"""
        out: list[CleanUnit] = []
        for page in self.pages:
            out.extend(u for u in page.units if u.kind != UnitKind.HEADING)
        return out

    def headings(self) -> list[CleanUnit]:
        out: list[CleanUnit] = []
        for page in self.pages:
            out.extend(u for u in page.units if u.kind == UnitKind.HEADING)
        return out


# --------------------------------------------------------------------------- #
# 正则与工具
# --------------------------------------------------------------------------- #
_LIST_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"[・·•●▪◦►▶\-*]\s*"
    r"|[（(]?\d{1,2}[)）.、．]\s+"
    r"|\d{1,2}[)）.、．](?=[^\d])"
    r"|[①-⑳]\s*"
    r"|[a-zA-Z][)．.]\s+"
    r")"
)

_CAPTION_RE = re.compile(
    r"^\s*(?:图|表|Figure|Table)\s*[\d０-９一二三四五六七八九十]+"
    r"(?:\s*[-－—.．]\s*\d+)?\s*[：:、.． ]"
)

_PAGE_NUM_RE = re.compile(r"^\s*(?:[-—–]\s*)?(?:\d{1,4}|[ivxlcdm]{1,8})(?:\s*[-—–])?\s*$", re.I)

_CODE_LINE_NUM_RE = re.compile(r"^\s*(\d{1,4})[ \t]+(.*)$")
_CODE_BARE_NUM_RE = re.compile(r"^\s*(\d{1,4})\s*$")

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_SOFT_HYPHEN = "\u00ad"
_LINE_END_HYPHENS = ("-", "\u2010", "\u2011")
_CONTINUATION_MARK = "\u21aa"  # '↪'：本书 PDF 中代码折行标记


def normalize_for_search(text: str) -> str:
    """检索侧归一化（C7）：全角→半角、统一连字符、小写、压缩空白。

    只用于检索 / 去重 / 哈希，不写回 content，避免破坏可引用性。
    """
    if not text:
        return ""
    out: list[str] = []
    for ch in unicodedata.normalize("NFKC", text):
        if ch in "\u2010\u2011\u2012\u2013\u2014\u2212":
            ch = "-"
        if ch.isspace():
            out.append(" ")
            continue
        out.append(ch.lower())
    return re.sub(r" +", " ", "".join(out)).strip()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_cjk_char(ch: str) -> bool:
    return bool(ch and _CJK_RE.match(ch))


def _is_ascii_alnum(ch: str) -> bool:
    return bool(ch) and ch.isascii() and ch.isalnum()


def join_wrapped(left: str, right: str) -> tuple[str, bool]:
    """把两行拼接成同一段落，返回 (结果, 是否修复了行尾连字符)。"""
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right, False
    if not right:
        return left, False
    if left.endswith(_CONTINUATION_MARK):
        left = left[: -len(_CONTINUATION_MARK)].rstrip()
        return left + right, False
    if left.endswith(_LINE_END_HYPHENS) and right[:1].isalpha():
        # 跨行断词：去掉行尾连字符后直接拼接
        return left[:-1].rstrip() + right, True
    if is_cjk_char(left[-1]) or is_cjk_char(right[0]):
        return left + right, False
    if _is_ascii_alnum(left[-1]) and _is_ascii_alnum(right[0]):
        return left + " " + right, False
    return left + right, False


def looks_like_page_number(text: str) -> bool:
    return bool(_PAGE_NUM_RE.match(text or ""))


def looks_like_caption(text: str) -> bool:
    return bool(_CAPTION_RE.match(text or ""))


def looks_like_list_item(text: str) -> bool:
    return bool(_LIST_MARKER_RE.match(text or ""))


def is_reference_title(title: str) -> bool:
    return normalize_title(title) in {normalize_title(t) for t in REFERENCE_TITLES}


# --------------------------------------------------------------------------- #
# 页面几何（页眉/页脚判据）
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PageGeometry:
    header_cut: float
    footer_cut: float
    left_margin: float
    right_margin: float


def estimate_geometry(extract: BookExtract) -> PageGeometry:
    """用正文行的分布估计页眉/页脚分界线与左右页边距。"""
    heights = [p.height for p in extract.body_pages if p.height] or [842.0]
    page_height = max(heights)
    body_size = extract.body_font_size or 12.0

    lefts: list[float] = []
    rights: list[float] = []
    for page in extract.body_pages:
        for line in page.lines:
            if abs(line.size - body_size) <= max(0.3, body_size * 0.06):
                lefts.append(line.x0)
                rights.append(line.x1)
    lefts.sort()
    rights.sort()

    def _pct(values: list[float], q: float, default: float) -> float:
        if not values:
            return default
        idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return values[idx]

    left_margin = _pct(lefts, 0.05, 60.0)
    right_margin = _pct(rights, 0.95, 500.0)
    body_top = extract.body_top if extract.body_top > 0 else page_height * 0.09
    header_cut = max(0.0, body_top - 5.0)
    return PageGeometry(
        header_cut=header_cut,
        footer_cut=page_height * 0.88,
        left_margin=left_margin,
        right_margin=right_margin,
    )


# --------------------------------------------------------------------------- #
# 代码块
# --------------------------------------------------------------------------- #
def _strip_code_line_numbers(
    lines: Sequence[TextLine],
) -> tuple[list[tuple[str, float]], int]:
    """识别并剥离代码行号；返回 ([(代码行文本, 行左边界 x0)], 剥离的行号数)。

    支持两种排版：
      A) 行内行号：``12 import os``；
      B) 独立行号行：本 PDF 的设计模式书把行号排成独立文本行（``12`` 一行、
         代码一行），需要按"1..N 连续递增"识别后整行剔除。
    只有连续递增才算行号，避免误伤以数字开头的代码。
    """
    texts = [ln.text.strip() for ln in lines]
    xs = [ln.x0 for ln in lines]

    # --- 模式 B：独立行号行（允许从任意行号开始，要求严格 +1 递增） ---
    number_rows: list[tuple[int, int]] = [
        (idx, int(text)) for idx, text in enumerate(texts) if text.isdigit()
    ]
    if len(number_rows) >= 3 and len(number_rows) >= max(3, int(len(texts) * 0.5)):
        seq_ok = all(
            number_rows[i + 1][1] == number_rows[i][1] + 1
            for i in range(len(number_rows) - 1)
        )
        if seq_ok:
            drop = {idx for idx, _num in number_rows}
            kept = [(texts[idx], xs[idx]) for idx in range(len(texts)) if idx not in drop]
            return kept, len(number_rows)

    # --- 模式 A：行内行号（同样允许从任意行号开始） ---
    parsed: list[tuple[int | None, str]] = []
    for text in texts:
        m = _CODE_LINE_NUM_RE.match(text)
        if m:
            parsed.append((int(m.group(1)), m.group(2)))
            continue
        m2 = _CODE_BARE_NUM_RE.match(text)
        if m2:
            parsed.append((int(m2.group(1)), ""))
            continue
        parsed.append((None, text))

    if parsed and parsed[0][0] is not None and parsed[0][0] <= 1000:
        prev = parsed[0][0] - 1
        seq = 0
        for num, _text in parsed:
            if num is None or num != prev + 1:
                break
            prev = num
            seq += 1
        need = max(2, int(len(parsed) * 0.8))
        if seq >= need and (len(parsed) > 2 or parsed[0][0] >= 1):
            # Only strip the verified consecutive prefix; later numeric code is real content.
            kept = [
                (parsed[idx][1].rstrip() if idx < seq else texts[idx].rstrip(), xs[idx])
                for idx in range(len(parsed))
            ]
            return kept, seq
    return [(t.rstrip(), xs[idx]) for idx, t in enumerate(texts)], 0


def _render_code(lines: Sequence[TextLine]) -> tuple[str, int]:
    """把代码行拼成代码块文本（还原缩进、处理 '↪' 折行标记）。"""
    rendered, stripped = _strip_code_line_numbers(lines)

    # 依据行左边界还原缩进（每级 2 空格）
    xs = [x for _text, x in rendered if _text.strip()]
    indent_unit = 10.0
    base = 0.0
    if xs:
        base = min(xs)
        diffs = sorted({round(x - base, 1) for x in xs if x - base > 3.0})
        if diffs:
            # 跨页同一代码块的左边界会有 2~5pt 抖动，小于 8pt 的差异不算缩进层级
            indent_unit = max(8.0, diffs[0])

    out: list[str] = []
    for text, x0 in rendered:
        piece = text.rstrip()
        if piece and xs:
            level = int(round(max(0.0, x0 - base) / indent_unit))
            prefix = "  " * min(level, 8)
            piece = prefix + piece.lstrip()
        if piece.rstrip().endswith(_CONTINUATION_MARK):
            piece = piece.rstrip()[: -len(_CONTINUATION_MARK)].rstrip()
            if out:
                out[-1] = out[-1].rstrip() + piece.lstrip()
            else:
                out.append(piece)
            continue
        out.append(piece)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out), stripped


# --------------------------------------------------------------------------- #
# 行分类
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _LineTag:
    line: TextLine
    kind: str  # body / code / caption / heading / dropped


def _classify_lines(
    page: PageText,
    *,
    geometry: PageGeometry,
    body_size: float,
    heading_norms: set[str],
) -> tuple[list[_LineTag], dict[str, int]]:
    """对单页的行做初步分类，返回 (标签列表, 各规则命中数)。"""
    tags: list[_LineTag] = []
    hits: dict[str, int] = {}
    heading_min_size = body_size * 1.05
    max_header_size = body_size * 1.35

    for line in page.lines:
        text = line.text.strip()
        norm = normalize_title(line.text)
        # --- 页眉：正文顶线之上，且不是大字号标题 ---
        if line.y0 < geometry.header_cut and line.size < max_header_size:
            hits["C2"] = hits.get("C2", 0) + 1
            tags.append(_LineTag(line, "dropped"))
            continue
        # --- 标题（章/节，来自大纲；字号需大于正文；参考文献标题即使不在大纲也识别） ---
        if (
            (norm in heading_norms or is_reference_title(text))
            and line.size + 1e-6 >= heading_min_size
        ):
            tags.append(_LineTag(line, "heading"))
            continue
        # --- 代码行（先于页脚页码判定：代码块行号常在页面底部） ---
        if line.is_code_like:
            tags.append(_LineTag(line, "code"))
            continue
        # --- 页脚孤立页码（页面底部 + 居中/靠右，避免误伤代码行号） ---
        if (
            line.y0 > geometry.footer_cut
            and line.x0 >= (page.width or 600.0) * 0.3
            and looks_like_page_number(text)
        ):
            hits["C3"] = hits.get("C3", 0) + 1
            tags.append(_LineTag(line, "dropped"))
            continue
        # --- 图注 / 表注 ---
        if looks_like_caption(text):
            tags.append(_LineTag(line, "caption"))
            continue
        tags.append(_LineTag(line, "body"))
    return tags, hits


# --------------------------------------------------------------------------- #
# 段落合并
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _Para:
    text: str
    last_text: str
    page_start: int
    page_end: int
    y0: float
    y1: float
    size: float
    last_x1: float
    line_count: int = 1


def _merge_paragraphs(
    lines: Sequence[TextLine],
    *,
    right_margin: float,
) -> tuple[list[_Para], int, int]:
    """把正文行合并成自然段。

    Returns:
        (段落列表, 合并次数 C5, 连字符修复次数 C6)
    """
    paras: list[_Para] = []
    cur: _Para | None = None
    merges = 0
    hyphen_fixes = 0

    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        if cur is None:
            cur = _Para(
                text=text,
                last_text=text,
                page_start=line.page,
                page_end=line.page,
                y0=line.y0,
                y1=line.y1,
                size=line.size,
                last_x1=line.x1,
            )
            continue
        same_size = abs(line.size - cur.size) <= max(0.4, cur.size * 0.12)
        page_contiguous = line.page <= cur.page_end + 1
        prev_full = cur.last_x1 >= right_margin - 18.0
        if same_size and page_contiguous and prev_full:
            joined, fixed = join_wrapped(cur.text, text)
            cur.text = joined
            cur.last_text = text
            cur.page_end = line.page
            cur.y1 = line.y1
            cur.last_x1 = line.x1
            cur.line_count += 1
            merges += 1
            hyphen_fixes += 1 if fixed else 0
            continue
        paras.append(cur)
        cur = _Para(
            text=text,
            last_text=text,
            page_start=line.page,
            page_end=line.page,
            y0=line.y0,
            y1=line.y1,
            size=line.size,
            last_x1=line.x1,
        )
    if cur is not None:
        paras.append(cur)
    return paras, merges, hyphen_fixes


def _finalize_kind(text: str) -> UnitKind:
    if looks_like_list_item(text):
        return UnitKind.LIST
    return UnitKind.BODY


_TERMINAL_PUNCT = set("。．！？；：….!?;:）)】》\"'”’")


def _is_caption_continuation(prev: TextLine, cur: TextLine, right_margin: float) -> bool:
    """图注被排版换行：上一行是满行且字号一致时，当前行并入图注。"""
    return (
        prev.x1 >= right_margin - 18.0
        and abs(cur.size - prev.size) <= max(0.4, prev.size * 0.12)
    )


def _continues_paragraph(unit: CleanUnit, line: TextLine) -> bool:
    """图注插入后，判断当前正文行是否仍是上一段的续写。"""
    if unit.kind not in (UnitKind.BODY, UnitKind.LIST, UnitKind.REFERENCE):
        return False
    if not unit.text or unit.text[-1] in _TERMINAL_PUNCT:
        return False
    if line.page > unit.page_end + 1:
        return False
    if abs(line.size - unit.size) > max(0.4, unit.size * 0.12):
        return False
    first = line.text.strip()[:1]
    return bool(first) and (is_cjk_char(first) or first.islower() or first.isdigit())


# --------------------------------------------------------------------------- #
# 主清洗流程
# --------------------------------------------------------------------------- #
def clean_book(extract: BookExtract) -> CleanResult:
    """执行 C1–C10，返回清洗后的页/单元与统计。"""
    geometry = estimate_geometry(extract)
    body_size = extract.body_font_size or 12.0

    heading_norms: set[str] = set()
    for chapter in extract.chapters:
        heading_norms.add(normalize_title(chapter.anchor_title))
        for section in chapter.sections:
            heading_norms.add(normalize_title(section.anchor_title))
    heading_norms.discard("")

    stats = CleanStats()
    warnings: list[str] = []

    cleaned_pages: list[CleanedPage] = []
    page_index: dict[int, CleanedPage] = {}
    for page in extract.pages:
        item = CleanedPage(book=page.book, pdf_page=page.pdf_page, print_page=page.print_page)
        if page.pdf_page < extract.body_start_page:
            item.blank = True
            item.dropped_lines = len(page.lines)
            stats.dropped_pages.append(page.pdf_page)
            stats.dropped_lines += len(page.lines)
            stats.hit("C1", pages=1, lines=len(page.lines))
        cleaned_pages.append(item)
        page_index[page.pdf_page] = item

    chapter_of_page: dict[int, int] = {}
    for idx, chapter in enumerate(extract.chapters):
        for pno in range(chapter.page_start, chapter.page_end + 1):
            chapter_of_page.setdefault(pno, idx)

    seen_units: dict[str, int] = {}
    reference_mode = False
    current_chapter = -2
    last_body_unit: CleanUnit | None = None
    caption_gap = False

    pending_body: list[TextLine] = []
    pending_caption: list[TextLine] = []
    pending_code: list[TextLine] = []

    def _target(lines: Sequence[TextLine]) -> CleanedPage:
        return page_index[lines[0].page]

    def _flush_code() -> None:
        nonlocal pending_code
        if not pending_code:
            return
        text, stripped_numbers = _render_code(pending_code)
        if text.strip():
            _target(pending_code).units.append(
                CleanUnit(
                    text=text,
                    kind=UnitKind.CODE,
                    page_start=pending_code[0].page,
                    page_end=pending_code[-1].page,
                    y0=pending_code[0].y0,
                    y1=pending_code[-1].y1,
                    size=pending_code[0].size,
                    line_count=len(pending_code),
                )
            )
            stats.code_blocks += 1
            if stripped_numbers:
                stats.hit("C4", stripped_line_numbers=stripped_numbers)
        pending_code = []

    def _flush_caption() -> None:
        nonlocal pending_caption
        if not pending_caption:
            return
        first = pending_caption[0]
        text = first.text.strip()
        for extra in pending_caption[1:]:
            text, _ = join_wrapped(text, extra.text.strip())
        _target(pending_caption).units.append(
            CleanUnit(
                text=text,
                kind=UnitKind.CAPTION,
                page_start=first.page,
                page_end=pending_caption[-1].page,
                y0=first.y0,
                y1=pending_caption[-1].y1,
                size=first.size,
                line_count=len(pending_caption),
            )
        )
        stats.hit("C8", captions=1)
        pending_caption = []

    def _flush_body() -> None:
        nonlocal pending_body, reference_mode, last_body_unit
        if not pending_body:
            return
        paras, merges, hyphen_fixes = _merge_paragraphs(
            pending_body, right_margin=geometry.right_margin
        )
        if merges:
            stats.hit("C5", merged_line_joins=merges)
        if hyphen_fixes:
            stats.hit("C6", hyphen_fixes=hyphen_fixes)
        for para in paras:
            text = _strip_soft_hyphen(para.text)
            if not text:
                continue
            kind = _finalize_kind(text)
            if reference_mode and kind in (UnitKind.BODY, UnitKind.LIST):
                kind = UnitKind.REFERENCE
            unit = CleanUnit(
                text=text,
                kind=kind,
                page_start=para.page_start,
                page_end=para.page_end,
                y0=para.y0,
                y1=para.y1,
                size=para.size,
                line_count=para.line_count,
            )
            if _maybe_dup(unit, seen_units):
                stats.hit("C10", duplicate_units=1)
                continue
            if kind == UnitKind.REFERENCE:
                stats.hit("C9", reference_units=1)
            # 单元归属"段落起始页"，跨页段落不会被错记到后一页
            page_index[para.page_start].units.append(unit)
            last_body_unit = unit
            if kind in (UnitKind.BODY, UnitKind.LIST):
                stats.body_units += 1
        pending_body = []

    def _flush_all() -> None:
        _flush_code()
        _flush_caption()
        _flush_body()

    for page in extract.pages:
        if page.pdf_page < extract.body_start_page:
            continue

        chapter_idx = chapter_of_page.get(page.pdf_page, -1)
        if chapter_idx != current_chapter:
            _flush_all()
            current_chapter = chapter_idx
            reference_mode = False
            caption_gap = False

        tags, hits = _classify_lines(
            page,
            geometry=geometry,
            body_size=body_size,
            heading_norms=heading_norms,
        )
        for rule, n in hits.items():
            stats.hit(rule, lines=n)
        stats.dropped_lines += hits.get("C2", 0) + hits.get("C3", 0)

        for tag in tags:
            line = tag.line
            if tag.kind == "dropped":
                continue
            if tag.kind == "code":
                if pending_body or pending_caption:
                    _flush_caption()
                    _flush_body()
                caption_gap = False
                pending_code.append(line)
                continue
            if pending_code:
                _flush_code()
            if tag.kind == "heading":
                _flush_caption()
                _flush_body()
                caption_gap = False
                text = line.text.strip()
                target = page_index[line.page]
                target.units.append(
                    CleanUnit(
                        text=text,
                        kind=UnitKind.HEADING,
                        page_start=line.page,
                        page_end=line.page,
                        y0=line.y0,
                        y1=line.y1,
                        size=line.size,
                    )
                )
                if is_reference_title(text):
                    reference_mode = True
                continue
            if tag.kind == "caption":
                _flush_body()
                pending_caption.append(line)
                caption_gap = True
                continue
            # --- 正文行 ---
            # 图注换行续行：上一行是满行，则继续并入同一图注
            if pending_caption and _is_caption_continuation(
                pending_caption[-1], line, geometry.right_margin
            ):
                pending_caption.append(line)
                continue
            if pending_caption:
                _flush_caption()
            # 图注把段落切开（图在段中）时，把后续正文接回上一段
            if (
                caption_gap
                and not pending_body
                and last_body_unit is not None
                and _continues_paragraph(last_body_unit, line)
            ):
                joined, fixed = join_wrapped(last_body_unit.text, line.text.strip())
                last_body_unit.text = _strip_soft_hyphen(joined)
                last_body_unit.page_end = max(last_body_unit.page_end, line.page)
                last_body_unit.y1 = line.y1
                last_body_unit.line_count += 1
                if fixed:
                    stats.hit("C6", hyphen_fixes=1)
                stats.hit("C5", merged_line_joins=1)
                caption_gap = False
                continue
            caption_gap = False
            pending_body.append(line)

    _flush_all()

    stats.kept_units = sum(len(p.units) for p in cleaned_pages)
    blank_pages = [p.pdf_page for p in cleaned_pages if not p.units and not p.blank]
    if blank_pages:
        warnings.append(f"清洗后仍有 {len(blank_pages)} 个空页（前 5 个：{blank_pages[:5]}）")
    if stats.rules.get("C2", {}).get("lines", 0) == 0:
        warnings.append("未命中任何页眉行，请检查 header_cut 估计")
    if stats.rules.get("C3", {}).get("lines", 0) == 0:
        warnings.append("未命中任何页脚页码，请检查页脚判据")

    return CleanResult(
        book=extract.book,
        book_title=extract.book_title,
        pages=cleaned_pages,
        stats=stats,
        body_start_page=extract.body_start_page,
        body_font_size=body_size,
        left_margin=geometry.left_margin,
        right_margin=geometry.right_margin,
        warnings=warnings,
    )


def _strip_soft_hyphen(text: str) -> str:
    return text.replace(_SOFT_HYPHEN, "") if _SOFT_HYPHEN in text else text


def _maybe_dup(unit: CleanUnit, seen: dict[str, int]) -> bool:
    """C10：跨页重复段落去重（仅对相邻页的同一文本）。"""
    norm = normalize_for_search(unit.text)
    if len(norm) < 30:
        return False
    key = hashlib.sha1(norm.encode("utf-8")).hexdigest()
    last_page = seen.get(key)
    if last_page is not None and unit.page_start == last_page + 1:
        return True
    seen[key] = unit.page_start
    return False


__all__ = [
    "CleanResult",
    "CleanStats",
    "CleanUnit",
    "CleanedPage",
    "PageGeometry",
    "UnitKind",
    "clean_book",
    "estimate_geometry",
    "is_reference_title",
    "join_wrapped",
    "looks_like_caption",
    "looks_like_list_item",
    "looks_like_page_number",
    "normalize_for_search",
    "sha256_hex",
]
