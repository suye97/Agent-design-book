# -*- coding: utf-8 -*-
"""结构化切片：清洗后的文本单元 → 契约 Chunk（app/models.py）。

策略（docs/DESIGN.md 5.2）：
    1. 章由大纲 level-1 决定，节由 level>=2 决定（deep 有 x.y 编号，design 无编号）；
    2. 段落在"章内"连续编号（para_index，1 基），跨页段落归属起始页；
    3. 目标 450–700 中文字符、相邻块重叠 80–120 字符，优先在段落边界切；
    4. 代码块整体成块（>120 行才按空行切），不与正文混在同一个 chunk；
    5. chunk_id 稳定：sha1(book|chapter_no|chapter_title|chunk_index|content_hash)[:32]；
       块间用 prev_id / next_id 串起来，供检索侧邻接扩展。

对外主入口：
    chunk_book(book_extract, clean_result, options=None) -> list[Chunk]
    chunk_stats(chunks) -> dict
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Sequence

from app.ingest.clean import CleanResult, CleanUnit, UnitKind, normalize_for_search
from app.ingest.extract import BookExtract, Chapter, cjk_count
from app.logging_conf import get_logger
from app.models import Chunk, ChunkType

logger = get_logger("ingest.chunk")

#: 目标块大小（中文字符）
MIN_CHARS = 450
MAX_CHARS = 700
#: 相邻块重叠
MIN_OVERLAP = 80
MAX_OVERLAP = 120
#: 单个代码块允许超过目标大小的上限（整体不碎切；超过 120 行才切）
CODE_SPLIT_LINES = 120

_SENTENCE_END = set("。！？；：….!?;:")
_TERMINAL_PUNCT = set("。．！？；：….!?;:）)】》\"'”’")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


# --------------------------------------------------------------------------- #
# 选项与统计
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ChunkOptions:
    min_chars: int = MIN_CHARS
    max_chars: int = MAX_CHARS
    min_overlap: int = MIN_OVERLAP
    max_overlap: int = MAX_OVERLAP
    code_split_lines: int = CODE_SPLIT_LINES
    overlap: bool = True


@dataclass(slots=True)
class ChunkStats:
    chunks: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    below_min: int = 0
    above_max: int = 0
    code_above_max: int = 0
    overlap_chunks: int = 0
    avg_chars: float = 0.0
    avg_token_len: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "chunks": self.chunks,
            "by_type": self.by_type,
            "below_min": self.below_min,
            "above_max": self.above_max,
            "code_above_max": self.code_above_max,
            "overlap_chunks": self.overlap_chunks,
            "avg_chars": round(self.avg_chars, 1),
            "avg_token_len": round(self.avg_token_len, 1),
            "warnings": self.warnings,
        }


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 字符 1:1，拉丁/数字词各 +1。"""
    if not text:
        return 0
    return cjk_count(text) + len(_WORD_RE.findall(text))


# --------------------------------------------------------------------------- #
# 内部结构
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class _AttributedUnit:
    unit: CleanUnit
    chapter: Chapter
    chapter_index: int
    section_no: str
    section_title: str
    para_index: int

    @property
    def kind(self) -> UnitKind:
        return self.unit.kind

    @property
    def text(self) -> str:
        return self.unit.text


@dataclass(slots=True)
class _RawChunk:
    units: list[_AttributedUnit]
    force_kind: UnitKind | None = None

    def text(self, sep: str = "\n") -> str:
        return sep.join(u.text for u in self.units if u.text.strip())

    @property
    def kind(self) -> UnitKind:
        if self.force_kind is not None:
            return self.force_kind
        return _dominant_kind([u.kind for u in self.units])

    @property
    def is_code_only(self) -> bool:
        return all(u.kind == UnitKind.CODE for u in self.units)


def _dominant_kind(kinds: Sequence[UnitKind]) -> UnitKind:
    if not kinds:
        return UnitKind.BODY
    if all(k == UnitKind.CODE for k in kinds):
        return UnitKind.CODE
    if any(k == UnitKind.REFERENCE for k in kinds) and all(
        k in (UnitKind.REFERENCE, UnitKind.CAPTION) for k in kinds
    ):
        return UnitKind.REFERENCE
    counts: dict[UnitKind, int] = {}
    for k in kinds:
        if k == UnitKind.HEADING:
            continue
        counts[k] = counts.get(k, 0) + 1
    content_units = sum(counts.values())
    if content_units and counts.get(UnitKind.LIST, 0) / content_units >= 0.6:
        return UnitKind.LIST
    return UnitKind.BODY


_TO_CHUNK_TYPE = {
    UnitKind.BODY: ChunkType.TEXT,
    UnitKind.LIST: ChunkType.LIST,
    UnitKind.CODE: ChunkType.CODE,
    UnitKind.CAPTION: ChunkType.CAPTION,
    UnitKind.REFERENCE: ChunkType.REFERENCES,
}


# --------------------------------------------------------------------------- #
# 章/节归属 + 段号
# --------------------------------------------------------------------------- #
def attribute_units(extract: BookExtract, clean_result: CleanResult) -> list[_AttributedUnit]:
    """给每个清洗单元标注 章 / 节 / 段号（段号在章内连续）。"""
    chapters = sorted(extract.chapters, key=lambda c: c.page_start)
    page_to_chapter: dict[int, int] = {}
    for idx, chapter in enumerate(chapters):
        for pno in range(chapter.page_start, chapter.page_end + 1):
            page_to_chapter.setdefault(pno, idx)

    # 每章内的节锚点（按 页, y 排序）
    anchors: dict[int, list[tuple[int, float, str, str]]] = {}
    for idx, chapter in enumerate(chapters):
        items: list[tuple[int, float, str, str]] = [
            (chapter.page_start, 0.0, "", "")
        ]
        for section in chapter.sections:
            items.append((section.page, section.y, section.no, section.title))
        items.sort(key=lambda it: (it[0], it[1]))
        anchors[idx] = items

    counters: dict[int, int] = {}
    out: list[_AttributedUnit] = []
    for page in clean_result.pages:
        for unit in page.units:
            if unit.kind == UnitKind.HEADING:
                continue
            chapter_idx = page_to_chapter.get(unit.page_start)
            if chapter_idx is None:
                # 尝试向前找最近的章（跨页段落/无章页兜底）
                for pno in range(unit.page_start - 1, 0, -1):
                    if pno in page_to_chapter:
                        chapter_idx = page_to_chapter[pno]
                        break
            if chapter_idx is None:
                continue
            chapter = chapters[chapter_idx]
            section_no, section_title = "", ""
            for page_no, y, no, title in anchors[chapter_idx]:
                if page_no < unit.page_start or (
                    page_no == unit.page_start and y <= unit.y0 + 1.0
                ):
                    section_no, section_title = no, title
                else:
                    break
            counters[chapter_idx] = counters.get(chapter_idx, 0) + 1
            out.append(
                _AttributedUnit(
                    unit=unit,
                    chapter=chapter,
                    chapter_index=chapter_idx,
                    section_no=section_no,
                    section_title=section_title,
                    para_index=counters[chapter_idx],
                )
            )
    return out


# --------------------------------------------------------------------------- #
# 文本切分
# --------------------------------------------------------------------------- #
def _split_to_fill(text: str, need: int, room: int) -> tuple[str, str]:
    """把一段切出"够填当前块"的前半部分与剩余部分（尽量在句末标点处切）。"""
    text = text.strip()
    if need <= 0 or room <= 0 or len(text) <= need + 20:
        return "", text
    upper = min(room, len(text) - 20)
    if upper <= need:
        return "", text
    cut = -1
    for idx in range(need, upper):
        if text[idx - 1] in _SENTENCE_END:
            cut = idx
            break
    if cut <= 0:
        cut = upper
    head, tail = text[:cut].strip(), text[cut:].strip()
    if len(head) < need or not tail:
        return "", text
    return head, tail


def split_text(text: str, max_chars: int = MAX_CHARS, min_chars: int = MIN_CHARS) -> list[str]:
    """按句末标点把超长文本切成不超过 max_chars 的片段（尽量不小于 min_chars）。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    sentences: list[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _SENTENCE_END or ch == "\n":
            sentences.append(text[start : i + 1])
            start = i + 1
    if start < len(text):
        sentences.append(text[start:])
    pieces: list[str] = []
    cur = ""
    for sent in sentences:
        if not sent:
            continue
        if cur and len(cur) + len(sent) > max_chars:
            pieces.append(cur)
            cur = sent
        else:
            cur += sent
        # 单句本身超长：硬切
        while len(cur) > max_chars:
            pieces.append(cur[:max_chars])
            cur = cur[max_chars:]
    if cur:
        pieces.append(cur)
    # 合并过短尾巴
    if (
        len(pieces) >= 2
        and len(pieces[-1]) < min_chars // 2
        and len(pieces[-2]) + len(pieces[-1]) <= max_chars
    ):
        tail = pieces.pop()
        pieces[-1] += tail
    return [p for p in pieces if p.strip()]


def _split_code_unit(unit: CleanUnit, max_lines: int) -> list[CleanUnit]:
    """代码块超过 max_lines 行时，按空行切成若干子块（保持代码整体性）。"""
    lines = unit.text.split("\n")
    if len(lines) <= max_lines:
        return [unit]
    parts: list[list[str]] = []
    cur: list[str] = []
    for line in lines:
        cur.append(line)
        if not line.strip() and len(cur) >= max_lines // 3:
            parts.append(cur)
            cur = []
        elif len(cur) >= max_lines:
            parts.append(cur)
            cur = []
    if cur:
        parts.append(cur)
    if len(parts) <= 1:
        return [unit]
    total = max(1, len(lines))
    out: list[CleanUnit] = []
    for part in parts:
        while part and not part[-1].strip():
            part.pop()
        while part and not part[0].strip():
            part.pop(0)
        if not part:
            continue
        text = "\n".join(part)
        out.append(
            CleanUnit(
                text=text,
                kind=UnitKind.CODE,
                page_start=unit.page_start,
                page_end=unit.page_end,
                y0=unit.y0,
                y1=unit.y1,
                size=unit.size,
                line_count=len(part),
            )
        )
    del total
    return out or [unit]


# --------------------------------------------------------------------------- #
# 拼块
# --------------------------------------------------------------------------- #
def _assemble_chapter(units: Sequence[_AttributedUnit], options: ChunkOptions) -> list[_RawChunk]:
    """把一章的单元拼成原始块（尚未加重叠、尚未生成 id）。"""
    # 拼块预算扣掉重叠长度，保证 "正文 + 重叠" 后仍落在目标区间内
    pack_max = max(options.min_chars, options.max_chars - options.max_overlap)
    chunks: list[_RawChunk] = []
    cur: list[_AttributedUnit] = []
    cur_len = 0
    pending_caption: list[_AttributedUnit] = []

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            chunks.append(_RawChunk(units=cur))
            cur = []
            cur_len = 0


    def attach_captions() -> None:
        nonlocal cur, cur_len
        if not pending_caption:
            return
        cap_len = sum(len(c.text) + 1 for c in pending_caption)
        if cap_len > pack_max:
            # 超大图注（图内文字被并成一块）：单独成块并按句切
            flush()
            cap_text = "\n".join(c.text for c in pending_caption)
            for piece in split_text(cap_text, options.max_chars, options.min_chars):
                chunks.append(
                    _RawChunk(
                        units=[_clone_unit(pending_caption[0], _replace_text(pending_caption[0].unit, piece))],
                        force_kind=UnitKind.CAPTION,
                    )
                )
            pending_caption.clear()
            return
        if cur and cur_len + cap_len <= pack_max:
            cur.extend(pending_caption)
            cur_len += cap_len
        elif cur:
            # 先把当前块收尾，图注再并入"紧邻的上一块"，保证段号单调
            flush()
            chunks[-1].units.extend(pending_caption)
        elif chunks and chunks[-1].units and chunks[-1].kind != UnitKind.CODE:
            chunks[-1].units.extend(pending_caption)
        else:
            cur.extend(pending_caption)
            cur_len += cap_len
        pending_caption.clear()

    for unit in units:
        if unit.kind == UnitKind.CAPTION:
            pending_caption.append(unit)
            continue
        if unit.kind == UnitKind.CODE:
            attach_captions()
            parts = _split_code_unit(unit.unit, options.code_split_lines)
            # Keep prose and code as separately typed chunks; prev/next links retain context.
            flush()
            for part in parts:
                chunks.append(
                    _RawChunk(units=[_clone_unit(unit, part)], force_kind=UnitKind.CODE)
                )
            continue
        attach_captions()
        size = len(unit.text) + 1
        if cur and cur_len + size > pack_max and cur_len >= options.min_chars:
            flush()
        if cur and cur_len + size > pack_max and cur_len < options.min_chars:
            # 当前块偏短：把这段按句切开，把当前块填到目标大小，剩余部分顺延
            head, tail = _split_to_fill(
                unit.text,
                need=options.min_chars - cur_len,
                room=pack_max - cur_len,
            )
            if head and tail:
                cur.append(_clone_unit(unit, _replace_text(unit.unit, head)))
                flush()
                unit = _clone_unit(unit, _replace_text(unit.unit, tail))
                size = len(unit.text) + 1
        if size > pack_max:
            # 超长单段按句切，段号保持不变
            if cur:
                flush()
            for piece in split_text(unit.text, pack_max, options.min_chars):
                echoed = _clone_unit(unit, _replace_text(unit.unit, piece))
                chunks.append(_RawChunk(units=[echoed]))
            continue
        cur.append(unit)
        cur_len += size
    attach_captions()
    flush()

    # 合并相邻的小代码块（同一节、合计不超过目标上限），避免图内代码碎片过多
    merged: list[_RawChunk] = []
    for rc in chunks:
        if (
            merged
            and rc.is_code_only
            and merged[-1].is_code_only
            and len(merged[-1].text()) + len(rc.text()) + 1 <= options.max_chars
        ):
            merged[-1].units.extend(rc.units)
        else:
            merged.append(rc)
    return merged


def _replace_text(unit: CleanUnit, text: str) -> CleanUnit:
    return CleanUnit(
        text=text,
        kind=unit.kind,
        page_start=unit.page_start,
        page_end=unit.page_end,
        y0=unit.y0,
        y1=unit.y1,
        size=unit.size,
        line_count=unit.line_count,
    )


def _clone_unit(attr: _AttributedUnit, unit: CleanUnit) -> _AttributedUnit:
    return _AttributedUnit(
        unit=unit,
        chapter=attr.chapter,
        chapter_index=attr.chapter_index,
        section_no=attr.section_no,
        section_title=attr.section_title,
        para_index=attr.para_index,
    )


# --------------------------------------------------------------------------- #
# 重叠
# --------------------------------------------------------------------------- #
def _trim_tail(text: str, min_len: int, max_len: int) -> tuple[str, int]:
    """从文本尾部截取 [min_len, max_len] 的重叠片段，尽量从句子边界开始。

    Returns: (片段, 片段在原文中的起始偏移)
    """
    text = text.rstrip()
    if not text:
        return "", 0
    if len(text) <= min_len:
        return text, 0
    start = max(0, len(text) - max_len)
    window = text[start:]
    # 在窗口前 1/3 内找句末标点，让重叠从完整句子开始
    limit = max(1, len(window) - min_len)
    cut = -1
    for idx in range(min(limit, len(window) - 1)):
        if window[idx] in _SENTENCE_END:
            cut = idx + 1
            break
    if cut <= 0:
        # 退化为空格边界
        space = window.find(" ", 0, limit)
        cut = space + 1 if space >= 0 else 0
    piece = window[cut:]
    if len(piece) < min_len:
        piece = window
        cut = 0
    return piece.strip(), start + cut


def _norm_page(text: str) -> str:
    """页文本归一化（去空白/统一连字符与全角），用于把 overlap 片段定位到实际页。"""
    return re.sub(r"\s+", "", normalize_for_search(text))


def _overlap_page_start(
    anchor: _AttributedUnit,
    offset_in_anchor: int,
    prev_page_end: int,
    overlap_text: str = "",
    page_texts: dict[int, str] | None = None,
) -> int:
    """确定 overlap 片段实际所在页。

    优先用已解析的页文本做校验（overlap 是原文，必然出现在某一页上）；
    找不到时按 anchor 单元跨页比例插值，最后用上一块 page_end 兜底。
    这样避免把整块 page_start 错误地前后移动一页。
    """
    unit = anchor.unit
    upper = min(unit.page_end, prev_page_end or unit.page_end)
    key = _norm_page(overlap_text)[:16] if overlap_text else ""
    if key and page_texts:
        for page in range(unit.page_start, max(unit.page_start, upper) + 1):
            if key in page_texts.get(page, ""):
                return page

    span = unit.page_end - unit.page_start
    if span <= 0:
        page = unit.page_start
    else:
        frac = offset_in_anchor / max(1, len(unit.text))
        page = unit.page_start + int(round(frac * span))
    page = max(unit.page_start, min(page, unit.page_end))
    if prev_page_end:
        page = min(page, prev_page_end)
    return page


def _tail_overlap_from_units(
    prev_units: Sequence[_AttributedUnit],
    options: ChunkOptions,
) -> tuple[str, _AttributedUnit | None, int]:
    """取上一块尾部 80–120 字符作为重叠，返回 (片段, 起点单元, 单元内偏移)。

    只在"上一块自身的单元文本"上取尾部（上一块 content 开头的 overlap 前缀不影响
    尾部），这样字符偏移与 prev_units 严格对齐；重叠可能从上一块中部的某个段落开始
    （上一块尾部可能挂着图注），必须按偏移定位单元，page/para 元数据才与实际起点一致。
    """
    own_text = "\n".join(u.text for u in prev_units if u.text.strip())
    piece, offset = _trim_tail(own_text, options.min_overlap, options.max_overlap)
    if not piece:
        return "", None, 0
    acc = 0
    for unit in prev_units:
        if not unit.text.strip():
            continue
        if acc + len(unit.text) > offset:
            return piece, unit, offset - acc
        acc += len(unit.text) + 1
    last = prev_units[-1] if prev_units else None
    return piece, last, 0


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def chunk_book(
    extract: BookExtract,
    clean_result: CleanResult,
    options: ChunkOptions | None = None,
) -> list[Chunk]:
    """把一本书切成 Chunk 列表（按阅读顺序，prev_id/next_id 串联）。"""
    opts = options or ChunkOptions()
    attributed = attribute_units(extract, clean_result)
    if not attributed:
        return []

    # 按章分组（保持章序）
    by_chapter: dict[int, list[_AttributedUnit]] = {}
    for unit in attributed:
        by_chapter.setdefault(unit.chapter_index, []).append(unit)

    # overlap 页码校验：用解析出的页文本（原文）确认片段实际所在页
    page_texts: dict[int, str] = {}
    if opts.overlap:
        page_texts = {page.pdf_page: _norm_page(page.text) for page in extract.pages}

    chunks: list[Chunk] = []
    for chapter_idx in sorted(by_chapter):
        units = by_chapter[chapter_idx]
        chapter = units[0].chapter
        raw_chunks = _assemble_chapter(units, opts)

        prev_units: list[_AttributedUnit] = []
        prev_page_end = 0
        prev_kind: UnitKind | None = None
        for idx, raw in enumerate(raw_chunks, start=1):
            own_units = raw.units
            own_text = raw.text()
            if not own_text.strip():
                continue
            overlap_text = ""
            page_start = own_units[0].unit.page_start
            anchor = own_units[0]
            if (
                opts.overlap
                and idx > 1
                and prev_units
                and prev_kind != UnitKind.CODE
                and raw.kind != UnitKind.CODE
            ):
                overlap_text, anchor_prev, off_in_anchor = _tail_overlap_from_units(
                    prev_units, opts
                )
                if overlap_text:
                    anchor = anchor_prev or anchor
                    # overlap 是上一块尾部，可能来自上一页；页码按 overlap 实际起点页修正
                    page_start = _overlap_page_start(
                        anchor, off_in_anchor, prev_page_end, overlap_text, page_texts
                    )
            content = f"{overlap_text}\n{own_text}" if overlap_text else own_text
            kind = raw.kind
            section = own_units[0]
            page_end = max(u.unit.page_end for u in own_units)
            # 代码块可能从页面底部开始（行被拆到下一页），按页文本校正起始页
            if kind == UnitKind.CODE and page_texts:
                key = _norm_page(content)[:16]
                if key:
                    for page in range(page_start, page_end + 1):
                        if key in page_texts.get(page, ""):
                            page_start = page
                            break
            chunk = Chunk(
                chunk_id="",  # 先占位，后面统一计算
                book=extract.book,
                book_title=extract.book_title,
                chapter_no=chapter.chapter_no,
                chapter_title=chapter.title,
                section_no=section.section_no,
                section_title=section.section_title,
                page_start=page_start,
                page_end=page_end,
                page_print_start=_print_page(extract, page_start),
                page_print_end=_print_page(extract, page_end),
                para_start=anchor.para_index,
                para_end=own_units[-1].para_index,
                chunk_type=_TO_CHUNK_TYPE.get(kind, ChunkType.TEXT),
                content=content,
                content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest()[:32],
                token_len=estimate_tokens(content),
                prev_id=None,
                next_id=None,
            )
            chunk.chunk_id = _chunk_id(chunk, idx)
            chunks.append(chunk)
            prev_units = list(own_units)
            prev_page_end = chunk.page_end
            prev_kind = kind

    # prev_id / next_id 串联
    for i, chunk in enumerate(chunks):
        chunk.prev_id = chunks[i - 1].chunk_id if i > 0 else None
        chunk.next_id = chunks[i + 1].chunk_id if i + 1 < len(chunks) else None
    return chunks


def _chunk_id(chunk: Chunk, index: int) -> str:
    raw = (
        f"{chunk.book}|{chunk.chapter_no}|{chunk.chapter_title}|{index}|"
        f"{chunk.content_hash}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]


def _print_page(extract: BookExtract, pdf_page: int) -> int:
    value = pdf_page - extract.print_offset
    return value if value >= 1 else pdf_page


def chunk_stats(chunks: Sequence[Chunk], options: ChunkOptions | None = None) -> ChunkStats:
    opts = options or ChunkOptions()
    stats = ChunkStats(chunks=len(chunks))
    total_chars = 0
    total_tokens = 0
    for chunk in chunks:
        stats.by_type[chunk.chunk_type.value] = (
            stats.by_type.get(chunk.chunk_type.value, 0) + 1
        )
        length = len(chunk.content)
        total_chars += length
        total_tokens += chunk.token_len
        if length < opts.min_chars:
            stats.below_min += 1
        if length > opts.max_chars:
            if chunk.chunk_type == ChunkType.CODE:
                stats.code_above_max += 1
            else:
                stats.above_max += 1
        if chunk.prev_id:
            stats.overlap_chunks += 1
    if chunks:
        stats.avg_chars = total_chars / len(chunks)
        stats.avg_token_len = total_tokens / len(chunks)
    if stats.below_min:
        stats.warnings.append(
            f"{stats.below_min} 个块小于 {opts.min_chars} 字符（章末/图注/短章）"
        )
    if stats.above_max:
        stats.warnings.append(f"{stats.above_max} 个正文块大于 {opts.max_chars} 字符")
    return stats


__all__ = [
    "ChunkOptions",
    "ChunkStats",
    "attribute_units",
    "chunk_book",
    "chunk_stats",
    "estimate_tokens",
    "split_text",
]
