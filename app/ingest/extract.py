# -*- coding: utf-8 -*-
"""PDF 解析：页级文本行、书签大纲、印刷页码偏移、章节结构。

只做"读出原文 + 定位结构"，不做任何改写（清洗在 clean.py）。
两本书均为 LaTeX 生成的文字版 PDF，PyMuPDF 可直接拿到行级 bbox / 字体 / 字号。

对外主入口：
    extract_book(book) -> BookExtract      # "design" / "deep"
    extract_all() -> dict[str, BookExtract]
    to_doc_pages(book_extract) -> list[DocPage]

除 DocPage 外，额外保留 TextLine（bbox / 字号 / 字体），供 clean.py 判定
页眉页脚、代码块、图注，以及 chunk.py 做章节归属与段落重建。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pymupdf

from app.config import settings
from app.logging_conf import get_logger
from app.models import BOOK_TITLES, DocPage

logger = get_logger("ingest.extract")

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
#: 正文字号候选区间（用于识别"主字号"）
_BODY_SIZE_RANGE = (6.0, 16.0)
#: 同一视觉行内 y 中心容差（pt）。LaTeX 行距 ≥ 15pt，6pt 足够安全。
_ROW_TOL = 6.0
#: 判断"标题行"的字号倍数（相对正文字号）
_HEADING_SIZE_RATIO = 1.15
#: 页眉/页脚区域相对页高的默认比例（仅兜底，优先用分布检测出的 body_top）
_HEADER_RATIO = 0.09
_FOOTER_RATIO = 0.88

#: 目录/封面类大纲标题（不参与正文起点判定）
_FRONT_MATTER_TITLES = {
    "目录",
    "封面",
    "版权",
    "扉页",
    "献辞",
    "contents",
    "tableofcontents",
}
#: 参考文献类小节的归一化标题（clean.py 用来标记 chunk_type=references）
REFERENCE_TITLES = {
    "参考文献",
    "参考资料",
    "参考文献与延伸阅读",
    "延伸阅读",
    "references",
}

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
}

#: PDF 页码 → 印刷页码 偏移（模块级缓存，便于测试替换）
_PRINT_OFFSET_HINTS: dict[str, int] = {"design": 13, "deep": 8}


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TextLine:
    """PDF 中一个视觉行（PyMuPDF dict 模式的 line）。"""

    text: str
    page: int                 # PDF 页码（1 基）
    x0: float
    y0: float
    x1: float
    y1: float
    size: float               # 主字号（按字符数加权的众数）
    fonts: dict[str, int] = field(default_factory=dict)
    sizes: dict[float, int] = field(default_factory=dict)  # 字号 → 字符数

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def mono_ratio(self) -> float:
        """等宽字体字符占比，用于区分代码行。"""
        total = sum(self.fonts.values())
        if not total:
            return 0.0
        mono = sum(n for f, n in self.fonts.items() if _is_mono_font(f))
        return mono / total

    @property
    def is_code_like(self) -> bool:
        # 正文行里也会夹着行内代码（等宽字体占比 30%~60%），阈值取 0.85 只认整行代码
        return self.mono_ratio() >= 0.85


@dataclass(slots=True)
class OutlineEntry:
    """书签大纲条目（归一化后的层级与解析出的编号）。"""

    level: int
    title: str
    page: int
    chapter_no: int = 0
    section_no: str = ""

    @property
    def kind(self) -> str:
        return "chapter" if self.level == 1 else "section"


@dataclass(slots=True)
class Section:
    """节：来自 level>=2 的大纲条目。"""

    no: str
    title: str
    page: int
    level: int
    y: float = 0.0            # 标题在页面上的纵坐标（0 = 未定位，按页首处理）
    heading_found: bool = False
    anchor_title: str = ""    # 大纲原始标题（用于定位标题行）

    def __post_init__(self) -> None:
        if not self.anchor_title:
            self.anchor_title = self.title


@dataclass(slots=True)
class Chapter:
    """章：来自 level==1 的大纲条目。"""

    chapter_no: int
    title: str
    page_start: int
    page_end: int
    anchor_title: str = ""
    heading_y: float = 0.0
    heading_found: bool = False
    sections: list[Section] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.anchor_title:
            self.anchor_title = self.title


@dataclass(slots=True)
class PageText:
    """单页原文（行级）。"""

    book: str
    pdf_page: int
    print_page: int
    lines: list[TextLine]
    width: float = 0.0
    height: float = 0.0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass(slots=True)
class BookExtract:
    """一本书的解析结果。"""

    book: str
    book_title: str
    pdf_path: str
    pdf_pages: int
    pages: list[PageText]
    outline: list[OutlineEntry]
    chapters: list[Chapter]
    print_offset: int
    print_offset_method: str
    body_start_page: int
    body_font_size: float = 12.0
    body_top: float = 0.0
    metadata_title: str = ""

    def page_map(self) -> dict[int, PageText]:
        return {p.pdf_page: p for p in self.pages}

    @property
    def body_pages(self) -> list[PageText]:
        return [p for p in self.pages if p.pdf_page >= self.body_start_page]


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
_MONO_FONT_MARKERS = (
    "mono",
    "menlo",
    "courier",
    "consolas",
    "dejavusansmono",
    "lmmono",
    "inconsolata",
    "sourcecodepro",
)


def _is_mono_font(font: str) -> bool:
    low = (font or "").lower()
    return any(marker in low for marker in _MONO_FONT_MARKERS)


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def cjk_count(text: str) -> int:
    return len(_CJK_RE.findall(text))


def normalize_title(text: str) -> str:
    """标题归一化：去空白、全角转半角、统一各种连字符与大小写。

    用于把大纲标题与页面上的标题行对齐（LaTeX 两处空格/连字符常不一致）。
    """
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:  # 全角 ASCII
            ch = chr(code - 0xFEE0)
        elif code == 0x3000:
            ch = " "
        if ch in "\u2010\u2011\u2012\u2013\u2014\u2212":
            ch = "-"
        if ch.isspace():
            continue
        out.append(ch.lower())
    return "".join(out)


_CN_NUM_RE = re.compile(r"^[零〇一二两三四五六七八九十百]+$")


def cn_to_int(text: str) -> int | None:
    """中文数字（一 ~ 九十九）转 int。"""
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if not _CN_NUM_RE.match(text):
        return None
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left) if left else 1
        ones = _CN_DIGITS.get(right) if right else 0
        if tens is None or ones is None or tens >= 10 or ones >= 10:
            return None
        return tens * 10 + ones
    total = 0
    for ch in text:
        val = _CN_DIGITS.get(ch)
        if val is None or val >= 10:
            return None
        total += val
    return total or None


_CHAPTER_RE = re.compile(r"^\s*第\s*([0-9]{1,4}|[零〇一二两三四五六七八九十百]{1,5})\s*章\s*[：:、.\-—]?\s*")
_SECTION_NO_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,3}){1,3})\s*[：:、.\-—]?\s*")
_APPENDIX_RE = re.compile(r"^\s*附\s*录\s*([A-Z])\s*[-—－]\s*")


def parse_chapter_no(title: str) -> int:
    """"第 12 章 xxx" → 12；非编号章节返回 0。"""
    m = _CHAPTER_RE.match(title or "")
    if not m:
        return 0
    return cn_to_int(m.group(1)) or 0


def strip_chapter_prefix(title: str) -> str:
    """去掉"第 N 章："前缀，保留纯章名（避免引用标签重复）。"""
    return _CHAPTER_RE.sub("", title or "").strip()


def parse_section_no(title: str) -> tuple[str, str]:
    """'1.2.3 xxx' → ('1.2.3', 'xxx')；无编号返回 ('', 原题)。"""
    m = _SECTION_NO_RE.match(title or "")
    if not m:
        return "", (title or "").strip()
    return m.group(1), (title or "")[m.end():].strip()


def _print_page(pdf_page: int, offset: int) -> int:
    return pdf_page - offset


# --------------------------------------------------------------------------- #
# 页级行提取
# --------------------------------------------------------------------------- #
def extract_page_lines(page: "pymupdf.Page", book: str, pno: int) -> list[TextLine]:
    """把一页的 dict 输出压平成视觉行，并按阅读顺序（行 y → 列 x）排序。"""
    raw: list[TextLine] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:  # 1 = 图片
            continue
        for line in block.get("lines", []):
            spans = line.get("spans") or []
            text = "".join(str(s.get("text") or "") for s in spans)
            if not text.strip():
                continue
            sizes: Counter[float] = Counter()
            fonts: Counter[str] = Counter()
            for span in spans:
                n = len(str(span.get("text") or ""))
                if not n:
                    continue
                sizes[round(float(span.get("size") or 0.0), 1)] += n
                fonts[str(span.get("font") or "")] += n
            size = sizes.most_common(1)[0][0] if sizes else 0.0
            x0, y0, x1, y1 = (float(v) for v in line["bbox"])
            raw.append(
                TextLine(
                    text=text.strip("\n"),
                    page=pno,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    size=size,
                    fonts=dict(fonts),
                    sizes=dict(sizes),
                )
            )
    return _reading_order(raw)


def _row_joiner(prev: TextLine, cur: TextLine) -> str:
    """同一视觉行内两个碎片之间的连接符（空格 / 无）。"""
    left = prev.text.rstrip()
    right = cur.text.lstrip()
    if not left or not right:
        return ""
    if left[-1].isspace() or right[0].isspace():
        return ""
    if is_cjk(left[-1]) or is_cjk(right[0]):
        return ""
    gap = cur.x0 - prev.x1
    return " " if gap > 1.0 else ""


def _merge_row(row: Sequence[TextLine]) -> TextLine:
    """把同一视觉行内被拆开的碎片（行内代码/表格单元格）合回一行。"""
    if len(row) == 1:
        return row[0]
    ordered = sorted(row, key=lambda ln: ln.x0)
    parts: list[str] = []
    fonts: Counter[str] = Counter()
    sizes: Counter[float] = Counter()
    for idx, piece in enumerate(ordered):
        if idx:
            parts.append(_row_joiner(ordered[idx - 1], piece))
        parts.append(piece.text)
        fonts.update(piece.fonts)
        sizes.update(piece.sizes)
    text = "".join(parts).strip()
    # 同一行内的行内代码/上标等碎片字号偏小；以最大字号代表该行字号（正文/标题）
    size = max(piece.size for piece in ordered)
    return TextLine(
        text=text,
        page=ordered[0].page,
        x0=min(p.x0 for p in ordered),
        y0=min(p.y0 for p in ordered),
        x1=max(p.x1 for p in ordered),
        y1=max(p.y1 for p in ordered),
        size=size,
        fonts=dict(fonts),
        sizes=dict(sizes),
    )


def _reading_order(lines: Sequence[TextLine]) -> list[TextLine]:
    """按"视觉行"排序并把同行碎片合并：同一 y 带内按 x，否则按 y。

    LaTeX 的同一物理行常被拆成多个 TextLine（行内代码、表格单元格、代码行号），
    这里按 y 带归并后横向拼接，恢复成"一整行"，段落合并才有意义。
    """
    if len(lines) <= 1:
        return list(lines)
    ordered = sorted(lines, key=lambda ln: (ln.center_y, ln.x0))
    rows: list[list[TextLine]] = []
    centers: list[float] = []
    for line in ordered:
        if rows and abs(line.center_y - centers[-1]) <= _ROW_TOL:
            rows[-1].append(line)
            centers[-1] = sum(ln.center_y for ln in rows[-1]) / len(rows[-1])
        else:
            rows.append([line])
            centers.append(line.center_y)
    return [_merge_row(row) for row in rows]


# --------------------------------------------------------------------------- #
# 页码偏移 / 正文起点
# --------------------------------------------------------------------------- #
_PAGE_NUM_RE = re.compile(r"^\s*(?:[-—–]\s*)?(?:\d{1,4}|[ivxlcdm]{1,8})(?:\s*[-—–])?\s*$", re.I)


def _looks_like_page_number(text: str) -> bool:
    return bool(_PAGE_NUM_RE.match(text or ""))


_LABEL_INT_RE = re.compile(r"^\s*(\d{1,4})\s*$")


def detect_print_offset(
    doc: "pymupdf.Document",
    pages: Sequence[PageText],
    body_start_page: int,
) -> tuple[int, str, float]:
    """自动检测"印刷页码 = PDF 页码 - offset"。

    首选 PDF 页标签（get_label 返回印刷页码）；设计模式书无标签，则用页眉/页脚
    中的数字投票。返回 (offset, method, confidence)。
    """
    # --- 方式一：PDF 页标签 ---
    votes: Counter[int] = Counter()
    numeric_labels = 0
    for i in range(doc.page_count):
        try:
            label = doc[i].get_label()
        except Exception:  # pragma: no cover - 旧版 pymupdf 无 get_label
            label = None
        if not label:
            continue
        m = _LABEL_INT_RE.match(str(label))
        if not m:
            continue
        pno = i + 1
        if pno < body_start_page:
            continue
        numeric_labels += 1
        votes[pno - int(m.group(1))] += 1
    if numeric_labels >= 20 and votes:
        offset, count = votes.most_common(1)[0]
        confidence = count / numeric_labels
        if confidence >= 0.6:
            return offset, "page-label", confidence

    # --- 方式二：页眉/页脚数字投票 ---
    total = 0
    for page in pages:
        if page.pdf_page < body_start_page:
            continue
        total += 1
        height = page.height or 842.0
        header_cut = height * _HEADER_RATIO
        footer_cut = height * _FOOTER_RATIO
        for line in page.lines:
            if not (line.y0 < header_cut or line.y0 > footer_cut):
                continue
            nums = re.findall(r"\d{1,4}", line.text)
            if not nums:
                continue
            # 设计模式书页眉形如 "16 第 1 章：提示链"：首个数字是页码
            cand = int(nums[0])
            if 1 <= cand <= doc.page_count:
                votes[page.pdf_page - cand] += 1
    if total and votes:
        offset, count = votes.most_common(1)[0]
        confidence = count / total
        if confidence >= 0.5:
            return offset, "running-head-number", min(confidence, 1.0)
    return 0, "none", 0.0


def detect_body_start(toc: Sequence[Sequence], pdf_pages: int) -> int:
    """正文起点：第一个非"目录/封面"类 level-1 大纲条目所在页。"""
    candidates: list[int] = []
    for level, title, page in toc:
        if int(level) != 1:
            continue
        if normalize_title(str(title)) in {normalize_title(t) for t in _FRONT_MATTER_TITLES}:
            continue
        page = int(page)
        if 1 <= page <= pdf_pages:
            candidates.append(page)
    if not candidates:
        return 1
    return max(1, min(candidates))


# --------------------------------------------------------------------------- #
# 章节结构
# --------------------------------------------------------------------------- #
def build_chapters(
    toc: Sequence[Sequence],
    body_start_page: int,
    pdf_pages: int,
) -> list[Chapter]:
    """把书签大纲整理成 章 → 节 两级结构。"""
    entries: list[tuple[int, str, int]] = []
    for level, title, page in toc:
        entries.append((int(level), str(title).strip(), int(page)))

    chapter_entries = [
        (lvl, title, page)
        for lvl, title, page in entries
        if lvl == 1
        and page >= body_start_page
        and page <= pdf_pages
        and normalize_title(title) not in {normalize_title(t) for t in _FRONT_MATTER_TITLES}
    ]
    if not chapter_entries:
        # 兜底：整本书作为一章
        return [
            Chapter(
                chapter_no=0,
                title="正文",
                page_start=body_start_page,
                page_end=pdf_pages,
                anchor_title="",
            )
        ]

    chapters: list[Chapter] = []
    for idx, (_, title, page) in enumerate(chapter_entries):
        next_page = (
            chapter_entries[idx + 1][2] - 1 if idx + 1 < len(chapter_entries) else pdf_pages
        )
        chapter_no = parse_chapter_no(title)
        chapter_title = strip_chapter_prefix(title) if chapter_no else title
        chapters.append(
            Chapter(
                chapter_no=chapter_no,
                title=chapter_title,
                page_start=page,
                page_end=max(page, next_page),
                anchor_title=title,
            )
        )

    # 章节归属：level>=2 的大纲条目
    for level, title, page in entries:
        if level < 2:
            continue
        sec_no, sec_title = parse_section_no(title)
        for chapter in chapters:
            if chapter.page_start <= page <= chapter.page_end:
                chapter.sections.append(
                    Section(
                        no=sec_no,
                        title=sec_title if sec_no else title,
                        page=page,
                        level=level,
                        anchor_title=title,
                    )
                )
                break
    for chapter in chapters:
        chapter.sections.sort(key=lambda s: (s.page, s.level))
    return chapters


def _find_heading_y(
    pages: dict[int, PageText],
    anchor_title: str,
    page_start: int,
    page_end: int,
    body_font_size: float,
) -> tuple[int, float, bool]:
    """在 [page_start, page_end] 内定位标题行，返回 (实际页码, y, 是否命中)。

    标题行需满足：字号 ≥ 正文字号 * 1.15，且归一化文本与大纲标题一致（或前缀一致）。
    大纲页码偶尔比标题实际所在页早一页（标题跨页），因此以"实际命中页"为准。
    """
    target = normalize_title(anchor_title)
    if not target:
        return page_start, 0.0, False
    min_size = body_font_size * _HEADING_SIZE_RATIO
    for pno in range(page_start, min(page_end, page_start + 2) + 1):
        page = pages.get(pno)
        if page is None:
            continue
        for line in page.lines:
            norm = normalize_title(line.text)
            if not norm:
                continue
            if line.size + 1e-6 < min_size:
                continue
            if norm == target or (len(norm) >= 8 and target.startswith(norm)):
                return pno, line.y0, True
            # 标题被排版拆成两行：取前 8 字前缀
            if len(norm) >= 8 and norm.startswith(target[:8]) and target[:8]:
                return pno, line.y0, True
    return page_start, 0.0, False


def locate_headings(chapters: Sequence[Chapter], pages: dict[int, PageText], body_font_size: float) -> None:
    """为每章/每节定位标题行 y（失败时 y=0，表示按页首处理）。"""
    for chapter in chapters:
        found_page, y, found = _find_heading_y(
            pages, chapter.anchor_title, chapter.page_start, chapter.page_end, body_font_size
        )
        chapter.heading_y, chapter.heading_found = y, found
        del found_page
        for section in chapter.sections:
            sy_page, sy, sfound = _find_heading_y(
                pages,
                section.anchor_title,
                section.page,
                min(chapter.page_end, section.page + 1),
                body_font_size,
            )
            section.y, section.heading_found = sy, sfound
            if sfound:
                section.page = sy_page
        chapter.sections.sort(key=lambda s: (s.page, s.y, s.level))


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def _resolve_pdf(book: str, path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    mapping = settings.pdf_paths
    if book not in mapping:
        raise ValueError(f"未知书目: {book}（可选 {sorted(mapping)}）")
    return mapping[book]


def _estimate_body_font_size(pages: Sequence[PageText], body_start: int) -> float:
    counter: Counter[float] = Counter()
    for page in pages:
        if page.pdf_page < body_start:
            continue
        for line in page.lines:
            for font, n in line.fonts.items():
                if _is_mono_font(font):
                    continue
                size = line.size
                if _BODY_SIZE_RANGE[0] <= size <= _BODY_SIZE_RANGE[1]:
                    counter[round(size, 1)] += n
    if not counter:
        return 12.0
    return counter.most_common(1)[0][0]


def _estimate_body_top(pages: Sequence[PageText], body_start: int, body_font_size: float) -> float:
    """正文字号行的最小 y0（用最大空隙法排除页眉）。"""
    ys: list[float] = []
    for page in pages:
        if page.pdf_page < body_start:
            continue
        for line in page.lines:
            if abs(line.size - body_font_size) <= max(0.3, body_font_size * 0.05):
                ys.append(round(line.y0, 1))
    if not ys:
        return 0.0
    uniq = sorted(set(ys))
    # 只看页面顶部 1/3，找最大空隙
    head = [y for y in uniq if y < 260]
    best_gap = 0.0
    body_top = uniq[0]
    for a, b in zip(head, head[1:]):
        gap = b - a
        if gap > best_gap:
            best_gap, body_top = gap, b
    if best_gap >= 10.0:
        return body_top
    return uniq[0]


def extract_book(
    book: str,
    *,
    path: Path | None = None,
    page_numbers: Sequence[int] | None = None,
    metadata: dict | None = None,
) -> BookExtract:
    """解析一本书。

    Args:
        book: "design" / "deep"。
        path: 覆盖 PDF 路径（测试用）。
        page_numbers: 只解析这些 PDF 页码（1 基，测试用）；None = 全部。
        metadata: 预置的附加信息（方便测试或上层复用）。
    """
    pdf_path = _resolve_pdf(book, path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到 PDF：{pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    try:
        pdf_pages = doc.page_count
        toc = doc.get_toc(simple=True) or []
        body_start = detect_body_start(toc, pdf_pages)

        pages: list[PageText] = []
        wanted = sorted(set(page_numbers)) if page_numbers else list(range(1, pdf_pages + 1))
        for pno in wanted:
            if not (1 <= pno <= pdf_pages):
                continue
            page = doc[pno - 1]
            lines = extract_page_lines(page, book, pno)
            pages.append(
                PageText(
                    book=book,
                    pdf_page=pno,
                    print_page=pno,  # 先占位，页码偏移检测后统一回填
                    lines=lines,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                )
            )
        offset, method, _confidence = detect_print_offset(doc, pages, body_start)
        for page in pages:
            page.print_page = _print_page(page.pdf_page, offset)
    finally:
        doc.close()

    body_font_size = _estimate_body_font_size(pages, body_start)
    body_top = _estimate_body_top(pages, body_start, body_font_size)
    chapters = build_chapters(toc, body_start, pdf_pages)
    page_map = {p.pdf_page: p for p in pages}
    locate_headings(chapters, page_map, body_font_size)

    outline = [
        OutlineEntry(
            level=int(level),
            title=str(title).strip(),
            page=int(page),
            chapter_no=parse_chapter_no(str(title)),
            section_no=parse_section_no(str(title))[0],
        )
        for level, title, page in toc
    ]

    meta = metadata or {}
    return BookExtract(
        book=book,
        book_title=meta.get("book_title") or BOOK_TITLES.get(book, book),
        pdf_path=str(pdf_path),
        pdf_pages=pdf_pages,
        pages=pages,
        outline=outline,
        chapters=chapters,
        print_offset=offset,
        print_offset_method=method,
        body_start_page=body_start,
        body_font_size=body_font_size,
        body_top=body_top,
        metadata_title=str(meta.get("metadata_title") or ""),
    )


def extract_all(*, page_numbers: Sequence[int] | None = None) -> dict[str, BookExtract]:
    """解析全部书目。"""
    out: dict[str, BookExtract] = {}
    for book in (settings.pdf_paths.keys() or ["design", "deep"]):
        out[book] = extract_book(book, page_numbers=page_numbers)
    return out


def to_doc_pages(book_extract: BookExtract) -> list[DocPage]:
    """转换为共享契约 DocPage（保留原始页文本，不含行级信息）。"""
    return [
        DocPage(
            book=page.book,
            pdf_page=page.pdf_page,
            print_page=page.print_page,
            text=page.text,
        )
        for page in book_extract.pages
    ]


def count_chapters(book_extract: BookExtract) -> int:
    """正文实际章数（含无编号的前言/附录）。"""
    return len(book_extract.chapters)


__all__ = [
    "BookExtract",
    "Chapter",
    "OutlineEntry",
    "PageText",
    "Section",
    "TextLine",
    "build_chapters",
    "cjk_count",
    "count_chapters",
    "detect_body_start",
    "detect_print_offset",
    "extract_all",
    "extract_book",
    "extract_page_lines",
    "is_cjk",
    "locate_headings",
    "normalize_title",
    "parse_chapter_no",
    "parse_section_no",
    "strip_chapter_prefix",
    "to_doc_pages",
]
