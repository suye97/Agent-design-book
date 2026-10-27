# -*- coding: utf-8 -*-
"""三级意图路由：显式指令覆盖 > LLM 结构化判定（含主题归属纠偏）> 关键词规则兜底。

输出 :class:`app.models.RouteDecision`，其中 ``reason`` 必须可解释（便于复盘误判）。

路由要回答两个问题，且**顺序不能颠倒**：

1. **主题归属**：这个主题是哪本书系统讲的？（上下文工程/压缩、后训练、评估指标…
   只有《深入理解》有系统章节；提示链/并行化/护栏模式… 只有《设计模式》有模式章节）
2. **答案形态**：用户要的是"模式目录式答案"（是什么/怎么用/代码/场景）还是
   "机制展开式答案"（定义+动机+方案对比+权衡+为什么，常带「详细了解/展开/讲透」）。

主题只在一本书系统讲解时，以主题归属为准；两本书都讲时，用答案形态分胜负。
``intent`` 只描述答案形态，最终 ``route`` 落到"有内容的那本书"。

置信度规则：≥ ``settings.route_confidence_single`` → 单库；``[both, single)`` →
双库并行；``< both`` → 双库并行 + ``clarify_hint``。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.logging_conf import get_logger
from app.models import (
    BOOK_TITLES,
    Intent,
    RouteDecision,
    RouteKind,
    RouterSource,
)
from app.rag.llm import LlmUnavailable, get_llm

logger = get_logger("rag.router")

__all__ = [
    "CLARIFY_HINT",
    "DESIGN_TERMS",
    "DEEP_TERMS",
    "route_question",
]

CLARIFY_HINT = "你是想了解某个设计模式的具体用法，还是想深入理解它背后的原理？"

# --------------------------------------------------------------------------- #
# 规则兜底词表：三类信号
#   1) 深度诉求 DEPTH_TERMS      → explain（要展开、要原理、要方案对比）
#   2) 模式诉求 PATTERN_TERMS    → concept（要模式用法、场景、代码）
#   3) 主题归属 TOPIC_COVERAGE   → 该主题由哪本书系统覆盖
# --------------------------------------------------------------------------- #
DEPTH_TERMS: dict[str, float] = {
    # 深度诉求（最容易被误判的信号：出现在问句里就说明“不要一句话定义”）
    "详细了解": 1.6,
    "详细讲解": 1.6,
    "详细解释": 1.6,
    "详细说明": 1.5,
    "详细讲": 1.5,
    "展开讲": 1.4,
    "展开说": 1.2,
    "系统讲": 1.4,
    "系统地": 1.2,
    "讲透": 1.6,
    "全面了解": 1.0,
    "细致": 1.0,
    "深入": 0.8,
    # 原理/机制/对比类
    "为什么": 1.4,
    "为何": 1.4,
    "原因": 0.8,
    "原理": 1.3,
    "机制": 1.1,
    "本质": 1.1,
    "底层": 1.1,
    "内部实现": 1.3,
    "工作机理": 1.3,
    "怎么工作": 0.9,
    "如何工作": 0.9,
    "演进": 1.0,
    "权衡": 0.9,
    "取舍": 0.9,
    "区别": 0.8,
    "对比": 0.5,
    "优缺点": 0.8,
    "边界条件": 0.8,
    "数学": 0.5,
    "推导": 0.9,
    "公式": 0.8,
    "设计哲学": 1.0,
    "理论基础": 1.0,
    # 方案/策略类（“处理方案/解决方案”意味着要做方案对比，不是报模式名）
    "处理方案": 1.2,
    "解决方案": 1.0,
    "方案对比": 1.2,
    "策略": 0.7,
    "方法论": 1.0,
}

PATTERN_TERMS: dict[str, float] = {
    "设计模式": 1.4,
    "模式": 0.5,
    "适用场景": 0.9,
    "什么时候用": 0.9,
    "何时用": 0.8,
    "怎么用": 0.6,
    "如何使用": 0.6,
    "怎么实现": 0.9,
    "如何实现": 0.9,
    "代码示例": 1.1,
    "示例代码": 1.1,
    "最佳实践": 0.7,
    "一图速览": 0.8,
    "关键要点": 0.7,
    "落地手段": 0.9,
    "落地": 0.5,
    "有哪些": 0.5,
}

# 概念疑问词：单独出现不足以定向（“是什么”既可能问模式定义，也可能问技术主题），
# 遇强深度诉求时按 _CONCEPT_DISCOUNT 降权。
CONCEPT_FORM_TERMS: dict[str, float] = {
    "是什么": 0.5,
    "什么是": 0.5,
    "概念": 0.5,
    "定义": 0.4,
    "含义": 0.4,
}

# 主题归属表：term -> (design 覆盖权重, deep 覆盖权重)
#   长词优先匹配（“记忆管理”先于“记忆”），匹配后从文本中消费，避免重复计分。
_TOPIC_COVERAGE_RAW: list[tuple[str, float, float]] = [
    # ---- 仅《设计模式》有系统章节（模式名；整章级覆盖 1.8） ----
    ("提示链", 1.8, 0.0),
    ("提示词链", 1.8, 0.0),
    ("并行化", 1.8, 0.0),
    ("路由", 1.6, 0.0),
    ("反思", 1.8, 0.0),
    ("工具使用", 1.2, 0.0),
    ("规划", 0.8, 0.0),
    ("记忆管理", 1.8, 0.0),
    ("学习与适应", 1.8, 0.0),
    ("目标设定", 1.8, 0.0),
    ("异常处理", 1.8, 0.0),
    ("人类参与环节", 1.8, 0.0),
    ("人工介入", 1.0, 0.0),
    ("hitl", 1.8, 0.0),
    ("a2a", 1.8, 0.0),
    ("智能体间通信", 1.8, 0.0),
    ("资源感知", 1.8, 0.0),
    ("优先级排序", 1.8, 0.0),
    ("探索与发现", 1.8, 0.0),
    # ---- 仅《深入理解》有系统章节（原理/机制主题；整章级覆盖 1.8） ----
    ("上下文工程", 0.0, 1.8),
    ("上下文压缩", 0.0, 1.8),
    ("kv cache", 0.0, 1.8),
    ("记忆系统", 0.0, 1.8),
    ("用户记忆", 0.0, 1.8),
    ("持续进化", 0.0, 1.8),
    ("工具设计", 0.0, 1.8),
    ("工具生态", 0.0, 1.0),
    ("coding agent", 0.0, 1.8),
    ("computer use", 0.0, 1.2),
    ("语音", 0.0, 0.8),
    ("机器人", 0.0, 0.8),
    ("评估指标", 0.0, 1.8),
    ("评估方法", 0.0, 1.8),
    ("评估环境", 0.0, 1.8),
    ("benchmark", 0.0, 1.2),
    ("后训练", 0.0, 1.8),
    ("sft", 0.0, 1.8),
    ("强化学习", 0.0, 1.8),
    ("蒸馏", 0.0, 1.2),
    ("预训练", 0.0, 1.2),
    ("agent 社会", 0.0, 1.2),
    ("失败模式", 0.0, 1.2),
    ("react", 0.0, 1.0),
    ("harness", 0.0, 1.8),
    ("观察空间", 0.0, 1.2),
    ("动作空间", 0.0, 1.2),
    # ---- 两本都讲、侧重不同（同时计分，靠形态信号分胜负） ----
    ("mcp", 1.4, 0.5),
    ("模型上下文协议", 1.4, 0.5),
    ("rag", 1.4, 0.5),
    ("检索增强", 1.2, 0.6),
    ("知识库", 0.4, 0.8),
    ("记忆", 0.8, 0.8),
    ("多智能体协作", 1.0, 1.0),
    ("多智能体", 0.8, 0.8),
    ("多 agent", 0.8, 0.8),
    ("多agent", 0.8, 0.8),
    ("工具调用", 1.2, 0.8),
    ("提示工程", 0.6, 1.2),
    ("护栏", 1.8, 0.3),
    ("guardrail", 1.8, 0.3),
    ("安全模式", 1.0, 0.2),
    ("评估", 0.6, 1.2),
    ("推理技术", 1.4, 0.4),
    ("编程智能体", 0.6, 1.2),
    ("上下文", 0.0, 0.4),
    ("缓存", 0.0, 0.8),
]
_TOPIC_COVERAGE = sorted(_TOPIC_COVERAGE_RAW, key=lambda item: -len(item[0]))

# 对外暴露的合集（便于调试 / 文档 / 旧接口兼容）
DESIGN_TERMS: dict[str, float] = {
    **PATTERN_TERMS,
    **CONCEPT_FORM_TERMS,
    **{term: dw for term, dw, _ in _TOPIC_COVERAGE_RAW if dw},
}
DEEP_TERMS: dict[str, float] = {
    **DEPTH_TERMS,
    **{term: hw for term, _, hw in _TOPIC_COVERAGE_RAW if hw},
}

# 打分饱和阈值：获胜方得分达到该值即视为“证据充分”
_SATURATION = 2.0
# 视为“强深度诉求”的深度得分门槛
_DEPTH_STRONG = 1.0
# LLM 与主题归属明显冲突时的纠偏倍数
_GUARD_RATIO = 1.3
# 纠偏后允许的最小置信度（避免纠偏结果掉进双库区间）
_GUARD_CONFIDENCE = 0.7


# --------------------------------------------------------------------------- #
# 显式指令覆盖
# --------------------------------------------------------------------------- #
_BOTH_PATTERNS = [
    r"两本\s*书?\s*都",
    r"两本\s*书",
    r"两本\s*(?:书)?\s*(?:对比|比较)",
    r"(?:对比|比较)\s*两本",
    r"两本[^，,。？?]{0,8}(?:对比|比较|异同)",
    r"同时\s*(?:参考|看|查|用)\s*两本",
    r"分别\s*(?:看|查|用|参考)\s*两本",
    r"双库",
    r"两本一起",
]

_DESIGN_BOOK_PATTERNS = [
    r"设计模式那本(?:书)?",
    r"设计模式这(?:本)?书",
    r"设计模式书",
    r"那本设计模式",
    r"这本设计模式",
    r"《[^》]{0,24}设计模式[^》]{0,24}》",
    r"智能体设计模式",
    r"agent\s*设计模式",
]

_DEEP_BOOK_PATTERNS = [
    r"深入理解[这那]本(?:书)?",
    r"深入理解(?:这|那)?书",
    r"《[^》]{0,24}深入理解[^》]{0,24}》",
    r"李博杰",
    r"原理那本(?:书)?",
    r"原理这(?:本)?书",
    r"原理书",
    r"深入理解\s*ai\s*agent",
]

# 弱信号：单独出现“深入理解”也视为想走深入理解书（设计方案第 7.2 节）
_DEEP_WEAK_PATTERN = r"深入理解"

_COMPILED_BOTH = [re.compile(p, re.IGNORECASE) for p in _BOTH_PATTERNS]
_COMPILED_DESIGN = [re.compile(p, re.IGNORECASE) for p in _DESIGN_BOOK_PATTERNS]
_COMPILED_DEEP = [re.compile(p, re.IGNORECASE) for p in _DEEP_BOOK_PATTERNS]
_COMPILED_DEEP_WEAK = re.compile(_DEEP_WEAK_PATTERN, re.IGNORECASE)


def _first_match(question: str, patterns: list[re.Pattern[str]]) -> str | None:
    for pattern in patterns:
        matched = pattern.search(question)
        if matched:
            return matched.group(0).strip()
    return None


def _rewrite_question(question: str, matched: str | None) -> str:
    """去掉显式书目指代，得到用于检索的 rewritten_query。"""
    original = question.strip()
    if not matched:
        return original
    rewritten = question.replace(matched, " ")
    rewritten = re.sub(r"^[里中内]\s*", "", rewritten.strip())
    rewritten = rewritten.strip(" ，,。；;：:、")
    return rewritten or original


def _explicit_override(
    question: str, book_override: str | None
) -> RouteDecision | None:
    """显式指令覆盖：调用方指定 > 用户问题里点名书目。"""
    if book_override:
        hint = str(book_override).strip().lower()
        if hint in {"both", "all", "两本"}:
            return RouteDecision(
                route=RouteKind.BOTH,
                intent=Intent.AMBIGUOUS,
                confidence=1.0,
                source=RouterSource.OVERRIDE,
                reason="显式指令覆盖：调用方指定同时检索两本书（book_override=both）",
                rewritten_query=question.strip(),
                book_hint=None,
            )
        if hint in {"design", "deep"}:
            route = RouteKind.DESIGN if hint == "design" else RouteKind.DEEP
            intent = Intent.CONCEPT if hint == "design" else Intent.EXPLAIN
            return RouteDecision(
                route=route,
                intent=intent,
                confidence=1.0,
                source=RouterSource.OVERRIDE,
                reason=(
                    f"显式指令覆盖：调用方指定 book_override={hint}"
                    f" → 《{BOOK_TITLES[hint]}》"
                ),
                rewritten_query=question.strip(),
                book_hint=hint,
            )
        logger.warning("忽略未知 book_override=%r（应为 design/deep/both/None）", book_override)

    both_match = _first_match(question, _COMPILED_BOTH)
    design_match = _first_match(question, _COMPILED_DESIGN)
    deep_match = _first_match(question, _COMPILED_DEEP)

    if both_match:
        return RouteDecision(
            route=RouteKind.BOTH,
            intent=Intent.AMBIGUOUS,
            confidence=1.0,
            source=RouterSource.OVERRIDE,
            reason=f"显式指令覆盖：问题中提到“{both_match}”，用户要求同时参考两本书",
            rewritten_query=_rewrite_question(question, both_match),
        )
    if design_match and deep_match:
        return RouteDecision(
            route=RouteKind.BOTH,
            intent=Intent.AMBIGUOUS,
            confidence=1.0,
            source=RouterSource.OVERRIDE,
            reason=(
                "显式指令覆盖：问题中同时点名两本书"
                f"（“{design_match}”/“{deep_match}”）"
            ),
            rewritten_query=_rewrite_question(question, design_match),
        )
    if design_match:
        return RouteDecision(
            route=RouteKind.DESIGN,
            intent=Intent.CONCEPT,
            confidence=1.0,
            source=RouterSource.OVERRIDE,
            reason=(
                f"显式指令覆盖：问题中提到“{design_match}”"
                f" → 《{BOOK_TITLES[RouteKind.DESIGN.value]}》"
            ),
            rewritten_query=_rewrite_question(question, design_match),
            book_hint=RouteKind.DESIGN.value,
        )
    if deep_match:
        return RouteDecision(
            route=RouteKind.DEEP,
            intent=Intent.EXPLAIN,
            confidence=1.0,
            source=RouterSource.OVERRIDE,
            reason=(
                f"显式指令覆盖：问题中提到“{deep_match}”"
                f" → 《{BOOK_TITLES[RouteKind.DEEP.value]}》"
            ),
            rewritten_query=_rewrite_question(question, deep_match),
            book_hint=RouteKind.DEEP.value,
        )
    weak = _COMPILED_DEEP_WEAK.search(question)
    if weak:
        return RouteDecision(
            route=RouteKind.DEEP,
            intent=Intent.EXPLAIN,
            confidence=1.0,
            source=RouterSource.OVERRIDE,
            reason=(
                "显式指令覆盖：问题中出现“深入理解”，视为要求深入解释"
                f" → 《{BOOK_TITLES[RouteKind.DEEP.value]}》"
            ),
            rewritten_query=_rewrite_question(question, weak.group(0)),
            book_hint=RouteKind.DEEP.value,
        )
    return None


# --------------------------------------------------------------------------- #
# 规则兜底
# --------------------------------------------------------------------------- #
def _score_terms(text_lower: str, table: dict[str, float]) -> tuple[float, list[str]]:
    total = 0.0
    hits: list[str] = []
    for term, weight in table.items():
        if term.isascii():
            # 英文/缩写用词边界匹配，避免 "rag" 命中 "storage"
            pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
            found = re.search(pattern, text_lower) is not None
        else:
            found = term in text_lower
        if found:
            total += weight
            hits.append(term)
    return total, hits


@dataclass
class _Signals:
    """一次提问的三类信号快照（供规则路由与 LLM 纠偏共用）。"""

    depth_score: float = 0.0
    pattern_score: float = 0.0
    concept_score: float = 0.0
    design_topic: float = 0.0
    deep_topic: float = 0.0
    depth_hits: list[str] = field(default_factory=list)
    pattern_hits: list[str] = field(default_factory=list)
    concept_hits: list[str] = field(default_factory=list)
    design_topic_hits: list[str] = field(default_factory=list)
    deep_topic_hits: list[str] = field(default_factory=list)

    @property
    def design_score(self) -> float:
        """概念疑问词（是什么/概念/定义）只描述“答案形态”，不指示“该查哪本书”。

        因此它不参与定向：``是什么`` 既能问模式定义（design），也能问技术主题定义（deep），
        真正决定书的是模式诉求与主题归属。
        """
        return self.pattern_score + self.design_topic

    @property
    def deep_score(self) -> float:
        return self.depth_score + self.deep_topic

    def summary(self) -> str:
        return (
            f"设计模式信号 {self.design_score:.2f}"
            f"（模式 {self.pattern_score:.2f} + 主题 {self.design_topic:.2f}）"
            f"／深入理解信号 {self.deep_score:.2f}"
            f"（深度 {self.depth_score:.2f} + 主题 {self.deep_topic:.2f}）"
        )


def _score_topics(text_lower: str) -> tuple[float, float, list[str], list[str]]:
    """主题归属打分：长词优先、命中即消费，返回 (design, deep, hits, hits)。"""
    remaining = text_lower
    design_score = deep_score = 0.0
    design_hits: list[str] = []
    deep_hits: list[str] = []
    for term, design_weight, deep_weight in _TOPIC_COVERAGE:
        if term.isascii():
            pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
            matched = re.search(pattern, remaining)
            if matched is None:
                continue
            start, end = matched.span()
            remaining = remaining[:start] + "　" * (end - start) + remaining[end:]
        else:
            if term not in remaining:
                continue
            remaining = remaining.replace(term, "　", 1)
        if design_weight:
            design_score += design_weight
            design_hits.append(term)
        if deep_weight:
            deep_score += deep_weight
            deep_hits.append(term)
    return design_score, deep_score, design_hits, deep_hits


def _analyze_question(question: str) -> _Signals:
    text_lower = question.lower()
    depth_score, depth_hits = _score_terms(text_lower, DEPTH_TERMS)
    pattern_score, pattern_hits = _score_terms(text_lower, PATTERN_TERMS)
    concept_score, concept_hits = _score_terms(text_lower, CONCEPT_FORM_TERMS)
    design_topic, deep_topic, design_topic_hits, deep_topic_hits = _score_topics(text_lower)
    return _Signals(
        depth_score=depth_score,
        pattern_score=pattern_score,
        concept_score=concept_score,
        design_topic=design_topic,
        deep_topic=deep_topic,
        depth_hits=depth_hits,
        pattern_hits=pattern_hits,
        concept_hits=concept_hits,
        design_topic_hits=design_topic_hits,
        deep_topic_hits=deep_topic_hits,
    )


def _rule_confidence(design_score: float, deep_score: float) -> float:
    """把两类得分归一化为 0–1 置信度：领先幅度 × 证据强度。

    领先幅度 = ``1 - 落后方/领先方``；证据强度 = ``min(1, 领先方/2)``。
    两者相乘避免“单个弱关键词”给出过高置信度。
    """
    if design_score <= 0 and deep_score <= 0:
        return 0.0
    winner = max(design_score, deep_score)
    loser = min(design_score, deep_score)
    margin = 1.0 - (loser / winner)
    strength = min(1.0, winner / _SATURATION)
    return round(margin * strength, 4)


def _fmt_hits(hits: list[str], limit: int = 4) -> str:
    if not hits:
        return "无"
    shown = "、".join(hits[:limit])
    if len(hits) > limit:
        shown += f" 等 {len(hits)} 个"
    return shown


def _rule_route(question: str) -> RouteDecision:
    signals = _analyze_question(question)
    design_score = signals.design_score
    deep_score = signals.deep_score
    confidence = _rule_confidence(design_score, deep_score)
    single_th = float(settings.route_confidence_single)
    both_th = float(settings.route_confidence_both)
    rewritten = question.strip()

    hit_parts = []
    if signals.depth_hits:
        hit_parts.append(f"深度[{_fmt_hits(signals.depth_hits)}]")
    if signals.pattern_hits:
        hit_parts.append(f"模式[{_fmt_hits(signals.pattern_hits)}]")
    if signals.concept_hits:
        hit_parts.append(f"概念[{_fmt_hits(signals.concept_hits)}]")
    topic_hits = signals.design_topic_hits + signals.deep_topic_hits
    if topic_hits:
        hit_parts.append(f"主题[{_fmt_hits(topic_hits)}]")
    score_detail = signals.summary()
    if hit_parts:
        score_detail += "；命中 " + " ".join(hit_parts)

    if confidence >= single_th and design_score != deep_score:
        if design_score > deep_score:
            route, intent = RouteKind.DESIGN, Intent.CONCEPT
            target = "设计模式书（模式用法 / 代码示例）"
        else:
            route, intent = RouteKind.DEEP, Intent.EXPLAIN
            target = "深入理解书（原理 / 机制 / 方案对比）"
        return RouteDecision(
            route=route,
            intent=intent,
            confidence=confidence,
            source=RouterSource.RULE,
            reason=(
                f"规则路由：{score_detail} ≥ 单库阈值 {single_th:.2f}"
                f" → {target}"
            ),
            rewritten_query=rewritten,
        )

    if confidence >= both_th:
        return RouteDecision(
            route=RouteKind.BOTH,
            intent=Intent.AMBIGUOUS,
            confidence=confidence,
            source=RouterSource.RULE,
            reason=(
                f"规则路由：{score_detail} 落在双库区间 "
                f"[{both_th:.2f}, {single_th:.2f}) → 两本书并行检索"
            ),
            rewritten_query=rewritten,
        )

    if design_score <= 0 and deep_score <= 0:
        reason = (
            "规则路由：问题中没有命中模式名/原理词/主题归属"
            f"（置信度 {confidence:.2f} < {both_th:.2f}）→ 双库并行并请求澄清"
        )
    else:
        reason = (
            f"规则路由：{score_detail} < 双库阈值 {both_th:.2f}"
            " → 证据不足，双库并行并请求澄清"
        )
    return RouteDecision(
        route=RouteKind.BOTH,
        intent=Intent.AMBIGUOUS,
        confidence=confidence,
        source=RouterSource.RULE,
        reason=reason,
        rewritten_query=rewritten,
        clarify_hint=CLARIFY_HINT,
    )


# --------------------------------------------------------------------------- #
# LLM 结构化判定
# --------------------------------------------------------------------------- #
_LLM_ROUTER_SYSTEM = """你是「Agent 设计问答」知识库的意图路由器。知识库只有两本书：
1. design =《智能体设计模式：智能系统构建实战指南》：按“模式”组织，讲某个模式的定位、
   适用场景、怎么用、代码示例、最佳实践。系统覆盖：提示链、路由、并行化、反思、工具使用、
   规划、多智能体协作、记忆管理、学习与适应、MCP、目标设定、异常处理、HITL、RAG、A2A、
   资源感知、推理技术、护栏、评估与监控、优先级排序、探索与发现。
2. deep =《深入理解 AI Agent：设计原理与工程实践》：按“原理/机制”组织，讲为什么这样设计、
   内部如何运作、方案对比与权衡、演进。系统覆盖：上下文工程与上下文压缩、KV Cache、提示工程、
   记忆系统、工具设计、Coding Agent、交互（语音/Computer Use/机器人）、Agent 评估、
   模型后训练（SFT/RL/蒸馏）、持续进化、多 Agent 协作机制与失败模式、Agent 社会。

判定顺序（必须遵守）：
第一步 判主题归属：这个问题问的主题，哪本书有系统讲解？只在一本书系统讲解时，book_hint 指向该书。
第二步 判答案形态：
- intent=concept：要“模式目录式”答案——某个模式是什么、怎么用、适合什么场景、有没有代码示例。
- intent=explain：要“机制展开式”答案——为什么这样设计、原理/机制/底层、方案对比与权衡、演进。
- intent=ambiguous：两本书都可能有、问题过于笼统，或用户明确要对比两本书。

关键规则：
- 出现“概念”二字不代表 concept；判断标准是“用户要的答案形态”，不是词面。
- “详细了解 / 详细讲 / 展开 / 系统讲 / 讲透 / 处理方案 / 解决方案 / 方案对比 / 权衡 / 原理 / 机制”
  都是 explain 的强信号。
- 主题只在一本书系统讲解时（如上下文压缩、后训练只 deep；提示链、并行化只 design），
  即使问法像“是什么”，也按该书路由：主题归属优先于疑问词形态。
- 两本书都讲的话题（MCP、RAG、记忆、多智能体、评估、护栏）用答案形态分胜负：
  问“模式怎么用 / 代码 / 场景” → design；问“为什么 / 机制 / 方案对比 / 权衡” → deep；难分 → ambiguous。
- 只是出现“场景”二字不算 explain。

只输出一个 JSON 对象，不要输出任何其他文字，字段如下：
{"intent":"concept|explain|ambiguous","confidence":0到1之间的浮点数,
 "book_hint":"design|deep|null","rewritten_query":"去掉指代后的检索用问题",
 "reason":"一句中文理由"}

示例：
问：提示链模式怎么用？有代码示例吗 → {"intent":"concept","confidence":0.95,"book_hint":"design","rewritten_query":"提示链模式怎么用 代码示例","reason":"询问模式用法与代码示例"}
问：并行化模式适合什么场景？ → {"intent":"concept","confidence":0.9,"book_hint":"design","rewritten_query":"并行化模式 适用场景","reason":"询问模式适用场景，属于模式用法"}
问：提示链模式是什么 → {"intent":"concept","confidence":0.93,"book_hint":"design","rewritten_query":"提示链模式 定义","reason":"询问某个设计模式的定义"}
问：我想详细了解上下文压缩的概念及处理方案 → {"intent":"explain","confidence":0.93,"book_hint":"deep","rewritten_query":"上下文压缩 概念 处理方案 策略对比","reason":"要深入展开概念与处理方案，且该主题仅深入理解书系统讲解"}
问：上下文工程是什么 → {"intent":"explain","confidence":0.85,"book_hint":"deep","rewritten_query":"上下文工程 定义 机制","reason":"该主题仅深入理解书系统讲解，需要定义加机制展开"}
问：上下文压缩有哪几种策略，各自的权衡 → {"intent":"explain","confidence":0.9,"book_hint":"deep","rewritten_query":"上下文压缩 策略 权衡","reason":"询问方案对比与取舍"}
问：为什么要给 Agent 加护栏？背后的控制机制是什么 → {"intent":"explain","confidence":0.92,"book_hint":"deep","rewritten_query":"Agent 护栏 设计原因 控制机制","reason":"询问设计原因与内部机制"}
问：护栏与安全模式有哪些落地手段 → {"intent":"concept","confidence":0.88,"book_hint":"design","rewritten_query":"护栏 落地手段","reason":"询问模式的具体落地手段"}
问：两本书怎么看待记忆管理 → {"intent":"ambiguous","confidence":0.9,"book_hint":null,"rewritten_query":"记忆管理","reason":"用户明确要对比两本书"}
问：多智能体怎么协调 → {"intent":"ambiguous","confidence":0.5,"book_hint":null,"rewritten_query":"多智能体协调","reason":"模式与机制都可能相关"}
"""


def _normalize_intent(value: Any) -> Intent | None:
    raw = str(value or "").strip().lower()
    if raw in {"concept", "概念", "pattern", "design", "设计", "模式"}:
        return Intent.CONCEPT
    if raw in {"explain", "explanation", "原理", "解释", "deep", "深入"}:
        return Intent.EXPLAIN
    if raw in {"ambiguous", "both", "两本", "unknown", "不确定"}:
        return Intent.AMBIGUOUS
    return None


def _normalize_book_hint(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if raw in {"design", "设计", "设计模式"}:
        return RouteKind.DESIGN.value
    if raw in {"deep", "深入", "深入理解", "原理"}:
        return RouteKind.DEEP.value
    return None


def _coerce_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf > 1.0:  # 容忍百分数写法
        conf = conf / 100.0
    return round(min(1.0, max(0.0, conf)), 4)


def _decision_from_llm(data: dict[str, Any], question: str) -> RouteDecision:
    intent = _normalize_intent(data.get("intent"))
    if intent is None:
        raise ValueError("LLM 返回的 intent 字段非法")
    confidence = _coerce_confidence(data.get("confidence"))
    rewritten = str(data.get("rewritten_query") or "").strip() or question.strip()
    llm_reason = str(data.get("reason") or "").strip() or "LLM 未给出理由"
    book_hint = _normalize_book_hint(data.get("book_hint"))
    single_th = float(settings.route_confidence_single)
    both_th = float(settings.route_confidence_both)
    signals = _analyze_question(question)
    guard_note = ""

    # 第一优先级：主题归属。book_hint 表示“这题该查哪本书”，比答案形态更硬：
    # “上下文工程是什么”是概念式问法，但该主题只有 deep 有系统讲解 → 仍落 deep。
    if book_hint and confidence >= single_th and intent in (Intent.CONCEPT, Intent.EXPLAIN):
        route = RouteKind(book_hint)
        final_intent = intent
        target = (
            "深入理解书（按 LLM 主题归属）"
            if route is RouteKind.DEEP
            else "设计模式书（按 LLM 主题归属）"
        )
    elif intent is Intent.CONCEPT and confidence >= single_th:
        route, final_intent = RouteKind.DESIGN, Intent.CONCEPT
        target = "设计模式书"
    elif intent is Intent.EXPLAIN and confidence >= single_th:
        route, final_intent = RouteKind.DEEP, Intent.EXPLAIN
        target = "深入理解书"
    elif intent in (Intent.CONCEPT, Intent.EXPLAIN):
        route, final_intent = RouteKind.BOTH, intent
        target = "双库并行（置信度未达单库阈值）"
    else:
        route, final_intent = RouteKind.BOTH, Intent.AMBIGUOUS
        target = "双库并行（意图不明确）"

    # 第二优先级：信号纠偏。LLM 被“概念/是什么”字面带偏时，用主题/深度信号拉回来
    # （只在高置信、且信号差距明显时生效，避免覆盖 LLM 的合理判断）。
    if route is RouteKind.DESIGN and signals.deep_score >= max(
        1.2, signals.design_score * _GUARD_RATIO
    ):
        route, final_intent = RouteKind.DEEP, Intent.EXPLAIN
        confidence = max(confidence, _GUARD_CONFIDENCE)
        guard_note = f"；主题纠偏：{signals.summary()} 明显指向深入理解书"
        target = "深入理解书（信号纠偏）"
    elif route is RouteKind.DEEP and signals.design_score >= max(
        1.2, signals.deep_score * _GUARD_RATIO
    ):
        route, final_intent = RouteKind.DESIGN, Intent.CONCEPT
        confidence = max(confidence, _GUARD_CONFIDENCE)
        guard_note = f"；主题纠偏：{signals.summary()} 明显指向设计模式书"
        target = "设计模式书（信号纠偏）"

    clarify = None
    if confidence < both_th:
        clarify = CLARIFY_HINT

    reason = (
        f"LLM 判定：{llm_reason}（intent={final_intent.value}, "
        f"confidence={confidence:.2f}）→ {target}{guard_note}"
    )
    return RouteDecision(
        route=route,
        intent=final_intent,
        confidence=confidence,
        source=RouterSource.LLM,
        reason=reason,
        rewritten_query=rewritten,
        book_hint=route.value if route is not RouteKind.BOTH else book_hint,
        clarify_hint=clarify,
    )


def _llm_route(question: str, llm: Any | None) -> RouteDecision:
    client = llm if llm is not None else get_llm()
    if not client.is_available():
        raise LlmUnavailable("LLM 未配置")
    messages = [
        {"role": "system", "content": _LLM_ROUTER_SYSTEM},
        {"role": "user", "content": f"问题：{question.strip()}"},
    ]
    # max_tokens 用 chat_json 的默认值（settings.llm_router_max_tokens）：
    # deepseek-flash 等推理模型会先输出 reasoning_content，小值会导致 content 为空。
    data = client.chat_json(messages, temperature=0.0)
    return _decision_from_llm(data, question)


# --------------------------------------------------------------------------- #
# 公共入口
# --------------------------------------------------------------------------- #
def route_question(
    question: str,
    book_override: str | None = None,
    llm: Any | None = None,
) -> RouteDecision:
    """三级路由主入口。

    Args:
        question: 用户原始问题。
        book_override: 调用方显式指定书目（``"design"`` / ``"deep"`` / ``"both"`` / None）。
        llm: 可注入的 LLM 客户端（测试/自定义端点用）；None 时用进程内单例。

    Returns:
        RouteDecision: 始终包含可读的 ``reason``；任何 LLM 失败都会降级到规则路由。
    """
    question = (question or "").strip()
    if not question:
        return RouteDecision(
            route=RouteKind.BOTH,
            intent=Intent.AMBIGUOUS,
            confidence=0.0,
            source=RouterSource.RULE,
            reason="规则路由：问题为空 → 双库并行并请求澄清",
            rewritten_query="",
            clarify_hint=CLARIFY_HINT,
        )

    override = _explicit_override(question, book_override)
    if override is not None:
        logger.info(
            "路由[override] route=%s conf=%.2f q=%r",
            override.route.value,
            override.confidence,
            question,
        )
        return override

    llm_error = ""
    try:
        decision = _llm_route(question, llm)
        logger.info(
            "路由[llm] route=%s conf=%.2f q=%r",
            decision.route.value,
            decision.confidence,
            question,
        )
        return decision
    except Exception as exc:  # noqa: BLE001 - 路由绝不能让问答链路崩溃
        llm_error = type(exc).__name__
        logger.info("LLM 路由不可用（%s），降级为规则路由", llm_error)

    decision = _rule_route(question)
    decision.reason = (
        f"LLM 判定不可用（{llm_error or '未知'}），已降级为规则路由。{decision.reason}"
    )
    logger.info(
        "路由[rule] route=%s conf=%.2f q=%r",
        decision.route.value,
        decision.confidence,
        question,
    )
    return decision
