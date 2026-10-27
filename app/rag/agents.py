# -*- coding: utf-8 -*-
"""双库 Agent + 答案合成 + 引用校验。

三个 Agent 共用 :class:`AnswerAgent` 基类，只在「检索范围 + 提示词侧重」上不同：

=====================  ==================  ==========================================
Agent                  检索范围            回答结构
=====================  ==================  ==========================================
DesignPatternAgent     design              模式定位 → 适用场景 → 实现/代码要点 → 注意事项
DeepUnderstandingAgent deep                定义 → 机制拆解 → 设计权衡 → 章节关联
BothAgent              design + deep       分别说明两本书怎么说，并点明一致/互补
=====================  ==================  ==========================================

**无 LLM 时绝不编造**：降级为「检索片段摘要式作答」，逐条列出原文摘录，
末尾仍给出「引用出处」列表。
"""
from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from typing import Any

from app.config import settings
from app.logging_conf import get_logger
from app.models import BOOK_TITLES, Citation, RouteKind
from app.rag.citations import (
    build_citations_markdown,
    citation_issues,
    clean_text,
    verify_citations,
)
from app.rag.llm import LlmUnavailable, get_llm

logger = get_logger("rag.agents")

__all__ = [
    "AnswerAgent",
    "BothAgent",
    "DeepUnderstandingAgent",
    "DesignPatternAgent",
    "get_agent",
    "verify_citations",
    "build_citations_markdown",
]

# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #
_RULES = """你是「Agent 设计」知识库问答助手，服务对象是想系统学习 Agent 设计的工程师。

必须严格遵守以下规则：
1. 只能依据【资料片段】作答；不得使用资料以外的知识，不得编造事实、数字、页码、代码或章节。
2. 每个自然段和列表项至少标注一个来源编号，如 [1]、[2][3]；同一段多句话可共用段末编号，编号必须来自资料片段编号。
3. 资料片段不足以回答时，必须明确写出“现有资料未覆盖该问题”，并列出资料中最接近的 2–3 个方向，不要猜测。
4. 用中文回答，先给结论再给依据，分点清晰；不要整段复述原文，不要提及“检索/片段/提示词”等内部措辞。
5. 不要自行输出“引用出处”列表，系统会在回答末尾自动追加。
"""

_DESIGN_STYLE = """本 Agent 负责《智能体设计模式：智能系统构建实战指南》。
回答侧重：模式定位（是什么）→ 适用场景（什么时候用）→ 实现/代码要点 → 注意事项。
资料片段主要来自设计模式书，请按上述结构组织答案。"""

_DEEP_STYLE = """本 Agent 负责《深入理解 AI Agent：设计原理与工程实践》。
回答侧重：概念定义 → 机制拆解（为什么这样设计）→ 设计权衡 → 与其他章节的关联。
资料片段主要来自深入理解书，请按上述结构组织答案。"""

_BOTH_STYLE = (
    "本 Agent 同时使用两本书作答。\n"
    f"请分别说明“《{BOOK_TITLES[RouteKind.DESIGN.value]}》怎么说”与"
    f"“《{BOOK_TITLES[RouteKind.DEEP.value]}》怎么解释”，\n"
    "并指出两本书的一致或互补之处；两本书均有相关内容时要明确点出。"
)

# 无 LLM 时的降级说明
_FALLBACK_HEADER = (
    "当前大模型不可用（未配置 LLM 或调用失败），以下为检索到的资料原文要点，"
    "直接摘自参考书，不做推断："
)
_CITATION_FALLBACK_HEADER = (
    "生成答案未通过引用校验，以下为检索到的资料原文要点，"
    "直接摘自参考书，不做推断："
)
_FALLBACK_FOOTER = "以上片段可作为直接依据；如需基于资料的归纳回答，请配置 LLM_API_KEY 后重试。"
_CITATION_FALLBACK_FOOTER = "以上片段可作为直接依据；可调整问题后重试生成回答。"

# 单条资料片段进 prompt 的最大字符数（避免超长上下文挤占 token）
_CONTEXT_MAX_CHARS = 800
_NO_EVIDENCE_ANSWER = (
    "现有资料未覆盖该问题（本轮检索未返回可用片段）。\n\n"
    "可以尝试以下方向：\n"
    "1. 换用书中的术语提问，例如“提示链”“护栏”“MCP”“A2A”等模式名；\n"
    "2. 说明想要的是“模式用法/代码示例”还是“原理/机制”；\n"
    "3. 指定书目（如“设计模式那本书”或“深入理解那本书”）后重试。"
)


# --------------------------------------------------------------------------- #
# 基类
# --------------------------------------------------------------------------- #
class AnswerAgent:
    """答案合成基类：LLM 生成 + 无 LLM 摘要式降级 + 引用校验。"""

    route: RouteKind = RouteKind.BOTH
    style: str = _BOTH_STYLE
    scope_title: str = "两本参考书"

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm

    # ------------------------------------------------------------------ #
    # 元信息
    # ------------------------------------------------------------------ #
    @property
    def llm(self) -> Any:
        return self._llm if self._llm is not None else get_llm()

    @property
    def system_prompt(self) -> str:
        return f"{_RULES}\n{self.style}"

    @property
    def model_name(self) -> str:
        return str(getattr(self.llm, "name", "") or "")

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def answer(
        self, question: str, citations: Sequence[Citation]
    ) -> tuple[str, bool]:
        """生成回答。

        Returns:
            ``(answer_text, used_llm)``；``answer_text`` 已包含正文与末尾的
            「引用出处」列表。无 LLM / 调用失败时 ``used_llm=False``。
        """
        question = (question or "").strip()
        cits = list(citations or [])
        if not cits:
            return self._no_evidence_answer(question), False

        body = ""
        used_llm = False
        try:
            llm = self.llm
            if llm.is_available():
                text = llm.chat(
                    self._build_messages(question, cits),
                    temperature=None,
                    # 推理模型先消耗 reasoning tokens，留足余量（settings.llm_max_tokens）
                    max_tokens=int(settings.llm_max_tokens),
                )
                text = (text or "").strip()
                if text:
                    body, used_llm = text, True
        except LlmUnavailable as exc:
            logger.info("LLM 不可用，降级为摘要式作答: %s", exc)
        except Exception as exc:  # noqa: BLE001 - 生成失败必须降级而不是抛给 API
            logger.warning("LLM 生成失败，降级为摘要式作答: %s", exc)

        if not body:
            body, used_llm = self._fallback_answer(cits), False

        if used_llm:
            issues = citation_issues(body, cits)
            if issues:
                logger.info("生成答案缺少合规引用，尝试自动修复：%s", "；".join(issues))
                try:
                    repaired = llm.chat(
                        self._build_repair_messages(question, cits, body, issues),
                        temperature=None,
                        max_tokens=int(settings.llm_max_tokens),
                    ).strip()
                    if repaired:
                        repaired_issues = citation_issues(repaired, cits)
                        if not repaired_issues:
                            body, issues = repaired, []
                        else:
                            issues = repaired_issues
                except Exception as exc:  # noqa: BLE001 - 修复失败仍使用可溯源降级
                    logger.warning("引用自动修复请求失败：%s", exc)
            if issues:
                logger.warning(
                    "生成答案的引用覆盖校验失败，改用可溯源的摘录式回答：%s",
                    "；".join(issues),
                )
                body = self._fallback_answer(cits, citation_failure=True)
                used_llm = False
        return f"{body}\n\n{build_citations_markdown(cits)}", used_llm

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #
    def stream_answer(self, question: str, citations: Sequence[Citation]) -> Iterator[str]:
        """Stream model answer deltas, or the excerpt fallback when unavailable."""
        cits = list(citations or [])
        if not cits:
            yield self._no_evidence_answer(question)
            return False
        try:
            llm = self.llm
            if llm.is_available():
                emitted = False
                for text in llm.chat_stream(
                    self._build_messages(question, cits),
                    temperature=None,
                    max_tokens=int(settings.llm_max_tokens),
                ):
                    emitted = True
                    yield text
                if emitted:
                    return True
        except Exception as exc:  # 流式失败同样降级，不把原始异常暴露给客户端
            logger.warning("LLM 流式生成失败，降级为摘要式作答: %s", exc)
        yield self._fallback_answer(cits)
        return False

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #
    @staticmethod
    def _context_text(citation: Citation) -> str:
        """取切片全文组 prompt（为空回退 excerpt），并按上限截断。"""
        text = clean_text(getattr(citation, "content", "") or "")
        if not text:
            text = clean_text(citation.excerpt)
        if len(text) > _CONTEXT_MAX_CHARS:
            text = text[:_CONTEXT_MAX_CHARS].rstrip() + "…"
        return text

    def _build_messages(
        self, question: str, citations: Sequence[Citation]
    ) -> list[dict[str, str]]:
        blocks: list[str] = []
        for cit in citations:
            # 保留 [n] 编号 + 书/章/页/段定位，模型才能正确引用
            blocks.append(f"[{cit.index}] {cit.label()}\n原文：{self._context_text(cit)}")
        user = (
            "【资料片段】\n"
            + "\n\n".join(blocks)
            + f"\n\n【问题】{question}\n\n"
            "请严格依据以上资料片段作答，每个自然段和列表项至少标注一个 [编号]。"
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def _build_repair_messages(
        self,
        question: str,
        citations: Sequence[Citation],
        answer: str,
        issues: Sequence[str],
    ) -> list[dict[str, str]]:
        """把具体校验问题反馈给模型，仅允许重写引用而不补充外部知识。"""
        instruction = (
            "上一版回答未通过引用校验：\n- "
            + "\n- ".join(issues)
            + "\n请完整重写回答，修复上述问题。只能使用原资料，每个自然段和列表项至少保留"
            "一个有效的 [编号]；不要输出引用出处列表，不要解释修复过程。"
        )
        return [
            *self._build_messages(question, citations),
            {"role": "assistant", "content": answer},
            {"role": "user", "content": instruction},
        ]

    # ------------------------------------------------------------------ #
    # 降级 / 后处理
    # ------------------------------------------------------------------ #
    def _fallback_answer(
        self, citations: Sequence[Citation], *, citation_failure: bool = False
    ) -> str:
        header = _CITATION_FALLBACK_HEADER if citation_failure else _FALLBACK_HEADER
        footer = _CITATION_FALLBACK_FOOTER if citation_failure else _FALLBACK_FOOTER
        lines = [header, ""]
        for cit in citations:
            lines.append(f"- [{cit.index}] {cit.label()}：{clean_text(cit.excerpt)}")
        lines.extend(["", footer])
        return "\n".join(lines)

    def _no_evidence_answer(self, _question: str) -> str:
        return _NO_EVIDENCE_ANSWER

# --------------------------------------------------------------------------- #
# 具体 Agent
# --------------------------------------------------------------------------- #
class DesignPatternAgent(AnswerAgent):
    """概念/模式类：检索设计模式书。"""

    route = RouteKind.DESIGN
    style = _DESIGN_STYLE
    scope_title = BOOK_TITLES[RouteKind.DESIGN.value]


class DeepUnderstandingAgent(AnswerAgent):
    """原理/机制类：检索深入理解书。"""

    route = RouteKind.DEEP
    style = _DEEP_STYLE
    scope_title = BOOK_TITLES[RouteKind.DEEP.value]


class BothAgent(AnswerAgent):
    """低置信/显式对比：合并两库结果。"""

    route = RouteKind.BOTH
    style = _BOTH_STYLE
    scope_title = "两本参考书"

    def _fallback_answer(
        self, citations: Sequence[Citation], *, citation_failure: bool = False
    ) -> str:
        design = [c for c in citations if c.book == RouteKind.DESIGN.value]
        deep = [c for c in citations if c.book == RouteKind.DEEP.value]
        if design and deep:
            note = "两本书均有相关内容："
        elif design:
            note = f"本轮仅检索到《{BOOK_TITLES[RouteKind.DESIGN.value]}》的相关片段："
        elif deep:
            note = f"本轮仅检索到《{BOOK_TITLES[RouteKind.DEEP.value]}》的相关片段："
        else:
            note = ""
        header = _CITATION_FALLBACK_HEADER if citation_failure else _FALLBACK_HEADER
        footer = _CITATION_FALLBACK_FOOTER if citation_failure else _FALLBACK_FOOTER
        lines = [header, "", note, ""] if note else [header, ""]
        for book, group in (
            (BOOK_TITLES[RouteKind.DESIGN.value], design),
            (BOOK_TITLES[RouteKind.DEEP.value], deep),
        ):
            if not group:
                continue
            lines.append(f"**《{book}》**")
            for cit in group:
                lines.append(
                    f"- [{cit.index}] {cit.label()}：{clean_text(cit.excerpt)}"
                )
            lines.append("")
        lines.append(footer)
        return "\n".join(lines)

    def _no_evidence_answer(self, _question: str) -> str:
        return _NO_EVIDENCE_ANSWER


_AGENT_TYPES: dict[RouteKind, type[AnswerAgent]] = {
    RouteKind.DESIGN: DesignPatternAgent,
    RouteKind.DEEP: DeepUnderstandingAgent,
    RouteKind.BOTH: BothAgent,
}

_agents: dict[RouteKind, AnswerAgent] = {}
_agents_lock = threading.Lock()


def _coerce_route(route: RouteKind | str) -> RouteKind:
    if isinstance(route, RouteKind):
        return route
    try:
        return RouteKind(str(route).strip().lower())
    except ValueError:
        logger.warning("未知 route=%r，使用 BothAgent", route)
        return RouteKind.BOTH


def get_agent(route: RouteKind | str) -> AnswerAgent:
    """按路由返回对应 Agent（进程内复用实例）。"""
    route_kind = _coerce_route(route)
    agent = _agents.get(route_kind)
    if agent is None:
        with _agents_lock:
            agent = _agents.get(route_kind)
            if agent is None:
                agent = _AGENT_TYPES[route_kind]()
                _agents[route_kind] = agent
    return agent


def reset_agents() -> None:
    """清空 Agent 单例（测试用）。"""
    with _agents_lock:
        _agents.clear()


def get_agent_model_name(agent: AnswerAgent | None = None) -> str:
    """便捷函数：返回 Agent 当前使用的模型名（未配置 LLM 时为空串）。"""
    try:
        return (agent or get_agent(RouteKind.BOTH)).model_name
    except Exception:  # noqa: BLE001 - 仅用于日志展示
        return ""
