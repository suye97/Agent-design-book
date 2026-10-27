# -*- coding: utf-8 -*-
"""双库 Agent 单测：LLM 生成、无 LLM 降级、引用校验与出处列表。

全部使用 Fake LLM，禁止真实网络调用。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.config import settings
from app.models import BOOK_TITLES, BookKey, ChunkType, Citation, RouteKind
from app.rag.agents import (
    BothAgent,
    DeepUnderstandingAgent,
    DesignPatternAgent,
    build_citations_markdown,
    get_agent,
    reset_agents,
    verify_citations,
)
from app.rag.llm import LlmUnavailable
from app.rag.router import route_question

DESIGN_TITLE = BOOK_TITLES[BookKey.DESIGN.value]
DEEP_TITLE = BOOK_TITLES[BookKey.DEEP.value]

DESIGN_EXCERPT = "工具使用模式让模型在推理过程中调用外部函数，从而把语言能力延伸到真实世界。"
DEEP_EXCERPT = "工具调用的本质是把模型输出映射为可执行动作，其机制依赖上下文与反馈闭环。"


# --------------------------------------------------------------------------- #
# Fakes / builders
# --------------------------------------------------------------------------- #
class FakeLlm:
    name = "fake-model"

    def __init__(
        self,
        reply: str = "",
        error: Exception | None = None,
        available: bool = True,
        stream_parts: list[str] | None = None,
        replies: list[str] | None = None,
    ):
        self.reply = reply
        self.error = error
        self.available = available
        self.stream_parts = stream_parts
        self.replies = list(replies or [])
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return self.available

    def chat(self, messages, json_mode=False, temperature=None, max_tokens=1024) -> str:  # noqa: ANN001
        self.calls.append(
            {
                "messages": messages,
                "json_mode": json_mode,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.error is not None:
            raise self.error
        return self.replies.pop(0) if self.replies else self.reply

    def chat_stream(self, messages, temperature=None, max_tokens=1024):  # noqa: ANN001
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        if self.error is not None:
            raise self.error
        yield from (self.stream_parts if self.stream_parts is not None else [self.reply])


def consume_stream(stream) -> tuple[str, bool]:  # noqa: ANN001
    """同时保留生成器的文本增量和 used_llm 返回值。"""
    parts: list[str] = []
    while True:
        try:
            parts.append(next(stream))
        except StopIteration as stop:
            return "".join(parts), bool(stop.value)


def make_citation(
    index: int = 1,
    book: str = "design",
    *,
    chunk_id: str | None = None,
    chapter_no: int = 5,
    chapter_title: str = "工具使用（函数调用）",
    page: int = 67,
    page_pdf: int = 80,
    para_start: int = 3,
    para_end: int = 3,
    excerpt: str = DESIGN_EXCERPT,
    score: float = 0.9,
    content: str = "",
) -> Citation:
    return Citation(
        index=index,
        chunk_id=chunk_id or f"{book}-{index}",
        book=book,
        book_title=DESIGN_TITLE if book == "design" else DEEP_TITLE,
        chapter_no=chapter_no,
        chapter_title=chapter_title,
        section_no="5.2",
        section_title="工具调用流程",
        page=page,
        page_pdf=page_pdf,
        para_start=para_start,
        para_end=para_end,
        chunk_type=ChunkType.TEXT,
        excerpt=excerpt,
        score=score,
        rank=index,
        content=content,
    )


@pytest.fixture(autouse=True)
def _reset_agent_singletons():
    reset_agents()
    yield
    reset_agents()


# --------------------------------------------------------------------------- #
# LLM 生成路径
# --------------------------------------------------------------------------- #
def test_design_agent_with_llm() -> None:
    fake = FakeLlm(reply="工具使用模式允许模型调用外部函数[1]，并需要处理幂等与错误[2]。")
    citations = [
        make_citation(1, excerpt=DESIGN_EXCERPT),
        make_citation(2, excerpt=DESIGN_EXCERPT[10:]),
    ]
    agent = DesignPatternAgent(llm=fake)

    answer, used_llm = agent.answer("工具使用模式怎么用？", citations)

    assert used_llm is True
    assert answer.startswith("工具使用模式允许模型调用外部函数[1]")
    assert "**引用出处**" in answer
    assert f"[1] 《{DESIGN_TITLE}》第 5 章 工具使用（函数调用） · 第 67 页 · 第 3 段" in answer
    assert verify_citations(answer, citations) is True

    # prompt 约束：只依据资料 + 资料片段编号 + 设计模式书结构
    system_prompt = fake.calls[0]["messages"][0]["content"]
    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "只能依据" in system_prompt
    assert "资料不足" in system_prompt or "未覆盖" in system_prompt
    assert "模式定位" in system_prompt
    assert "[1]" in user_prompt
    assert "工具使用模式怎么用？" in user_prompt
    assert fake.calls[0]["temperature"] is None
    # 推理模型（deepseek-flash）会先消耗 reasoning tokens，取 settings.llm_max_tokens
    assert fake.calls[0]["max_tokens"] == settings.llm_max_tokens
    assert len(fake.calls) == 1  # 首次引用合规时不产生额外请求


def test_answer_repairs_missing_citation_once_before_fallback() -> None:
    original = "压缩可以节省 token[1]。\n\n**3. 压缩率并非越小越好。** 压得太狠会丢信息或引发碎片化。"
    repaired = "压缩可以节省 token[1]。\n\n**3. 压缩率并非越小越好。** 压得太狠会丢信息或引发碎片化[1]。"
    fake = FakeLlm(replies=[original, repaired])

    answer, used_llm = DesignPatternAgent(llm=fake).answer("压缩率怎么选？", [make_citation()])

    assert used_llm is True
    assert answer.startswith(repaired)
    assert "未通过引用校验" not in answer
    assert len(fake.calls) == 2
    repair_messages = fake.calls[1]["messages"]
    assert repair_messages[-2] == {"role": "assistant", "content": original}
    assert "段落或列表项缺少引用" in repair_messages[-1]["content"]
    assert "只能使用原资料" in repair_messages[-1]["content"]


def test_answer_uses_current_settings_llm_max_tokens(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_max_tokens", 3072)
    fake = FakeLlm(reply="工具使用模式允许模型调用外部函数[1]。")
    citations = [make_citation(1, excerpt=DESIGN_EXCERPT)]

    DesignPatternAgent(llm=fake).answer("工具使用模式怎么用？", citations)

    assert fake.calls[0]["max_tokens"] == 3072


def test_fallback_when_reasoning_model_content_is_empty() -> None:
    # 推理模型耗尽 max_tokens：content 为空，LlmClient 抛 LlmUnavailable → 摘要式降级
    fake = FakeLlm(
        error=LlmUnavailable(
            "LLM 返回空内容（推理模型耗尽 max_tokens，finish_reason=length）"
        )
    )
    citations = [make_citation(1, excerpt=DESIGN_EXCERPT)]

    answer, used_llm = DesignPatternAgent(llm=fake).answer("工具使用模式怎么用？", citations)

    assert used_llm is False
    assert "大模型不可用" in answer
    assert "**引用出处**" in answer
    assert verify_citations(answer, citations) is True


def test_prompt_uses_full_content_when_available() -> None:
    detail = "这是只有全文里才出现的细节：幂等键必须由调用方提供。"
    full_content = DESIGN_EXCERPT + detail + "补充上下文说明。" * 60
    citation = make_citation(1, excerpt=DESIGN_EXCERPT, content=full_content)
    fake = FakeLlm(reply="依据全文作答[1]。")

    DesignPatternAgent(llm=fake).answer("工具使用模式怎么用？", [citation])

    user_prompt = fake.calls[0]["messages"][1]["content"]
    # excerpt 只有前 40 字，detail 只存在于 content，能出现在 prompt 才说明用了全文
    assert detail not in citation.excerpt
    assert detail in user_prompt
    assert full_content in user_prompt
    assert "[1]" in user_prompt
    assert "第 67 页" in user_prompt  # 保留书/章/页/段定位


def test_prompt_falls_back_to_excerpt_when_content_empty() -> None:
    citation = make_citation(1, excerpt=DESIGN_EXCERPT)  # content 默认 ""
    fake = FakeLlm(reply="依据摘录作答[1]。")

    DesignPatternAgent(llm=fake).answer("工具使用模式怎么用？", [citation])

    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert DESIGN_EXCERPT in user_prompt


def test_prompt_truncates_long_content() -> None:
    long_content = "首" + "长" * 2000
    citation = make_citation(1, excerpt=DESIGN_EXCERPT, content=long_content)
    fake = FakeLlm(reply="作答[1]。")

    DesignPatternAgent(llm=fake).answer("工具使用模式怎么用？", [citation])

    user_prompt = fake.calls[0]["messages"][1]["content"]
    prompt_content = user_prompt.split("原文：", 1)[1].split("\n\n【问题】", 1)[0]
    assert prompt_content.endswith("…")
    assert len(prompt_content) <= 801  # 800 字 + 省略号


def test_deep_agent_prompt_style() -> None:
    fake = FakeLlm(reply="从机制上看，工具调用把模型输出映射为动作[1]。")
    citations = [
        make_citation(
            1,
            book="deep",
            chapter_no=3,
            chapter_title="上下文工程",
            page=55,
            excerpt=DEEP_EXCERPT,
        )
    ]
    agent = DeepUnderstandingAgent(llm=fake)

    answer, used_llm = agent.answer("工具调用的底层机制是什么？", citations)

    assert agent.route is RouteKind.DEEP
    assert used_llm is True
    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "机制拆解" in system_prompt
    assert DEEP_TITLE in system_prompt
    assert f"《{DEEP_TITLE}》" in answer


def test_both_agent_prompt_asks_to_compare_books() -> None:
    fake = FakeLlm(reply="设计模式书给出模式定位[1]；深入理解书解释其原理[2]。")
    citations = [
        make_citation(1, book="design"),
        make_citation(2, book="deep", chunk_id="deep-2", excerpt=DEEP_EXCERPT),
    ]
    agent = BothAgent(llm=fake)

    answer, used_llm = agent.answer("工具调用模式与原理", citations)

    assert used_llm is True
    system_prompt = fake.calls[0]["messages"][0]["content"]
    assert "分别说明" in system_prompt
    assert DESIGN_TITLE in system_prompt
    assert DEEP_TITLE in system_prompt
    assert verify_citations(answer, citations) is True


# --------------------------------------------------------------------------- #
# 无 LLM 降级路径
# --------------------------------------------------------------------------- #
def test_fallback_when_llm_unavailable() -> None:
    fake = FakeLlm(error=LlmUnavailable("未配置 key"))
    citations = [
        make_citation(1, excerpt=DESIGN_EXCERPT),
        make_citation(2, excerpt=DESIGN_EXCERPT),
    ]
    agent = DesignPatternAgent(llm=fake)

    answer, used_llm = agent.answer("工具使用模式怎么用？", citations)

    assert used_llm is False
    assert "大模型不可用" in answer
    assert "[1]" in answer
    assert DESIGN_EXCERPT in answer
    assert "**引用出处**" in answer
    assert verify_citations(answer, citations) is True
    # 降级答案不编造结论
    assert "直接摘自参考书，不做推断" in answer


def test_fallback_when_llm_not_available_flag() -> None:
    agent = DesignPatternAgent(llm=FakeLlm(available=False))
    citations = [make_citation(1, excerpt=DESIGN_EXCERPT)]

    answer, used_llm = agent.answer("问题", citations)

    assert used_llm is False
    assert "大模型不可用" in answer
    assert verify_citations(answer, citations) is True


def test_fallback_when_llm_raises_generic_error() -> None:
    agent = DesignPatternAgent(llm=FakeLlm(error=RuntimeError("gateway 502")))
    citations = [make_citation(1, excerpt=DESIGN_EXCERPT)]

    answer, used_llm = agent.answer("问题", citations)

    assert used_llm is False
    assert "大模型不可用" in answer
    assert verify_citations(answer, citations) is True


def test_fallback_when_llm_returns_empty() -> None:
    agent = DeepUnderstandingAgent(llm=FakeLlm(reply="   "))
    citations = [make_citation(1, book="deep", chunk_id="d1", excerpt=DEEP_EXCERPT)]

    answer, used_llm = agent.answer("为什么这样设计？", citations)

    assert used_llm is False
    assert "大模型不可用" in answer
    assert verify_citations(answer, citations) is True


def test_both_agent_fallback_mentions_both_books() -> None:
    agent = BothAgent(llm=FakeLlm(error=LlmUnavailable("no key")))
    citations = [
        make_citation(1, book="design", excerpt=DESIGN_EXCERPT),
        make_citation(2, book="deep", chunk_id="d1", excerpt=DEEP_EXCERPT),
    ]

    answer, used_llm = agent.answer("两本书怎么看待工具调用", citations)

    assert used_llm is False
    assert "两本书均有相关内容" in answer
    assert DESIGN_TITLE in answer
    assert DEEP_TITLE in answer
    assert verify_citations(answer, citations) is True


def test_no_citations_answer_says_not_covered() -> None:
    agent = DesignPatternAgent(llm=FakeLlm(reply="不该被调用"))

    answer, used_llm = agent.answer("完全未知的问题", [])

    assert used_llm is False
    assert "现有资料未覆盖该问题" in answer
    assert "**引用出处**" not in answer
    assert verify_citations(answer, []) is False


# --------------------------------------------------------------------------- #
# 引用校验后处理
# --------------------------------------------------------------------------- #
def test_llm_answer_with_out_of_range_marker_falls_back() -> None:
    agent = DesignPatternAgent(llm=FakeLlm(reply="回答覆盖不足[9]。"))
    citations = [make_citation(1)]

    answer, used_llm = agent.answer("工具使用模式", citations)

    assert used_llm is False
    assert "生成答案未通过引用校验" in answer
    assert "当前大模型不可用" not in answer
    assert "请配置 LLM_API_KEY" not in answer
    assert "直接摘自参考书" in answer
    assert "回答覆盖不足" not in answer
    assert verify_citations(answer, citations) is True


def test_llm_answer_without_markers_falls_back() -> None:
    fake = FakeLlm(replies=["这是一个没有引用标记的回答。", "重写后仍没有引用。"])
    agent = DesignPatternAgent(llm=fake)
    citations = [make_citation(1)]

    answer, used_llm = agent.answer("工具使用模式", citations)

    assert len(fake.calls) == 2  # 最多自动修复一次
    assert used_llm is False
    assert "直接摘自参考书" in answer
    assert "没有引用标记的回答" not in answer
    assert verify_citations(answer, citations) is True


def test_valid_llm_answer_has_no_warning() -> None:
    agent = DesignPatternAgent(llm=FakeLlm(reply="工具使用模式让模型调用外部函数[1]。"))
    citations = [make_citation(1)]

    answer, _ = agent.answer("工具使用模式", citations)

    assert "提示：" not in answer


# --------------------------------------------------------------------------- #
# 真实模型增量输出与降级
# --------------------------------------------------------------------------- #
def test_stream_answer_preserves_deltas_and_used_llm_return() -> None:
    fake = FakeLlm(stream_parts=["工具使用模式", "允许调用外部函数[1]。"])
    agent = DesignPatternAgent(llm=fake)

    answer, used_llm = consume_stream(agent.stream_answer("怎么用？", [make_citation()]))

    assert answer == "工具使用模式允许调用外部函数[1]。"
    assert used_llm is True
    assert fake.calls[0]["max_tokens"] == settings.llm_max_tokens


def test_stream_answer_falls_back_without_leaking_exception() -> None:
    agent = DesignPatternAgent(llm=FakeLlm(error=LlmUnavailable("offline")))

    answer, used_llm = consume_stream(agent.stream_answer("怎么用？", [make_citation()]))

    assert used_llm is False
    assert "直接摘自参考书" in answer
    assert "offline" not in answer


# --------------------------------------------------------------------------- #
# 公共工具 / 工厂
# --------------------------------------------------------------------------- #
def test_build_citations_markdown_reexported() -> None:
    citations = [
        make_citation(1),
        make_citation(
            2,
            book="deep",
            chunk_id="d2",
            chapter_no=3,
            chapter_title="上下文工程",
            page=55,
            para_start=2,
            para_end=4,
            excerpt=DEEP_EXCERPT,
        ),
    ]
    markdown = build_citations_markdown(citations)

    assert "[1] 《" in markdown
    assert "[2] 《" in markdown
    assert "第 2–4 段" in markdown


def test_get_agent_returns_matching_agent_types() -> None:
    assert isinstance(get_agent(RouteKind.DESIGN), DesignPatternAgent)
    assert isinstance(get_agent(RouteKind.DEEP), DeepUnderstandingAgent)
    assert isinstance(get_agent(RouteKind.BOTH), BothAgent)
    assert isinstance(get_agent("design"), DesignPatternAgent)
    # 单例复用
    assert get_agent(RouteKind.DESIGN) is get_agent("design")
    assert isinstance(get_agent("unknown-route"), BothAgent)


def test_agent_model_name() -> None:
    agent = DesignPatternAgent(llm=FakeLlm())
    assert agent.model_name == "fake-model"


# --------------------------------------------------------------------------- #
# 离线端到端（无 LLM、无真实检索）
# --------------------------------------------------------------------------- #
def test_offline_pipeline_route_then_answer() -> None:
    decision = route_question("什么是提示链？", llm=FakeLlm(error=LlmUnavailable("no key")))
    assert decision.route is RouteKind.DESIGN

    citations = [make_citation(1, excerpt=DESIGN_EXCERPT)]
    agent = DesignPatternAgent(llm=FakeLlm(error=LlmUnavailable("no key")))

    answer, used_llm = agent.answer("什么是提示链？", citations)

    assert used_llm is False
    assert "**引用出处**" in answer
    assert verify_citations(answer, citations) is True
