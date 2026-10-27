# -*- coding: utf-8 -*-
"""意图路由单测：显式覆盖 / LLM JSON 判定 / 规则兜底 / 置信度阈值。

全部使用 Fake LLM，禁止真实网络调用。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.config import settings
from app.models import BOOK_TITLES, BookKey, Intent, RouteKind, RouterSource
from app.rag.llm import LlmClient, LlmUnavailable
from app.rag.router import CLARIFY_HINT, _GUARD_CONFIDENCE, route_question

DESIGN_TITLE = BOOK_TITLES[BookKey.DESIGN.value]
DEEP_TITLE = BOOK_TITLES[BookKey.DEEP.value]


# --------------------------------------------------------------------------- #
# Fake LLM
# --------------------------------------------------------------------------- #
class FakeLlm:
    """可编程的假 LLM：返回固定 JSON、抛异常或标记不可用。"""

    name = "fake-llm"

    def __init__(self, payload=None, error: Exception | None = None, available: bool = True):
        self.payload = payload
        self.error = error
        self.available = available
        self.calls: list[list[dict[str, str]]] = []
        self.chat_json_kwargs: list[dict] = []

    def is_available(self) -> bool:
        return self.available

    def chat(self, messages, json_mode=False, temperature=None, max_tokens=None) -> str:  # noqa: ANN001
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        import json

        return json.dumps(self.payload, ensure_ascii=False)

    def chat_json(self, messages, temperature=0.0, max_tokens=None):  # noqa: ANN001
        self.calls.append(messages)
        self.chat_json_kwargs.append(
            {"temperature": temperature, "max_tokens": max_tokens}
        )
        if self.error is not None:
            raise self.error
        return self.payload


def rule_only_llm() -> FakeLlm:
    return FakeLlm(error=LlmUnavailable("未配置 key"))


def llm_payload(intent: str, confidence: float, **extra) -> dict:  # noqa: ANN003
    payload = {
        "intent": intent,
        "confidence": confidence,
        "book_hint": extra.get("book_hint"),
        "rewritten_query": extra.get("rewritten_query", ""),
        "reason": extra.get("reason", "fake 理由"),
    }
    return payload


# --------------------------------------------------------------------------- #
# 8 条路由用例（问题 → route / source）
# --------------------------------------------------------------------------- #
ROUTE_CASES = [
    ("《Agent设计模式》里工具使用模式怎么写代码？", RouteKind.DESIGN, RouterSource.OVERRIDE),
    ("深入理解那本书怎么解释护栏的演进？", RouteKind.DEEP, RouterSource.OVERRIDE),
    ("两本书都看一下记忆管理", RouteKind.BOTH, RouterSource.OVERRIDE),
    ("提示链、路由、并行化这些设计模式分别是什么？", RouteKind.DESIGN, RouterSource.RULE),
    ("多智能体协调机制的底层原理是什么？", RouteKind.DEEP, RouterSource.RULE),
    ("反思模式怎么用，有代码示例吗", RouteKind.DESIGN, RouterSource.RULE),
    ("后训练与强化学习的区别和权衡", RouteKind.DEEP, RouterSource.RULE),
    ("Agent 是什么？", RouteKind.BOTH, RouterSource.RULE),
]


@pytest.mark.parametrize("question,expected_route,expected_source", ROUTE_CASES)
def test_route_cases(question: str, expected_route: RouteKind, expected_source: RouterSource) -> None:
    decision = route_question(question, llm=rule_only_llm())

    assert decision.route is expected_route
    assert decision.source is expected_source
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.reason.strip(), "路由结果必须可解释（reason 非空）"


# --------------------------------------------------------------------------- #
# 显式指令覆盖
# --------------------------------------------------------------------------- #
def test_explicit_design_override() -> None:
    decision = route_question("《Agent设计模式》里工具使用模式怎么写代码？", llm=rule_only_llm())

    assert decision.route is RouteKind.DESIGN
    assert decision.source is RouterSource.OVERRIDE
    assert decision.confidence == pytest.approx(1.0)
    assert decision.intent is Intent.CONCEPT
    assert "工具使用模式" in decision.rewritten_query  # 去掉书名指代
    assert DESIGN_TITLE in decision.reason


def test_explicit_deep_override() -> None:
    decision = route_question("深入理解那本书怎么解释护栏的演进？", llm=rule_only_llm())

    assert decision.route is RouteKind.DEEP
    assert decision.source is RouterSource.OVERRIDE
    assert decision.intent is Intent.EXPLAIN
    assert "深入理解那本书" in decision.reason
    assert DEEP_TITLE in decision.reason


def test_explicit_both_override() -> None:
    decision = route_question("两本书都看一下记忆管理", llm=rule_only_llm())

    assert decision.route is RouteKind.BOTH
    assert decision.source is RouterSource.OVERRIDE
    assert decision.intent is Intent.AMBIGUOUS
    assert "两本书" in decision.reason or "两本" in decision.reason


def test_explicit_both_by_naming_two_books() -> None:
    question = "《Agent设计模式》和《深入理解 AI Agent》对 MCP 的说法一致吗"
    decision = route_question(question, llm=rule_only_llm())

    assert decision.route is RouteKind.BOTH
    assert decision.source is RouterSource.OVERRIDE


def test_book_override_argument_has_highest_priority() -> None:
    # 问题文本点名设计模式书，但调用方指定 deep → 以调用方为准
    decision = route_question("《Agent设计模式》里工具使用模式怎么写代码？", book_override="deep")

    assert decision.route is RouteKind.DEEP
    assert decision.source is RouterSource.OVERRIDE
    assert "book_override=deep" in decision.reason


def test_book_override_unknown_value_falls_back_to_rule() -> None:
    decision = route_question("什么是提示链？", book_override="不存在的书", llm=rule_only_llm())

    assert decision.source is RouterSource.RULE
    assert decision.route is RouteKind.DESIGN


# --------------------------------------------------------------------------- #
# 规则兜底与置信度阈值
# --------------------------------------------------------------------------- #
def test_rule_design_high_confidence() -> None:
    decision = route_question("提示链、路由、并行化这些设计模式分别是什么？", llm=rule_only_llm())

    assert decision.route is RouteKind.DESIGN
    assert decision.intent is Intent.CONCEPT
    assert decision.source is RouterSource.RULE
    assert decision.confidence >= settings.route_confidence_single
    assert decision.clarify_hint is None
    assert "设计模式信号" in decision.reason


def test_rule_deep_high_confidence() -> None:
    decision = route_question("后训练与强化学习的区别和权衡", llm=rule_only_llm())

    assert decision.route is RouteKind.DEEP
    assert decision.intent is Intent.EXPLAIN
    assert decision.confidence >= settings.route_confidence_single


def test_rule_low_confidence_both_with_clarify() -> None:
    decision = route_question("Agent 是什么？", llm=rule_only_llm())

    assert decision.route is RouteKind.BOTH
    assert decision.intent is Intent.AMBIGUOUS
    assert decision.confidence < settings.route_confidence_both
    assert decision.clarify_hint == CLARIFY_HINT


def test_rule_mid_confidence_both_without_clarify() -> None:
    decision = route_question("记忆机制是什么", llm=rule_only_llm())

    assert decision.route is RouteKind.BOTH
    assert settings.route_confidence_both <= decision.confidence < settings.route_confidence_single
    assert decision.clarify_hint is None
    assert "双库" in decision.reason


def test_rule_empty_question_returns_both_with_clarify() -> None:
    decision = route_question("   ", llm=rule_only_llm())

    assert decision.route is RouteKind.BOTH
    assert decision.clarify_hint is not None
    assert decision.confidence == 0.0


def test_all_route_decisions_explain_themselves() -> None:
    questions = [
        "提示链怎么用",
        "为什么需要记忆管理",
        "护栏是什么",
        "MCP 与 A2A 的联系",
        "两本书都看评估方法",
        "你好",
    ]
    for question in questions:
        decision = route_question(question, llm=rule_only_llm())
        assert decision.reason.strip(), question
        assert 0.0 <= decision.confidence <= 1.0, question


# --------------------------------------------------------------------------- #
# LLM 结构化判定
# --------------------------------------------------------------------------- #
def test_llm_route_success() -> None:
    fake = FakeLlm(
        payload=llm_payload(
            "concept",
            0.92,
            book_hint="design",
            rewritten_query="提示链怎么用 代码示例",
            reason="询问模式用法与代码示例",
        )
    )
    decision = route_question("提示链怎么用？有代码示例吗", llm=fake)

    assert decision.route is RouteKind.DESIGN
    assert decision.intent is Intent.CONCEPT
    assert decision.source is RouterSource.LLM
    assert decision.confidence == pytest.approx(0.92)
    assert decision.rewritten_query == "提示链怎么用 代码示例"
    assert decision.book_hint == "design"
    assert "LLM 判定" in decision.reason
    assert fake.calls, "应调用 LLM"


def test_llm_route_low_confidence_goes_both_with_clarify() -> None:
    fake = FakeLlm(payload=llm_payload("explain", 0.2, reason="理由"))
    decision = route_question("某个 Agent 问题", llm=fake)

    assert decision.route is RouteKind.BOTH
    assert decision.source is RouterSource.LLM
    assert decision.clarify_hint == CLARIFY_HINT


def test_llm_route_mid_confidence_both_without_clarify() -> None:
    fake = FakeLlm(payload=llm_payload("concept", 0.45, reason="理由"))
    decision = route_question("某个 Agent 问题", llm=fake)

    assert decision.route is RouteKind.BOTH
    assert decision.clarify_hint is None


def test_llm_route_invalid_payload_falls_back_to_rule() -> None:
    fake = FakeLlm(payload={"intent": "banana", "confidence": 0.9})
    decision = route_question("什么是提示链？", llm=fake)

    assert decision.source is RouterSource.RULE
    assert decision.route is RouteKind.DESIGN
    assert "已降级为规则路由" in decision.reason


def test_llm_route_exception_falls_back_to_rule() -> None:
    fake = FakeLlm(error=RuntimeError("boom"))
    decision = route_question("后训练与强化学习的区别和权衡", llm=fake)

    assert decision.source is RouterSource.RULE
    assert decision.route is RouteKind.DEEP
    assert "RuntimeError" in decision.reason


def test_llm_unavailable_falls_back_to_rule() -> None:
    decision = route_question("什么是护栏？", llm=FakeLlm(available=False))

    assert decision.source is RouterSource.RULE
    assert decision.route is RouteKind.DESIGN
    assert "LlmUnavailable" in decision.reason or "不可用" in decision.reason


def test_llm_route_ambiguous_intent_returns_both() -> None:
    fake = FakeLlm(payload=llm_payload("ambiguous", 0.8, reason="两本书都相关"))
    decision = route_question("怎么看待记忆管理", llm=fake)

    assert decision.route is RouteKind.BOTH
    assert decision.intent is Intent.AMBIGUOUS
    assert decision.source is RouterSource.LLM


# --------------------------------------------------------------------------- #
# 无 key 的 LlmClient：必须快速抛 LlmUnavailable，且不发起网络请求
# --------------------------------------------------------------------------- #
def test_llm_client_without_api_key_raises_unavailable() -> None:
    client = LlmClient(api_key="", base_url="https://invalid.example", model="fake-model")

    assert client.is_available() is False
    with pytest.raises(LlmUnavailable):
        client.chat([{"role": "user", "content": "hi"}])
    with pytest.raises(LlmUnavailable):
        client.chat_json([{"role": "user", "content": "hi"}])


# --------------------------------------------------------------------------- #
# 推理模型（deepseek-flash）适配：max_tokens 取 settings；空 content 可降级
# --------------------------------------------------------------------------- #
class _FakeCompletions:
    """模拟 OpenAI SDK 的 chat.completions.create，记录调用参数。"""

    def __init__(self, content: str = "ok", finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=self.content,
                        reasoning_content="推理过程" * 20,  # 推理模型先输出的 reasoning_content
                    ),
                    finish_reason=self.finish_reason,
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )


def _client_with_fake_sdk(content: str = "ok", finish_reason: str = "stop"):
    client = LlmClient(
        api_key="test-key",
        base_url="https://fake.invalid/v1",
        model="deepseek-flash",
    )
    completions = _FakeCompletions(content, finish_reason)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_chat_default_max_tokens_from_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_max_tokens", 4096)
    client, completions = _client_with_fake_sdk("答案[1]")

    assert client.chat([{"role": "user", "content": "hi"}]) == "答案[1]"
    assert completions.last_kwargs is not None
    assert completions.last_kwargs["max_tokens"] == 4096


def test_chat_json_default_max_tokens_from_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_router_max_tokens", 3072)
    client, completions = _client_with_fake_sdk('{"intent":"concept","confidence":0.9}')

    data = client.chat_json([{"role": "user", "content": "hi"}])

    assert data["intent"] == "concept"
    assert completions.last_kwargs is not None
    assert completions.last_kwargs["max_tokens"] == 3072
    assert completions.last_kwargs["response_format"] == {"type": "json_object"}


def test_chat_stream_yields_only_answer_deltas(monkeypatch) -> None:
    class StreamingCompletions:
        def __init__(self) -> None:
            self.last_kwargs = None

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return iter([
                SimpleNamespace(choices=[]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                    content=None, reasoning_content="不应输出的推理"
                ))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="答案"))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                    content=[{"text": "[1]"}, SimpleNamespace(text="。")]
                ))]),
            ])

    monkeypatch.setattr(settings, "llm_max_tokens", 4096)
    client, _ = _client_with_fake_sdk()
    completions = StreamingCompletions()
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    assert list(client.chat_stream([{"role": "user", "content": "hi"}])) == [
        "答案", "[1]", "。",
    ]
    assert completions.last_kwargs["stream"] is True
    assert completions.last_kwargs["max_tokens"] == 4096


def test_chat_stream_wraps_sdk_failure() -> None:
    client, _ = _client_with_fake_sdk()
    client._client.chat.completions.create = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("connection lost")
    )

    with pytest.raises(LlmUnavailable, match="流式调用失败"):
        list(client.chat_stream([{"role": "user", "content": "hi"}]))


def test_reasoning_model_empty_content_raises_unavailable() -> None:
    client, _ = _client_with_fake_sdk("", "length")

    with pytest.raises(LlmUnavailable) as excinfo:
        client.chat([{"role": "user", "content": "hi"}])

    assert "finish_reason=length" in str(excinfo.value)


def test_router_does_not_force_small_router_max_tokens() -> None:
    fake = FakeLlm(payload=llm_payload("concept", 0.9))

    decision = route_question("提示链是什么？", llm=fake)

    assert decision.route is RouteKind.DESIGN
    # 不硬编码小值，交由 LlmClient.chat_json 使用 settings.llm_router_max_tokens
    assert fake.chat_json_kwargs[-1]["max_tokens"] is None


def test_router_falls_back_when_reasoning_model_returns_empty_content() -> None:
    fake = FakeLlm(error=LlmUnavailable("LLM 返回空内容（finish_reason=length）"))

    decision = route_question("什么是提示链？", llm=fake)

    assert decision.source is RouterSource.RULE
    assert decision.route is RouteKind.DESIGN
    assert "已降级为规则路由" in decision.reason


# --------------------------------------------------------------------------- #
# 路由优化回归：主题归属优先 + 深度诉求识别 + LLM 纠偏
# --------------------------------------------------------------------------- #
def test_rule_detail_request_goes_deep_by_topic_and_depth() -> None:
    """“详细了解 + 处理方案 + 仅 deep 系统覆盖的主题”必须走深入理解书。"""
    decision = route_question("我想详细了解上下文压缩的概念及处理方案", llm=rule_only_llm())

    assert decision.route is RouteKind.DEEP
    assert decision.intent is Intent.EXPLAIN
    assert decision.confidence >= settings.route_confidence_single
    assert "上下文压缩" in decision.reason


def test_rule_what_is_deep_only_topic_goes_deep() -> None:
    """“上下文工程是什么”是概念式问法，但主题只有深入理解书系统讲解 → 仍走 deep。"""
    decision = route_question("上下文工程是什么", llm=rule_only_llm())

    assert decision.route is RouteKind.DEEP
    assert decision.confidence >= settings.route_confidence_single


def test_rule_pattern_question_stays_design() -> None:
    """模式名 + 场景/用法类问法仍走设计模式书（不能被“深入”类词带偏）。"""
    for question in (
        "提示链模式适合什么场景？",
        "反思模式怎么用，有代码示例吗？",
        "护栏与安全模式有哪些落地手段？",
    ):
        decision = route_question(question, llm=rule_only_llm())
        assert decision.route is RouteKind.DESIGN, question


def test_llm_wrong_concept_is_corrected_by_topic_signal() -> None:
    """LLM 把“上下文压缩”误判成 concept 时，主题/深度信号要纠偏到 deep。"""
    fake = FakeLlm(
        payload=llm_payload(
            "concept",
            0.86,
            book_hint=None,
            rewritten_query="上下文压缩 概念 处理方案",
            reason="询问概念与方案",
        )
    )
    decision = route_question("我想详细了解上下文压缩的概念及处理方案", llm=fake)

    assert decision.route is RouteKind.DEEP
    assert decision.intent is Intent.EXPLAIN
    assert decision.confidence >= _GUARD_CONFIDENCE
    assert "纠偏" in decision.reason


def test_llm_wrong_explain_is_corrected_by_pattern_signal() -> None:
    """LLM 把“模式怎么用”误判成 explain 时，模式信号要把路由拉回 design。"""
    fake = FakeLlm(
        payload=llm_payload(
            "explain",
            0.9,
            book_hint=None,
            rewritten_query="提示链 代码示例",
            reason="询问实现细节",
        )
    )
    decision = route_question("提示链模式怎么用？有代码示例吗", llm=fake)

    assert decision.route is RouteKind.DESIGN
    assert decision.intent is Intent.CONCEPT
    assert "纠偏" in decision.reason


def test_llm_book_hint_overrides_intent_form() -> None:
    """book_hint 体现主题归属：即便 intent=concept，也应落到 deep。"""
    fake = FakeLlm(
        payload=llm_payload(
            "concept",
            0.9,
            book_hint="deep",
            rewritten_query="上下文工程 定义",
            reason="主题属于深入理解书",
        )
    )
    decision = route_question("上下文工程是什么", llm=fake)

    assert decision.route is RouteKind.DEEP
    assert decision.book_hint == "deep"
    assert "主题归属" in decision.reason
