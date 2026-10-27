# -*- coding: utf-8 -*-
"""OpenAI 兼容 Chat Completions 封装。

设计要点：

- **无 key 也能跑**：未配置 ``LLM_API_KEY`` 时 ``is_available()`` 为 False，
  ``chat()`` 抛 :class:`LlmUnavailable`；调用方（router / agents）据此降级到
  规则路由与「检索片段摘要式」作答，绝不产生真实网络调用。
- **可插拔**：``base_url`` / ``model`` / ``api_key`` 全部来自 :mod:`app.config`，
  DeepSeek、DashScope Qwen 等任意 OpenAI 兼容端点均可切换。
- **重试与超时**：tenacity 指数退避（次数取 ``settings.llm_max_retries``），
  单次调用超时取 ``settings.llm_timeout``。
- **JSON 模式**：路由判定使用 ``response_format={"type": "json_object"}``，
  并提供宽松的 JSON 解析（容忍 ```json 代码块 / 前后缀文本）。
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, Callable, Iterator, TypeVar

from app.config import settings
from app.logging_conf import get_logger

logger = get_logger("rag.llm")

_T = TypeVar("_T")


class LlmUnavailable(RuntimeError):
    """LLM 不可用（未配置 key / 网络失败 / 返回非法）。调用方必须降级而非崩溃。"""


# --------------------------------------------------------------------------- #
# 可选依赖：openai SDK / tenacity
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - 依赖已安装时必定成功
    import openai  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    openai = None  # type: ignore
    _HAS_OPENAI = False

try:  # pragma: no cover - 依赖已安装时必定成功
    from tenacity import (
        Retrying,
        retry_if_exception_type,
        retry_if_not_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    _HAS_TENACITY = True
except ImportError:  # pragma: no cover
    _HAS_TENACITY = False


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型输出里尽力解析一个 JSON 对象；失败返回 ``None``（不抛异常）。"""
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        data = None
    if isinstance(data, dict):
        return data
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _response_text(resp: Any) -> str:
    """从 ChatCompletion 响应里取文本，兼容 content 为字符串列表的情况。

    注意：deepseek-flash 等推理模型会先输出 ``reasoning_content`` 再输出
    ``content``；若 ``max_tokens`` 不足会 ``finish_reason=length`` 且
    ``content`` 为空，这里转成带提示的 :class:`LlmUnavailable` 供调用方降级。
    """
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise LlmUnavailable("LLM 返回空 choices")
    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        logger.debug("LLM reasoning_content=%d 字符（不参与引用）", len(str(reasoning)))
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):  # 多模态风格的分段 content
        parts: list[str] = []
        for part in content:
            piece = getattr(part, "text", None)
            if piece is None and isinstance(part, dict):
                piece = part.get("text")
            if piece:
                parts.append(str(piece))
        text = "".join(parts)
    else:
        text = ""
    if not text.strip():
        detail = f"（finish_reason={finish_reason}）" if finish_reason else ""
        if finish_reason == "length":
            raise LlmUnavailable(
                "LLM 返回空内容（推理模型耗尽 max_tokens，finish_reason=length，"
                "请调大 llm_max_tokens / llm_router_max_tokens）"
            )
        raise LlmUnavailable(f"LLM 返回空内容{detail}")
    usage = getattr(resp, "usage", None)
    if usage is not None:
        logger.debug(
            "LLM usage prompt=%s completion=%s",
            getattr(usage, "prompt_tokens", "?"),
            getattr(usage, "completion_tokens", "?"),
        )
    return text


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
class LlmClient:
    """OpenAI 兼容 Chat Completions 客户端（懒创建底层 SDK client）。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else settings.llm_api_key) or ""
        self.base_url = (base_url if base_url is not None else settings.llm_base_url) or ""
        self.model = (model if model is not None else settings.llm_model) or ""
        self.timeout = float(timeout if timeout is not None else settings.llm_timeout)
        self.max_retries = int(
            max_retries if max_retries is not None else settings.llm_max_retries
        )
        self._client: Any | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 可用性 / 元信息
    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        """是否配置了可用的 LLM endpoint（只检查配置，不发起网络请求）。"""
        return bool(self.api_key and self.base_url and self.model)

    @property
    def name(self) -> str:
        """审计日志里使用的模型标识。"""
        return self.model or ""

    # ------------------------------------------------------------------ #
    # 底层
    # ------------------------------------------------------------------ #
    def _get_client(self) -> Any:
        if not _HAS_OPENAI:
            raise LlmUnavailable("未安装 openai SDK，无法调用 LLM")
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                try:
                    self._client = openai.OpenAI(  # type: ignore[union-attr]
                        api_key=self.api_key,
                        base_url=self.base_url or None,
                        timeout=self.timeout,
                        max_retries=0,  # 重试由本模块统一控制
                    )
                except Exception as exc:  # pragma: no cover - 配置异常
                    raise LlmUnavailable(
                        f"创建 LLM 客户端失败（{type(exc).__name__}）"
                    ) from exc
        return self._client

    def _invoke(self, fn: Callable[[], _T]) -> _T:
        """带指数退避的重试执行；最终失败仍向上抛出原异常。

        只重试网络/服务端等瞬时故障：``LlmUnavailable``（未配置、返回空内容等
        确定性失败）不重试，避免推理模型截断时白白等待多次退避。
        """
        attempts = max(1, self.max_retries + 1)

        def _is_retryable(exc: BaseException) -> bool:
            return not isinstance(exc, LlmUnavailable)

        if _HAS_TENACITY:
            retryer: Callable[[Callable[[], _T]], _T] = Retrying(
                retry=retry_if_exception_type(Exception)
                & retry_if_not_exception_type(LlmUnavailable),
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
                reraise=True,
            )
            return retryer(fn)
        last_exc: Exception | None = None  # pragma: no cover - 兜底路径
        for _ in range(attempts):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover
                if not _is_retryable(exc):
                    raise
                last_exc = exc
        assert last_exc is not None
        raise last_exc  # pragma: no cover

    # ------------------------------------------------------------------ #
    # 公共 API
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """调用 Chat Completions，返回纯文本。

        Args:
            max_tokens: 输出 token 上限；None 时取 ``settings.llm_max_tokens``
                （推理模型会先消耗 reasoning tokens，需留足余量）。

        Raises:
            LlmUnavailable: 未配置 key、网络失败、返回为空等任何不可用情形。
        """
        if not self.is_available():
            raise LlmUnavailable("未配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL")
        if not messages:
            raise LlmUnavailable("messages 不能为空")

        # 运行期读 settings，避免函数默认参数在 import 时固化过期值（也便于测试 monkeypatch）
        resolved_max_tokens = int(
            max_tokens if max_tokens is not None else settings.llm_max_tokens
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                settings.llm_temperature if temperature is None else float(temperature)
            ),
            "max_tokens": resolved_max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        client = self._get_client()

        def _call() -> str:
            resp = client.chat.completions.create(**kwargs)
            return _response_text(resp)

        try:
            return self._invoke(_call)
        except LlmUnavailable:
            raise
        except Exception as exc:
            logger.warning("LLM 调用失败: %s: %s", type(exc).__name__, exc)
            raise LlmUnavailable(
                f"LLM 调用失败（{type(exc).__name__}）"
            ) from exc

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield actual answer deltas; reasoning_content is never exposed as citations."""
        if not self.is_available() or not messages:
            raise LlmUnavailable("LLM 未配置或 messages 为空")
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else float(temperature),
            "max_tokens": int(max_tokens if max_tokens is not None else settings.llm_max_tokens),
            "stream": True,
        }
        try:
            stream = self._invoke(lambda: client.chat.completions.create(**kwargs))
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None)
                if isinstance(content, str) and content:
                    yield content
                elif isinstance(content, list):
                    for part in content:
                        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                        if text:
                            yield str(text)
        except LlmUnavailable:
            raise
        except Exception as exc:
            raise LlmUnavailable(f"LLM 流式调用失败（{type(exc).__name__}）") from exc

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """强制 JSON 模式并解析为 dict；解析失败抛 :class:`LlmUnavailable`。

        ``max_tokens`` 为 None 时取 ``settings.llm_router_max_tokens``
        （推理模型先输出 reasoning_content，256 这类小值会导致 JSON 为空）。
        """
        resolved_max_tokens = int(
            max_tokens if max_tokens is not None else settings.llm_router_max_tokens
        )
        text = self.chat(
            messages,
            json_mode=True,
            temperature=temperature,
            max_tokens=resolved_max_tokens,
        )
        data = extract_json_object(text)
        if data is None:
            raise LlmUnavailable("LLM 未返回合法 JSON 对象")
        return data


# 语义化别名（外部按需引用）
OpenAICompatibleLlm = LlmClient


_llm: LlmClient | None = None
_llm_lock = threading.Lock()


def get_llm() -> LlmClient:
    """进程内单例（懒加载，配置只读一次）。"""
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                _llm = LlmClient()
    return _llm


def reset_llm() -> None:
    """丢弃单例（测试 / 运行期切换配置后使用）。"""
    global _llm
    with _llm_lock:
        _llm = None


__all__ = [
    "LlmClient",
    "LlmUnavailable",
    "OpenAICompatibleLlm",
    "extract_json_object",
    "get_llm",
    "reset_llm",
]
