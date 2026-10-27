# -*- coding: utf-8 -*-
"""中心配置：环境变量 / .env / 路径 / 阈值。

变量名保持与《设计方案.txt》一致，便于直接粘贴已有配置：
  EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL, EMBEDDING_DIMENSION
  SYMBOL_AGENT_MILVUS_URI, SYMBOL_AGENT_MILVUS_TOKEN
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 路径 ----------
    data_dir: Path = BASE_DIR / "data"
    logs_dir: Path = BASE_DIR / "logs"
    docs_dir: Path = BASE_DIR / "docs"

    # ---------- 原始语料 ----------
    design_pdf: Path = BASE_DIR / "Agent设计模式-zh-CN.pdf"
    deep_pdf: Path = BASE_DIR / "深入理解 AI Agent-zh-CN.pdf"

    # ---------- Embedding（DashScope / OpenAI 兼容） ----------
    embedding_api_key: str = Field(
        default="", validation_alias=AliasChoices("EMBEDDING_API_KEY", "DASHSCOPE_API_KEY")
    )
    embedding_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("EMBEDDING_BASE_URL", "DASHSCOPE_BASE_URL"),
    )
    embedding_model: str = Field(default="text-embedding-v4", validation_alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1024, validation_alias="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=10, validation_alias="EMBEDDING_BATCH_SIZE")
    embedding_timeout: float = 60.0
    embedding_max_retries: int = 5
    # 无 key 时用确定性桩向量跑通链路（仅开发/测试，检索语义无效）
    embedding_mock: bool = Field(default=False, validation_alias="EMBEDDING_MOCK")

    # ---------- Milvus ----------
    # 连接信息只从环境变量读取；仓库里不保存真实地址与凭据。
    milvus_uri: str = Field(
        default="http://127.0.0.1:19530",
        validation_alias=AliasChoices("SYMBOL_AGENT_MILVUS_URI", "MILVUS_URI"),
    )
    milvus_token: str = Field(
        default="",
        validation_alias=AliasChoices("SYMBOL_AGENT_MILVUS_TOKEN", "MILVUS_TOKEN"),
    )
    milvus_collection: str = Field(default="agent_design_kb", validation_alias="MILVUS_COLLECTION")
    milvus_timeout: float = 30.0

    # ---------- 运维接口 ----------
    admin_api_key: str = Field(default="", validation_alias="ADMIN_API_KEY")

    # ---------- LLM（意图路由 + 答案生成） ----------
    llm_api_key: str = Field(
        default="", validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")
    )
    llm_base_url: str = Field(
        default="https://api.deepseek.com", validation_alias=AliasChoices("LLM_BASE_URL", "DEEPSEEK_BASE_URL")
    )
    llm_model: str = Field(default="deepseek-chat", validation_alias=AliasChoices("LLM_MODEL", "DEEPSEEK_MODEL"))
    llm_temperature: float = 0.2
    llm_timeout: float = 90.0
    llm_max_retries: int = 3
    # deepseek-flash 这类推理模型会先消耗 reasoning tokens，max_tokens 需留足余量
    llm_max_tokens: int = Field(default=2048, validation_alias="LLM_MAX_TOKENS")
    llm_router_max_tokens: int = Field(default=2048, validation_alias="LLM_ROUTER_MAX_TOKENS")

    # ---------- 检索 / 路由 ----------
    top_k: int = Field(default=8, validation_alias="RAG_TOP_K")
    max_per_chapter: int = 3
    neighbor_expand: int = 1
    rrf_k: int = 60
    route_confidence_single: float = 0.60
    route_confidence_both: float = 0.35

    # ---------- 服务 ----------
    host: str = "127.0.0.1"
    port: int = 8000

    # ---------- 派生属性 ----------
    @property
    def embedding_enabled(self) -> bool:
        return bool(self.embedding_api_key) or self.embedding_mock

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def pdf_paths(self) -> dict[str, Path]:
        return {"design": self.design_pdf, "deep": self.deep_pdf}

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.logs_dir, self.docs_dir, self.data_dir / "pages"):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
