# -*- coding: utf-8 -*-
"""项目验收脚本：一次跑完「配置 / 入库 / 页码可溯源 / 路由准确率 / 问答引用」五项检查。

用法：
    python scripts/acceptance.py                # 全部检查（需要 .env 已配置 key）
    python scripts/acceptance.py --skip-qa      # 跳过最耗时的问答生成
    python scripts/acceptance.py --questions 3  # 只跑 3 条问答

退出码：0 = 全部通过；1 = 存在失败项。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# 允许从项目根目录直接运行：python scripts/acceptance.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402

from app.config import settings  # noqa: E402

OK = "PASS"
NG = "FAIL"


def _line(name: str, ok: bool, detail: str = "") -> tuple[bool, str]:
    return ok, f"{name:<28} {OK if ok else NG}  {detail}"


# --------------------------------------------------------------------------- #
# 1. 配置
# --------------------------------------------------------------------------- #
def check_config() -> tuple[bool, str]:
    problems = []
    if not settings.embedding_enabled:
        problems.append("Embedding 未配置")
    if not settings.llm_enabled:
        problems.append("LLM 未配置（问答只能降级）")
    for key, pdf in settings.pdf_paths.items():
        if not pdf.exists():
            problems.append(f"{key} PDF 缺失")
    if not problems:
        detail = f"embedding={settings.embedding_model}/{settings.embedding_dimension}d llm={settings.llm_model}"
        return _line("配置检查", True, detail)
    return _line("配置检查", False, "; ".join(problems))


# --------------------------------------------------------------------------- #
# 2. 入库一致性
# --------------------------------------------------------------------------- #
def _load_chunks() -> list[dict]:
    path = settings.data_dir / "chunks.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def check_ingest() -> tuple[bool, str]:
    chunks = _load_chunks()
    if not chunks:
        return _line("入库产物", False, "data/chunks.jsonl 不存在或为空")
    try:
        from app.ingest.milvus_store import get_store

        rows = get_store().count()
    except Exception as exc:
        return _line("入库产物", False, f"Milvus 不可用: {type(exc).__name__}")
    ok = rows == len(chunks)
    return _line("入库产物", ok, f"chunks.jsonl={len(chunks)} milvus={rows}")


# --------------------------------------------------------------------------- #
# 3. 页码可溯源（随机抽样，用 PDF 原文反查）
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u2011", "-").replace("\u2010", "-")
    text = text.replace("：", ":").replace("，", ",").replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", text)


def check_page_attribution(sample_size: int = 30) -> tuple[bool, str]:
    chunks = _load_chunks()
    if len(chunks) < sample_size:
        return _line("页码可溯源", False, "切片数量不足")
    rng = random.Random(20261027)
    sample = rng.sample(chunks, sample_size)

    docs = {
        book: pymupdf.open(settings.pdf_paths[book])
        for book in ("design", "deep")
        if settings.pdf_paths[book].exists()
    }

    exact = off_by_one = miss = 0
    for c in sample:
        doc = docs.get(c["book"])
        if doc is None:
            miss += 1
            continue
        needle = _norm(c["content"])[:16]
        pages = range(c["page_start"], c["page_end"] + 1)

        def hits(page: int) -> bool:
            return 1 <= page <= doc.page_count and needle in _norm(doc[page - 1].get_text("text"))

        if any(hits(p) for p in pages):
            exact += 1
        elif hits(c["page_start"] - 1):
            off_by_one += 1
        else:
            miss += 1
    for d in docs.values():
        d.close()

    rate = exact / len(sample)
    # 关键指标是"没有页码错置"（off_by_one==0）；exact 受图内文字/行号口径影响，不作为唯一门槛
    ok = off_by_one == 0 and rate >= 0.85
    return _line(
        "页码可溯源",
        ok,
        f"抽样 {len(sample)}：精确 {exact}（{rate:.1%}）页码偏移 {off_by_one} 未匹配 {miss}（图内文字/行号口径）",
    )


# --------------------------------------------------------------------------- #
# 4. 意图路由
# --------------------------------------------------------------------------- #
ROUTER_CASES: list[tuple[str, str]] = [
    ("提示链模式适合什么场景？", "design"),
    ("反思模式怎么用，有代码示例吗？", "design"),
    ("并行化模式适合什么场景？", "design"),
    ("优先级排序模式是什么，怎么实现？", "design"),
    ("A2A 智能体间通信协议怎么接入？", "design"),
    ("护栏与安全模式有哪些落地手段？", "design"),
    ("提示链和路由有什么区别？", "design"),
    ("为什么上下文工程决定了 Agent 的能力上限？", "deep"),
    ("多 Agent 协作为什么会失败，底层原因是什么？", "deep"),
    ("模型后训练和 SFT 的区别与权衡？", "deep"),
    ("RAG 的检索质量为什么受知识组织方式影响？", "deep"),
    ("Agent 的评估指标为什么需要统计显著性？", "deep"),
    ("Coding Agent 的工作原理和演进机制是什么？", "deep"),
    # 主题归属优先：概念式问法 + 仅深入理解书系统覆盖的主题 → deep
    ("我想详细了解上下文压缩的概念及处理方案", "deep"),
    ("上下文工程是什么", "deep"),
    ("上下文压缩有哪几种策略，各自的权衡", "deep"),
    ("《Agent设计模式》里记忆管理怎么写代码？", "design"),
    ("深入理解那本书里怎么解释工具调用？", "deep"),
    ("两本书对多智能体协作的说法有什么不同？", "both"),
]


def check_router() -> tuple[bool, str]:
    from app.rag.router import route_question

    hits = 0
    misses: list[str] = []
    for question, expect in ROUTER_CASES:
        try:
            decision = route_question(question)
            if decision.route.value == expect:
                hits += 1
            else:
                misses.append(f"{question[:16]}→{decision.route.value}(期望{expect})")
        except Exception as exc:  # noqa: BLE001
            misses.append(f"{question[:16]}→异常{type(exc).__name__}")
    rate = hits / len(ROUTER_CASES)
    return _line(
        "意图路由准确率",
        rate >= 0.85,
        f"{hits}/{len(ROUTER_CASES)} = {rate:.0%}" + (f"；偏差：{'; '.join(misses[:3])}" if misses else ""),
    )


# --------------------------------------------------------------------------- #
# 5. 问答 + 引用
# --------------------------------------------------------------------------- #
QA_QUESTIONS = [
    "提示链模式适合什么场景？",
    "为什么上下文工程决定了 Agent 的能力上限？",
    "护栏与安全模式如何防止越权操作？",
    "MCP 与工具函数调用有什么区别？",
    "多智能体协作为什么会失败？",
]


def check_qa(count: int) -> tuple[bool, str]:
    from app.rag.agents import get_agent, verify_citations
    from app.rag.retriever import get_retriever
    from app.rag.router import route_question

    retriever = get_retriever()
    details: list[str] = []
    passed = 0
    total_ms = 0

    for question in QA_QUESTIONS[:count]:
        t0 = time.perf_counter()
        try:
            route = route_question(question)
            citations = retriever.retrieve(route.rewritten_query or question, route.route)
            answer, used_llm = get_agent(route.route).answer(question, citations)
            valid = verify_citations(answer, citations)
        except Exception as exc:  # noqa: BLE001
            details.append(f"{question[:12]}: 异常 {type(exc).__name__}")
            continue

        ms = int((time.perf_counter() - t0) * 1000)
        total_ms += ms
        # 每条引用必须能打印出 书/章/页/段
        labels_ok = all("第" in c.label() and "页" in c.label() and "段" in c.label() for c in citations)
        good = bool(citations) and valid and labels_ok
        passed += good
        details.append(
            f"{question[:12]}: {route.route.value} {len(citations)}引用 "
            f"{'OK' if good else 'NG'} {used_llm and 'LLM' or '摘要'} {ms}ms"
        )

    ok = passed == min(count, len(QA_QUESTIONS))
    return _line("问答与引用", ok, f"{passed}/{count} 通过，平均 {total_ms // max(passed, 1)}ms")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agent 设计问答 Agent 验收")
    parser.add_argument("--skip-qa", action="store_true", help="跳过问答生成（最耗时）")
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--sample", type=int, default=30, help="页码抽样条数")
    parser.add_argument("--no-router", action="store_true", help="跳过路由评测")
    args = parser.parse_args(argv)

    results: list[tuple[bool, str]] = []
    started = time.perf_counter()

    print("=" * 96)
    print("Agent 设计问答 Agent —— 验收")
    print("=" * 96)

    results.append(check_config())
    print(results[-1][1])
    results.append(check_ingest())
    print(results[-1][1])
    results.append(check_page_attribution(args.sample))
    print(results[-1][1])
    if not args.no_router:
        results.append(check_router())
        print(results[-1][1])
    if not args.skip_qa:
        results.append(check_qa(args.questions))
        print(results[-1][1])

    elapsed = time.perf_counter() - started
    failed = [line.split()[0] for ok, line in results if not ok]
    print("-" * 96)
    print(f"合计 {sum(1 for ok, _ in results if ok)}/{len(results)} 项通过，用时 {elapsed:.1f}s")
    if failed:
        print("未通过：", "；".join(failed))
        return 1
    print("结论：验收通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
