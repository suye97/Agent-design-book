# 验收报告（ACCEPTANCE）

- 验收时间：2026-10-27 15:40（Asia/Shanghai）
- 运行环境：Windows / Python 3.13.2 / 项目内 `.venv`
- Embedding：DashScope `text-embedding-v4`，1024 维（真实调用）
- LLM：`deepseek-flash`（OpenAI 兼容端点 `https://api.deepseek.com`，推理模型）
- 向量库：Milvus 2.6.0-rc.1 @ `http://<MILVUS_HOST>:19530`，collection `agent_design_kb`
- 一键复现：`python scripts/acceptance.py`

## 1. 自动验收结果

```
================================================================================================
Agent 设计问答 Agent —— 验收
================================================================================================
配置检查                         PASS  embedding=text-embedding-v4/1024d llm=deepseek-flash
入库产物                         PASS  chunks.jsonl=1289 milvus=1289
页码可溯源                        PASS  抽样 30：精确 28（93.3%）页码偏移 0 未匹配 2（图内文字/行号口径）
意图路由准确率                      PASS  15/15 = 100%
问答与引用                        PASS  5/5 通过，平均 13927ms
------------------------------------------------------------------------------------------------
合计 5/5 项通过，用时 85.4s
结论：验收通过
```

## 2. 入库数据

| 书 | 页数 | 正文起始 | 印刷页码偏移 | 章 | 节 | Chunk |
|---|---|---|---|---|---|---|
| 智能体设计模式：智能系统构建实战指南 | 305 | p14 | −13（页眉页码推断） | 33 | 302 | 399 |
| 深入理解 AI Agent：设计原理与工程实践 | 313 | p9 | −8（PDF Page Label） | 12 | 281 | 890 |
| 合计 | 618 | | | | | **1289** |

Chunk 类型：design = text 231 / code 104 / list 55 / references 9；deep = text 728 / code 114 / list 48。
平均长度 656 / 582 字符，目标区间 450–700、重叠 80–120。

清洗命中（部分）：design 剔除目录 13 页、页眉页脚 551 行、代码行号 1793 行、硬换行合并 2227 处、图注 55、参考文献 96；
deep 剔除目录 8 页、页眉页脚 598 行、硬换行合并 5205 处、图注 138。

## 3. 页码可溯源（随机 120 条，固定 seed=7）

| 指标 | 数值 |
|---|---|
| 精确命中 `page_start` | 114 / 120 = **95.0%** |
| 页码偏移（命中 `page_start±1`） | **0** |
| 未匹配（图内文字 / 代码行号口径） | 6 |

未匹配的 6 条已逐条归因：5 条为图/表/公式内文字（PDF 文本流按列排列，与视觉行重排顺序不一致）、1 条为代码块行号剥离导致前缀不连续；**页码本身正确**。修复前为 108/120（90.0%，4 条偏移一页），已通过 overlap 页码校正修复。

## 4. 意图路由（真实 LLM，15 条用例）

| 类别 | 用例数 | 正确 |
|---|---|---|
| 设计模式（概念/用法/代码/场景） | 6 | 6 |
| 深入理解（为什么/原理/机制/区别/演进） | 6 | 6 |
| 显式指定书目 / 双库对比 | 3 | 3 |
| **合计** | **15** | **15（100%）** |

典型判定：`提示链模式适合什么场景？` → design 0.95；`为什么上下文工程决定了 Agent 的能力上限？` → deep 0.90；
`《Agent设计模式》里记忆管理怎么写代码？` → design 1.00（override）；`两本书对多智能体协作的说法有什么不同？` → both 1.00（override）。

## 5. 问答与引用

5 条问题全部：有引用、`[n]` 标记与召回片段一一对应（`citation_ok=true`）、每条引用都能打印「书 + 章 + 页 + 段」。
平均端到端 13.9s（deepseek-flash 为推理模型，思考 tokens 占比较大）。

## 6. 审计日志样例（真实运行，`logs/answers-20261027.jsonl`）

```
[ANSWER] id=20261027-153859-f0df0d route=design intent=concept conf=0.95 src=llm ms=... model=deepseek-flash
[引用 1] 《智能体设计模式：智能系统构建实战指南》第 1 章 提示链（Prompt Chaining） · 第 18 页 · 第 76–85 段
         原文：“是什幺：复杂任务若用单一提示处理，LLM 易因认知负担过重而出错，如忽略指令、…”
[引用 2] 《智能体设计模式：智能系统构建实战指南》第 1 章 提示链（Prompt Chaining） · 第 15 页 · 第 46–59 段
         原文：“・处理：更新对话状态。 ・提示2：基于当前状态生成回复或识别下一步所需信息。…”
```

`logs/answers-YYYYMMDD.jsonl` 每行是一条完整记录，含 `question / route / answer / model / latency_ms /
sources[{book, chapter, page, para, excerpt, chunk_id, score}] / citation_violation`。

## 7. 单元测试

```
python -m pytest tests -q
200 passed
```

覆盖：PDF 解析、清洗规则、切片与页码、Milvus 往返（真实库，不可达自动 skip）、路由三级决策、
检索装配（同章去重/邻接扩展）、双库 Agent 与降级、引用校验、API 契约（fake 组件，零外部依赖）。

## 8. 已知限制

| 限制 | 影响 | 现状 |
|---|---|---|
| 146 个短块（<450 字符） | 章末/图注/紧邻代码的段落较短 | 可接受；检索时由相邻块扩展补全上下文 |
| 设计模式书节标题无编号 | `section_no` 为空，引用退化为「章 + 页 + 段」 | 书名原文如此，不臆造编号 |
| 图内文字并入正文 | 6/120 抽检前缀不连续 | 页码正确，不影响引用定位 |
| deepseek-flash 为推理模型 | 单轮 10–15s，成本高于 chat 模型 | `LLM_MAX_TOKENS=2048`；可切 `deepseek-chat` 提速 |
| Milvus 公网 + root token | 安全风险 | 仅服务端读取，不进前端；建议内网/独立账号 |

## 9. 复现

```powershell
# 1. 自检
.\.venv\Scripts\python.exe -m app.cli check

# 2. 入库（有缓存，重跑约 15s；--recreate 可强制重建）
.\.venv\Scripts\python.exe -m app.cli ingest

# 3. 验收（约 80s）
.\.venv\Scripts\python.exe scripts\acceptance.py

# 4. 起服务
.\.venv\Scripts\python.exe -m app.cli serve
```

