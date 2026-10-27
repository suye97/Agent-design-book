# 「Agent 设计」问答 Agent —— 技术框架与设计思路

> 依据《设计方案.txt》编写。本文件是开发前的设计基线，包含选型、架构、数据流水线、
> 检索与路由设计、引用与日志规范、工程结构、验收标准与待确认项。

---

## 1. 需求拆解

| 编号 | 需求 | 落地目标 |
|---|---|---|
| R1 | 切片两本 PDF，判断是否需要数据清洗，用 Embedding 模型算向量，写入 Milvus | 离线流水线：解析 → 清洗 → 结构化切片 → 向量化 → 入库，全程可重跑、幂等、有报告 |
| R2 | 意图路由：概念类问题走「设计模式查询 Agent」，详细解释类问题走「深入理解查询 Agent」 | Router 输出 `route ∈ {design, deep, both}` + 置信度，两本各自独立的检索 Agent |
| R3 | 每次回复在日志中打印：哪本书、第几章、第几页、哪段文字 | 结构化引用对象 + 控制台可读日志 + JSONL 审计日志，答案正文带 `[n]` 行内引用 |

隐含需求：答案必须**可溯源、不臆造**；检索失败要明确说"资料未覆盖"，而不是编造。

---

## 2. 现状调研（已完成，非假设）

### 2.1 两本 PDF（均为 LaTeX 生成的文字版 PDF，无需 OCR）

| 书名（元数据标题） | 文件 | 页数 | 大纲条目 | 正文字符/页（中位） | 印刷页码偏移 |
|---|---|---|---|---|---|
| 智能体设计模式：智能系统构建实战指南 | `Agent设计模式-zh-CN.pdf` | 305 | 336 | ≈824 | 印刷页 = PDF 页 − 13 |
| 深入理解 AI Agent：设计原理与工程实践（李博杰） | `深入理解 AI Agent-zh-CN.pdf` | 313 | 293 | ≈1447 | 印刷页 = PDF 页 − 8 |

关键结论：

- **两份 PDF 都带完整书签大纲**（含章、节、页码），章节目录可直接从 `/Outline` 取得，不需要正则硬猜章节；正则只作为兜底。
- 文本层完整，可抽取；但存在典型排版噪声：页眉（书名/章名 + 页码）、页脚页码、目录页、代码行号、断行、连字符 `‑`、LaTeX 连写（`GoogleAgentDevelopmentKit`）。
- 两本书结构不同：设计模式书 21 章 + 结论/术语表/附录 A–G；深入理解书 引言 + 1–10 章 + 后记，且有大段代码（含行号）。
- 全量正文约 70 万字符，估算切片 1200–1800 条 —— Embedding 成本与 Milvus 数据量都很小。

### 2.2 Milvus

- 地址 `http://<MILVUS_HOST>:19530`（token `<MILVUS_TOKEN>`），连通。
- 服务端版本 **2.6.0-rc.1** → 原生支持 **BM25 全文检索 / 稀疏向量 / 中文分词（analyzer: chinese）/ 混合检索（RRF）**。
- 现有 collection：`market_research_samples`（419 行）、`trade_knowledge`（1 行）—— **属于其他项目，本项目不读写、不删除，仅新建独立 collection**。

### 2.3 运行环境

- `.venv`：Python **3.13.2**，已预装 FastAPI 0.162 / Uvicorn / Pydantic 2.14 / python-dotenv / PyYAML（强烈提示本项目以 FastAPI 为服务层）。
- 本次已补装：`pymilvus 3.0.2`、`pypdf`、`pdfminer.six`、`openai`、`httpx`、`pydantic-settings`、`python-multipart`、`tenacity`。
- 外网可达：DashScope（443）、Milvus（19530）、PyPI。

### 2.4 当前缺口（需你提供）

- `EMBEDDING_API_KEY` 为空 —— 设计文档只给了占位符，未给真实 key。
- **没有任何 LLM 配置** —— 意图路由与答案生成需要大模型（OpenAI 兼容端点）。设计文档提到"DeepSeek 无 Embedding API"，推断你侧有 DeepSeek key，但未在该文件中给出。

---

## 3. 技术框架

| 层次 | 选型 | 理由 |
|---|---|---|
| 语言/运行时 | Python 3.13 + 项目内 `.venv` | 目录已预置该虚拟环境 |
| Web 服务 | **FastAPI + Uvicorn** | venv 预装；Pydantic v2 天然做请求/响应契约；SSE 流式简单 |
| 配置 | pydantic-settings + `.env` | 兼容设计文档的 `${...}` 变量名与 `SYMBOL_AGENT_MILVUS_*` |
| PDF 解析 | **PyMuPDF(fitz) 优先 + pypdf 兜底** | 需要页级文本 + 大纲 + 版面块（区分正文/代码/图注） |
| 数据清洗 | 自研规则引擎（正则 + 页级一致率统计） | 页眉页脚/目录/行号等噪声必须按书自适应，规则可单测 |
| 切片 | 结构化分层切片（章 → 节 → 段），带段落编号与页区间 | 引用要求"第几章第几页第几段"，切片必须携带该元数据 |
| Embedding | **DashScope `text-embedding-v4`，1024 维**（OpenAI 兼容 `/embeddings`） | 设计文档指定；中文效果最好 |
| 向量库 | **Milvus 2.6**（pymilvus `MilvusClient`） | 设计文档指定服务器 |
| 检索 | **稠密向量 + 原生 BM25 稀疏向量，RRF 融合**；章节去重 + 相邻块扩展 | Milvus 2.6 原生支持；纯向量对术语/模式名（如"护栏""A2A"）召回不稳，需混合检索 |
| 意图路由 | **LLM 结构化 JSON（json mode）+ 关键词规则兜底 + 显式指令覆盖** | 双保险：无 LLM key 也能跑；有 LLM 时更准；用户明确说"设计模式那本书"时强制覆盖 |
| LLM | OpenAI 兼容 Chat Completions（DeepSeek / DashScope Qwen 可切换） | 可插拔，不绑死厂商 |
| 日志 | stdlib logging（控制台）+ 按日 JSONL 审计日志 | 满足"每次回复打印出处"且可机器复盘 |
| 前端 | 单页原生 HTML/JS/CSS，FastAPI 静态托管 | 零构建，够用；引用侧栏可视化 |
| 测试 | pytest + `fastapi.testclient` | 清洗/切片/路由/引用映射都可离线单测 |

> 一句话：**FastAPI 服务 + 离线批处理流水线 + Milvus 混合检索 + 双库双 Agent 路由 + 强制引用审计**。

---

## 4. 系统架构

```mermaid
flowchart LR
  subgraph Offline[离线：一次性/可重跑]
    PDF1[Agent设计模式.pdf] --> EX[解析 Extract<br/>页文本/大纲/印刷页码]
    PDF2[深入理解 AI Agent.pdf] --> EX
    EX --> CL[清洗 Clean<br/>页眉页脚/目录/行号/断行]
    CL --> CH[结构化切片 Chunk<br/>书/章/节/页/段/metadata]
    CH --> EM[Embedding<br/>text-embedding-v4 · 1024d<br/>批量+重试+sha256缓存]
    CH --> BM[BM25 稀疏向量<br/>中文 analyzer]
    EM --> MV[(Milvus<br/>agent_design_kb)]
    BM --> MV
  end

  subgraph Online[在线：问答]
    Q[用户问题] --> RT{Router<br/>LLM JSON / 规则兜底}
    RT -->|concept| A1[设计模式 Query Agent]
    RT -->|explain| A2[深入理解 Query Agent]
    RT -->|低置信/显式两本| A3[双库并行]
    A1 --> RET[混合检索<br/>dense+BM25 → RRF → 去重/邻接扩展]
    A2 --> RET
    A3 --> RET
    RET --> MV
    RET --> COMP[答案合成<br/>带 [n] 行内引用 + 覆盖度自检]
    COMP --> LOG[(引用审计日志<br/>书/章/页/段/原文摘录)]
    COMP --> API[FastAPI /api/chat]
    API --> UI[单页 Web UI]
  end
```

---

## 5. 数据流水线与清洗设计

### 5.1 需要清洗吗？—— 需要，且是质量关键

`docs/INGEST_REPORT.md` 会记录每步丢弃量。规则按"先统计、再删除"执行，每条规则可单测：

| # | 噪声 | 处理方式 |
|---|---|---|
| C1 | 目录页（设计模式书 PDF 第 1–13 页） | 依据大纲定位正文起点，按页区间整页剔除；目录页不参与检索 |
| C2 | 页眉（章名/书名 + 页码，左右交替） | 统计全书中出现率 >30% 的短行 + 含页码的行 → 按页上下文删除，而非全局黑名单 |
| C3 | 页脚孤立页码 | 行内容为纯数字/罗马数字且位于首尾两行 → 删除 |
| C4 | 代码行号（`1 import os`） | 识别代码块后按行剥离行号前缀，保留代码原文；代码块单独 `chunk_type=code` |
| C5 | 硬换行 | 中文行尾无标点 → 与下一行合并；英文按空格合并 |
| C6 | 连字符/软连字符 `‑` `-` | `‑` 后紧跟非空白 → 删除连接符；跨行断词再拼接 |
| C7 | LaTeX 连写 | 保留原文（不改写语义），仅在检索侧做归一化（全角→半角、大小写） |
| C8 | 图注/表注/公式行 | 单独标记为 `caption`，不单独成块（并入其所属段落上下文） |
| C9 | 参考文献条目 | 章节末尾"参考文献/参考资料"小节整段标记 `references`，默认降权/过滤 |
| C10 | 空白/重复行 | 连续空行压缩；页间重叠内容按 `sha256` 去重 |

**不做的清洗**（避免破坏可引用性）：不做摘要改写、不做机器翻译、不删除代码、不改动原文措辞。

### 5.2 切片策略（结构化 + 语义）

1. **定位**：PDF 大纲 → 章（chapter_no / chapter_title）；`x.y` 标题 → 节（section_no / section_title）；无节则继承章。
2. **段落重建**：页内按"块"（PyMuPDF blocks）还原自然段，结合 C5 合并断行；段落序号 `para_index` 在**章内**连续编号（跨页段落归属起始页，记录 `page_start/page_end`）。
3. **切块**：目标 450–700 中文字符，重叠 80–120 字符；**优先在段落边界切**；超长段落按句号/分号切；代码块整体成块（>120 行再按函数/空行切）。
4. **块元数据**（写入 Milvus 标量字段，供过滤与引用）：
   `id, book, book_title, chapter_no, chapter_title, section_no, section_title,
    page_start, page_end, page_print_start, page_print_end,
    para_start, para_end, chunk_type(text/code/list/caption/references),
    content, content_hash, token_len, prev_id, next_id`
5. **幂等**：`id = sha1(book|chapter|chunk_index|content_hash)[:32]`；重跑时按 `content_hash` 复用 embedding 缓存，避免重复计费。

### 5.3 引用语义（R3 的根）

- 引用单元 = `(book_title, chapter_no + chapter_title, 印刷页码, para_start–para_end, excerpt)`。
- 页：**优先印刷页码**（读者翻书所见），同时保留 PDF 页码用于排查。
- 段：`para_index`（章内序号）+ 20–40 字原文摘录 —— 两者同时给出才满足"哪段文字"。

---

## 6. Milvus 设计

- **collection**：公开名称 `agent_design_kb` 作为 alias，指向带版本后缀的物理 collection（独立新建，不碰现有两个 collection）。重建先写 staging、校验行数再切换 alias；旧版本保留用于恢复。旧版同名物理 collection 先改名为 legacy 备份再建 alias，失败恢复原名。
- **一致性**：普通入库使用幂等 upsert + 旧 ID 清理；正式 Milvus 更新成功后才提交 `data/chunks.jsonl`。staging/备份不自动删除，由运维按恢复策略清理。
- **字段**：见 5.2；`vector` = `FLOAT_VECTOR(1024)`；`sparse` = `SPARSE_FLOAT_VECTOR`（BM25 Function 由 `content` 自动生成，analyzer=`chinese`）。
- **索引**：`vector` → HNSW，`M=16, efConstruction=200`，度量 `COSINE`；`sparse` → `SPARSE_INVERTED_INDEX`（`BM25`）；标量 `book / chapter_no / chunk_type` → `INVERTED`。
- **检索**：`hybrid_search`（dense + sparse）→ `RRFRanker(k=60)`，`limit=8~12`；`filter` 按 `book` 限定（路由决定）或 `book in [design, deep]`（双库）。
- **可选 rerank**：对 RRF 前 20 条用 LLM 相关性打分（开关，默认关，避免额外延迟/费用）。
- **章节多样性**：同一章最多保留 3 块；相邻块（`next_id`）自动扩展 1 块补全上下文。

---

## 7. 意图路由设计

### 7.1 两个维度（不是词面匹配）

| 维度 | 取值 | 说明 |
|---|---|---|
| 答案形态 `intent` | `concept` / `explain` / `ambiguous` | 用户要「模式目录式」答案（是什么/怎么用/代码/场景），还是「机制展开式」答案（为什么/原理/方案对比/权衡/演进） |
| 主题归属 `route` | `design` / `deep` / `both` | 这个主题**哪本书有系统讲解** |

判定顺序：**先主题归属，再答案形态**。

- 主题只在一本书系统讲解时（如上下文工程/上下文压缩/后训练只 deep；提示链/并行化/护栏只 design），**以主题归属为准**——即使问法是"是什么"。
- 两本书都讲的话题（MCP、RAG、记忆、多智能体、评估）用答案形态分胜负；难分则 `both`。
- `intent` 只描述答案形态，`route` 必须落到"有内容的那本书"。

### 7.2 决策流程（四步）

1. **显式指令覆盖**（最高优先）："设计模式那本书/《Agent设计模式》"→ design；"深入理解/李博杰/原理那本"→ deep；"两本都看/对比两本书"→ both。
2. **LLM 结构化判定**：`json_object` 输出
   `{intent, confidence, book_hint, rewritten_query, reason}`；temperature=0。
   系统提示词里给出两本书的**主题清单**、判定顺序、深度信号词，以及 11 条 few-shot
   （含"我想详细了解上下文压缩的概念及处理方案 → explain/deep"这类反例修正）。
3. **信号纠偏**（LLM 之后的确定性保险）：用三类词表计算信号——
   - **深度诉求**：详细了解 / 详细讲 / 展开 / 系统讲 / 讲透 / 处理方案 / 方案对比 / 权衡 /
     为什么 / 原理 / 机制 / 底层 / 内部实现 / 演进 / 区别 …
   - **模式诉求**：设计模式 / 模式 / 适用场景 / 什么时候用 / 怎么用 / 代码示例 / 落地手段 …
   - **主题归属表**（`_TOPIC_COVERAGE`，长词优先匹配并消费）：
     上下文工程/上下文压缩/KV Cache/后训练/Coding Agent/评估指标 → deep；
     提示链/路由/并行化/反思/记忆管理/HITL/A2A/护栏 → design；
     MCP/RAG/记忆/多智能体/评估 → 双覆盖（同时计分）。
   当 LLM 的路由与信号明显冲突（对方得分 ≥ 本侧 1.3 倍且 ≥ 1.2）时纠偏，
   置信度抬到 ≥ 0.7，`reason` 标注"主题纠偏"。
4. **规则兜底**（无 key / LLM 失败）：同一套信号直接出 `route`，置信度 = 领先幅度 × 证据强度。

> 关键变化：**"是什么 / 概念"这类词不再参与"选书"**——它们只说明答案形态，不说明主题归属。
> 旧实现把它们当作 design 增益词，导致"上下文工程是什么"被误送设计模式书。

### 7.3 典型判定对照

| 问题 | 主题归属 | 答案形态 | 结果 |
|---|---|---|---|
| 我想详细了解上下文压缩的概念及处理方案 | deep（唯一系统讲解） | explain（要展开） | **deep** |
| 上下文工程是什么 | deep（唯一系统讲解） | concept 式问法 | **deep**（主题优先） |
| 上下文压缩有哪几种策略，各自的权衡 | deep | explain | **deep** |
| 提示链模式是什么 / 怎么用 | design（模式章节） | concept | **design** |
| 提示链和路由有什么区别 | design（两个模式都在本书） | concept | **design** |
| 护栏与安全模式有哪些落地手段 | design（第 18 章） | concept | **design** |
| 为什么要给 Agent 加护栏、控制机制是什么 | deep（机制/权衡） | explain | **deep** |
| 多智能体怎么协调 | 双覆盖 | 难分 | **both + 澄清** |

### 7.4 置信度策略

- `confidence ≥ 0.6` → 单库路由；
- `0.35 ~ 0.6` → **双库并行**（both），答案中标注"两本书均有相关内容"；
- `< 0.35` → 默认 `both` + 返回 `clarify_hint`（"你是想了解某个设计模式，还是想深入理解背后原理？"），但仍先给出检索结果，避免空转。
- 路由结果**始终回传**给前端并在日志落盘，便于复盘误判。

> 注：需求写的是二路由。`both` 只在"低置信/用户明确要求对比"时启用，是对二路由的兜底，不改变主路径语义。

---

## 8. 双 Agent 与答案合成

| Agent | 职责 | 检索范围 | 提示词侧重 |
|---|---|---|---|
| `DesignPatternAgent` | 回答"是什么/怎么用/代码怎么写" | `book = design` | 结构化：模式定位 → 适用场景 → 代码要点 → 注意事项 |
| `DeepUnderstandingAgent` | 回答"为什么/原理/机制/权衡" | `book = deep` | 概念链：定义 → 机制拆解 → 设计权衡 → 与其他章节的关联 |
| `BothAgent` | 合并两库结果 | `book in (design, deep)` | 明确区分"设计模式书怎么说"与"深入理解书怎么解释" |

**答案合成规则**：

- 只允许使用检索到的片段作答；证据不足时明确输出"现有资料未覆盖该问题"，并给最接近的 2–3 个方向。
- 引用格式：正文 `[1][2]`，末尾"引用出处"列表给出 `[1] 《智能体设计模式》第 5 章 工具使用（函数调用） · 第 67 页 · 第 3 段`。
- **引用校验**：回答生成后，校验正文中每个 `[n]` 都存在且指向真实片段；无引用或引用越界 → 追加提示并记录 `citation_violation`。

---

## 9. 日志规范（R3）

每轮问答产生一条结构化记录（`logs/answers-YYYYMMDD.jsonl`），并同时打印控制台可读行：

```
[ANSWER] id=20261027-142233-8f3a  route=design  conf=0.86  ms=1842
[引用 1] 《智能体设计模式：智能系统构建实战指南》 第 5 章 工具使用（函数调用）· 第 67 页 · 第 3 段
         原文：“工具使用模式让模型在推理过程中调用外部函数……”
[引用 2] 《智能体设计模式：智能系统构建实战指南》 第 10 章 模型上下文协议（MCP）· 第 122 页 · 第 1 段
         原文：“MCP 与工具函数调用的区别在于……”
```

审计字段：`answer_id, ts, question, rewritten_query, route, confidence, router_source(llm/rule/override),
latency_ms, model, sources[{chunk_id, book, chapter_no, chapter_title, page, para_range, excerpt, score, rank}],
cited_ids, citation_violation, token_usage, top_k, filters`。

---

## 10. API / CLI / UI

**API（FastAPI）**

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | 问答主入口，返回 `answer / citations / route / latency` |
| POST | `/api/chat/stream` | SSE 流式，先下发 `route` + `citations`，再流式下发 token |
| POST | `/api/search` | 只检索不生成（透明度/调试） |
| GET | `/api/health` | 服务 + Milvus + 配置自检 |
| GET | `/api/stats` | collection 统计、各书章节/块数 |
| GET | `/api/books` | 两本书的章节目录树（前端导航/引用跳转） |
| POST | `/api/admin/ingest` | 触发重建索引（可 `dry_run`、按书重建、强制重算向量），需独立 `ADMIN_API_KEY` Bearer token |

**CLI**：`python -m app.cli ingest [--book design|deep|all] [--dry-run] [--recreate]`、`ask "问题"`、`stats`、`check`（连通性自检）、`serve`。

**Web UI**：单页聊天界面，右侧"引用出处"面板；每条引用可展开原文；顶部显示本次路由决策与置信度。

---

## 11. 工程结构

```
Agent-design-book/
├─ app/
│  ├─ main.py            # FastAPI 入口 + 静态托管
│  ├─ config.py          # pydantic-settings（兼容设计方案.txt 变量）
│  ├─ models.py          # Pydantic 契约：Chat/Citation/Route/Chunk
│  ├─ logging_conf.py    # 控制台 + JSONL 审计日志
│  ├─ cli.py             # ingest / ask / stats / check / serve
│  ├─ ingest/
│  │  ├─ extract.py      # PDF → 页文本 / 大纲 / 印刷页码偏移
│  │  ├─ clean.py        # C1–C10 清洗规则（可单测）
│  │  ├─ chunk.py        # 结构化切片 + 元数据
│  │  ├─ embed.py        # DashScope 兼容 /embeddings，批量+重试+sqlite 缓存
│  │  ├─ milvus_store.py # schema/index/upsert/hybrid_search
│  │  └─ pipeline.py     # 编排 + docs/INGEST_REPORT.md 生成
│  ├─ rag/
│  │  ├─ router.py       # 三级意图路由
│  │  ├─ retriever.py    # 混合检索 + 融合/去重/邻接扩展
│  │  ├─ agents.py       # 双库 Agent + 答案合成 + 引用校验
│  │  └─ llm.py          # OpenAI 兼容客户端（重试/超时/json mode）
│  └─ api/
│     ├─ chat.py  search.py  admin.py  health.py
├─ web/  index.html  app.js  styles.css
├─ tests/ test_clean.py test_chunk.py test_router.py test_citations.py test_api.py
├─ scripts/ check_env.py
├─ docs/  DESIGN.md  INGEST_REPORT.md
├─ data/  pages/ chunks.jsonl  embed_cache.sqlite3   （中间产物，可重建）
├─ logs/  answers-YYYYMMDD.jsonl
├─ .env.example  requirements.txt  README.md
```

---

## 12. 验收标准

1. `python -m app.cli check` 全绿（Milvus 连通、collection 就绪、embedding/LLM 可用）。
2. `python -m app.cli ingest` 输出报告：两本书各自页数、清洗丢弃行数、chunk 数、向量数、耗时、失败数；重复执行结果一致（幂等）。
3. `GET /api/stats` 的 chunk 数与 `data/chunks.jsonl` 行数一致；随机抽 20 条可人工核对"书/章/页/段"正确。
4. 路由用例集 ≥20 条（概念/原理/显式指定），准确率 ≥ 85%（LLM 模式）/ ≥ 75%（规则兜底）。
5. 回答的每条事实都能对应到日志中的"书 + 章 + 页 + 段 + 原文摘录"；无引用回答占比 0。
6. `pytest` 全绿；无 key 时 `check`/`ingest --dry-run`/规则路由仍可运行。

---

## 13. 风险与待确认

| 风险/待确认 | 影响 | 处理 |
|---|---|---|
| **缺 `EMBEDDING_API_KEY`** | 无法真实入库/检索 | 需你提供；先用 `--dry-run` 与桩向量验证链路 |
| **缺 LLM 配置** | 路由只能规则兜底、答案无法生成 | 需提供 base_url/model/api_key（DeepSeek 或 DashScope Qwen 均可） |
| Milvus token 为 `<MILVUS_TOKEN>`、公网 IP | 安全风险 | 仅项目内使用；不写入前端、不回显；建议后续改为内网/独立账号 |
| 印刷页码偏移按书不同 | 引用页码错位 | 自动检测偏移（抽样页脚数字 vs PDF 页序），失败则回退 PDF 页码并标注 |
| LaTeX 连写/代码行号 | 影响检索与可读性 | 清洗规则 + 单测覆盖；不改写原文语义 |
| 扫描页/图页 | 无文本可检索 | 已探测两本书无低文本页（仅封面 1 页），无需 OCR |

---

## 14. 实施结果（交付后回填）

| 设计项 | 落地情况 |
|---|---|
| 解析/清洗/切片 | 已实现；两书 618 页 → 1289 块，10 类清洗规则全部可单测（规则命中见 `docs/INGEST_REPORT.md`） |
| Embedding | DashScope `text-embedding-v4` 真实调用，1024 维，sqlite 缓存命中重跑约 15s |
| Milvus | `agent_design_kb` 已建表并写入 1289 行；HNSW(COSINE) + 中文 analyzer BM25 + RRFRanker 混合检索实测可用，未降级 |
| 页码 | 印刷页码自动检测（design −13 / deep −8）；overlap 跨页校正后 120 条抽检页码偏移为 0 |
| 路由 | 三级路由全部落地；真实 LLM 15 条用例 100%（含 3 条 override） |
| 引用审计 | 控制台 + JSONL 双写；每条引用含 书/章/页/段/原文摘录；`citation_ok` 校验生效 |
| 无 key 降级 | Embedding 桩向量（拒绝写入正式库）+ 规则路由 + 检索摘要作答，已实现 |
| UI | 单页 Web，路由徽标 + 引用侧栏；截图见 `docs/ui-demo.png` |

与原设计的偏差与原因：
1. **LLM 为推理模型**（`deepseek-flash`，先输出 reasoning 再输出 content）：`max_tokens` 默认提升到 2048，
   路由与答案生成均验证通过；可通过 `LLM_MODEL=deepseek-chat` 降到 3–5s。
2. **`/api/chat` 响应中的 `answer` 保留引用清单**，前端展示时拆分到右侧面板，API 消费者仍拿到完整文本。
3. **`Citation.content` 设为 `exclude=True`**：切片全文只供 Agent 组 prompt，不进 API 响应体，避免响应过大。
4. **验收口径**：页码抽检以"无页码偏移"为硬指标，精确率受图内文字/代码行号影响，见 `docs/ACCEPTANCE.md`。
