# 离线入库报告（INGEST_REPORT）

- 生成时间：2026-10-27 15:34:20
- 运行模式：正式入库
- collection：`agent_design_kb`（Milvus：http://<MILVUS_HOST>:19530）
- Embedding：text-embedding-v4，维度 1024
- 切片文件：`data\chunks.jsonl`
- 耗时：15091 ms；写入 Milvus 行数：1289

## 汇总

| 书 | 页数 | 正文起始页 | 印刷页码偏移 | 章数 | 节数 | 清洗单元 | Chunk 数 |
|---|---|---|---|---|---|---|---|
| 智能体设计模式：智能系统构建实战指南 | 305 | 14 | -13（running-head-number） | 33 | 302 | 2476 | 399 |
| 深入理解 AI Agent：设计原理与工程实践 | 313 | 9 | -8（page-label） | 12 | 281 | 4763 | 890 |
| **合计** | 618 | - | - | - | - | - | **1289** |

## 清洗统计（C1–C10）

### 《智能体设计模式：智能系统构建实战指南》

- 剔除目录/封面页：13 页（[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]）
- 丢弃行数：896
- 规则命中：C1: pages=13, lines=345；C10: duplicate_units=2；C2: lines=259；C3: lines=292；C4: stripped_line_numbers=1793；C5: merged_line_joins=2227；C6: hyphen_fixes=6；C8: captions=55；C9: reference_units=96
- 清洗单元：{'body': 1381, 'list': 843, 'code': 101, 'caption': 55, 'reference': 96}
- Chunk：399 个，按类型 {'text': 231, 'code': 104, 'list': 55, 'references': 9}，平均 656.0 字符；低于目标 83，正文超上限 1，代码超上限（豁免）52

### 《深入理解 AI Agent：设计原理与工程实践》

- 剔除目录/封面页：8 页（[1, 2, 3, 4, 5, 6, 7, 8]）
- 丢弃行数：909
- 规则命中：C1: pages=8, lines=311；C2: lines=293；C3: lines=305；C5: merged_line_joins=5205；C6: hyphen_fixes=36；C8: captions=138
- 清洗单元：{'body': 4138, 'caption': 138, 'code': 150, 'list': 337}
- Chunk：890 个，按类型 {'text': 728, 'code': 114, 'list': 48}，平均 582.3 字符；低于目标 63，正文超上限 7，代码超上限（豁免）5

## 抽样核对（书 / 章 / 节 / 页 / 段 / 类型）

### 《智能体设计模式：智能系统构建实战指南》

| # | 书 | 章 | 节 | PDF页 | 印刷页 | 段 | 类型 | 长度 | 正文摘录 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | design | 0 简介 |  前言 | 14-14 | 1-1 | 1-5 | text | 480 | 欢迎阅读《智能体设计模式：智能系统构建实战指南》。在现代人工智能领域，我们见证了从简单的响应式程序到能够理解上下文、做出 |
| 2 | design | 9 学习与适应 |   | 112-112 | 99-99 | 1-7 | list | 565 | 学习与适应是提升智能体能力的关键。这些过程使智能体能够突破预设参数，通过经验和环境交互自主改进。通过学习和适应，智能体能 |
| 3 | design | 18 护栏与安全模式 |  实战代码 CrewAI 示例 | 206-207 | 193-194 | 25-28 | text | 526 | 任务evaluate_input_task 动态嵌入SAFETY_GUARDRAIL_PROMPT 和用户输入，要求输出 |

### 《深入理解 AI Agent：设计原理与工程实践》

| # | 书 | 章 | 节 | PDF页 | 印刷页 | 段 | 类型 | 长度 | 正文摘录 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | deep | 0 引言 |   | 9-9 | 1-1 | 1-3 | text | 505 | 2025 年8 月至10 月，我在图灵《AI Agent 实战营》中讲授了一系列课程。课程围绕一个核心问题展开：如何以可 |
| 2 | deep | 4 工具 | 4.1 工具的分类 | 111-112 | 103-104 | 13-16 | text | 622 | 更复杂的原因是使用不同的模型、工具、提示词和上下文执行不同的任务，实现更好的效果。第10 章将进一步讲解多Agent 架 |
| 3 | deep | 7 Agent 的评估 | 7.6.3 Agent 系统的成本分析 | 213-213 | 205-205 | 375-386 | text | 652 | 一次网页搜索返回的内容可能就占用2000-5000 个token，而且在后续每轮推理中都会作为输入被反复计费。 基础设施 |

## 页码可溯源抽检（固定口径，便于回归）

- 抽样：随机 120 条（seed=7）；口径：content 去空白归一化后的前 16 字符是否出现在对应 PDF 页文本中。
- **page_start 精确命中：114/120（95.0%）**；page_start-1：0；page_start+1：0；落在声明页区间内：0；无法在任何声明页匹配：6。

| chunk | 书 | 章 | 页区间 | 分类 | 内容前缀 |
|---|---|---|---|---|---|
| db052ee0d7 | deep | 10 | 281-281 | figure/formula-text-order | ↑ 所有阶段共享同一对话历史+ 工具调用参数传递结构化数据 ✓ 完整轨迹✓ 模块 |
| 8c102a051e | deep | 6 | 178-178 | figure/formula-text-order | 快慢不一致 "今天星期几"→快思考已对→慢思考仍运行 8s独立思考路径，假设可能 |
| 5ad62fb957 | design | 17 | 196-196 | code-line-number-order |   # 创建Agent 图   from langgraph.graph imp |
| 3b1a20cde6 | deep | 7 | 222-222 | figure/formula-text-order | "将罐子放入锅中" 评估指标 成功率完成时间泛化能力Sim-to-Real 罐子 |
| 69a85c2cae | deep | 3 | 94-94 | figure/formula-text-order | 词频饱和 (TF)逆文档频率 (IDF)长度归一化 (b) k₁ 控制饱和速度衡 |
| db14e14209 | deep | 10 | 299-299 | code-line-number-order | 输入: 用户需求描述输入: PRD.md输入: design.md输入: tas |

**匹配口径限制（非页码错置）**：图/表/公式内的文字在 PDF 文本流中按列或按块排列，我们按视觉行重排后并入正文，与原始文本流顺序不同，归一化前缀无法连续命中；代码块在 PDF 文本中行号与代码同行，剥离行号后前缀与页文本不再连续。这两类仅影响本抽检口径；page_start 已用 overlap 所在页 / 页文本校验校正。

## Milvus

- 状态：已写入 1289 行（collection=agent_design_kb，row_count=1289）

## 已知限制 / 待办

- 83 个块小于 450 字符（章末/图注/短章）
- 1 个正文块大于 700 字符
- 63 个块小于 450 字符（章末/图注/短章）
- 7 个正文块大于 700 字符

## 复现命令

```powershell
# dry-run（无 API key 也可跑）
python -m app.ingest.pipeline --dry-run
# 正式入库（需要 EMBEDDING_API_KEY）
python -m app.ingest.pipeline --book all
```
