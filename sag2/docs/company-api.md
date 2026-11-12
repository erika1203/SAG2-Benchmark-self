# 公司 API：检索、答案生成与独立评测

命令在仓库的 `sag2/` 目录执行。Windows、麒麟 x86 Linux 使用同一个 Python 入口。

`scripts/run_sag2.py` 只负责推理并保存结果；即使输入为 Benchmark 数据集，也不计算评测指标。
需要评测时，再单独运行 `scripts/evaluate_sag2.py`。原来的论文复现入口
`scripts/run_sag2_benchmark.py` 及其 JEV/本地模型流程仍然可用。

## 安装与配置

```bash
uv sync --frozen --python 3.11
```

首次配置时复制 `.env.example` 为 `.env`（Linux：`cp .env.example .env`；
PowerShell：`Copy-Item .env.example .env`），修改以下值。已有 `.env` 时补充这些字段。
首次建库需要 Linux 的 GCC/Clang 或 Windows 的 MSVC C++ Build Tools。

```dotenv
RERANK_BASE_URL=http://your-reranker-host:8000/v1
RERANK_ENDPOINT=/rerank
RERANK_API_KEY=your-reranker-key
RERANK_MODEL=your-reranker-model
RERANK_BATCH_SIZE=32

LLM_BASE_URL=http://your-llm-host:8000/v1
LLM_API_KEY=your-llm-key
LLM_MODEL=your-llm-model
LLM_API=chat
LLM_REASONING_EFFORT=default
LLM_JSON_MODE=true
LLM_STRICT_TOOLS=false
LLM_PARALLEL_TOOL_CALLS_PARAMETER=false

# 不填则逐项复用 LLM_*；可用不同的模型、服务和 Key 生成最终答案。
ANSWER_BASE_URL=
ANSWER_API_KEY=
ANSWER_MODEL=
ANSWER_API=

# 仅 --scorer jev 使用；JEV_API_KEY 为空时兼容 OPENROUTER_API_KEY。
JEV_BASE_URL=https://openrouter.ai/api/alpha/decisions
JEV_API_KEY=your-jev-key
JEV_MODEL=typesafe/jev-1.13
```

调用优先使用显式 CLI 参数，然后使用环境变量/`.env`；进程中已有的环境变量优先于 `.env`。
可用 `--env-file /path/to/company.env` 选择另一份配置。Key 只从环境或配置文件读取，
不会写入结果配置和 HTTP 请求日志；`.env` 已被 Git 忽略。

普通检索只需当前评分服务的配置；开启 `--agentic` 才需要规划 LLM；`--task qa` 才需要答案 LLM。
Embedding 不参与这个 SAG2 流程。模型都通过 API 调用，本机无需安装 PyTorch 或模型权重。

## 三个独立开关

| 选项 | 默认 | 作用 |
|---|---|---|
| `--task retrieve` / `--task qa` | `retrieve` | 只检索 / 检索后生成答案 |
| `--scorer reranker` / `--scorer jev` | `reranker` | 公司 reranker / 原 JEV |
| `--no-agentic` / `--agentic` | `--no-agentic` | 固定 SAG2 流程 / LLM 规划多次检索 |

以下命令均可以在 Bash 或 PowerShell 中作为一行执行。示例语料位于 `examples/corpus.json`。

```bash
# 1. reranker + 无 Agentic + 仅检索
uv run python scripts/run_sag2.py --corpus examples/corpus.json --query "云杉项目承建公司的总部在哪？" --task retrieve --scorer reranker --no-agentic --output-dir output/demo-retrieve

# 2. reranker + 无 Agentic + 检索和答案生成
uv run python scripts/run_sag2.py --corpus examples/corpus.json --query "云杉项目承建公司的总部在哪？" --task qa --scorer reranker --no-agentic --output-dir output/demo-qa

# 3. reranker + Agentic + 检索和答案生成
uv run python scripts/run_sag2.py --corpus examples/corpus.json --query "云杉项目承建公司的总部在哪？" --task qa --scorer reranker --agentic --output-dir output/demo-agentic-qa

# 4. 换回 JEV + Agentic + 仅检索
uv run python scripts/run_sag2.py --corpus examples/corpus.json --query "云杉项目承建公司的总部在哪？" --task retrieve --scorer jev --agentic --output-dir output/demo-jev
```

三组选项互相独立，可任意组合，共八种组合。`--agentic` 控制的是检索规划；
最后的答案生成是单独一次 LLM 调用，不要求工具调用功能。规划 LLM 必须支持原生工具调用。
若规划接口拒绝 `response_format`，设置 `LLM_JSON_MODE=false` 或传 `--no-llm-json-mode`；
模型仍需按照提示词返回合法 JSON。Responses 接口也保留支持，可设置 `LLM_API=responses`。

## 自有语料与批量问题

语料文件为 UTF-8 JSON 数组，每项包含 `title` 和 `text`。批量问题格式为：

```json
[
  {"id": "q-001", "question": "云杉项目承建公司的总部在哪？"},
  {"id": "q-002", "question": "银杏项目由谁承建？"}
]
```

```bash
uv run python scripts/run_sag2.py --corpus my-corpus.json --questions my-questions.json --task qa --scorer reranker --output-dir output/my-batch
```

问题 ID 必须是非空且唯一的字符串。不需要标准答案或评测标签。

## Benchmark 数据也可以只做推理

```bash
# 仅下载、建库，不调用模型
uv run python scripts/run_sag2.py --dataset musique --download --index-only --output-dir output/musique-index

# 用前 10 个问题做检索和生成，保存结果，不执行评测
uv run python scripts/run_sag2.py --dataset musique --limit 10 --task qa --scorer reranker --output-dir output/musique-qa

# 需要时，再对已保存的结果做离线检索评测，不重复检索或生成
uv run python scripts/evaluate_sag2.py --input output/musique-qa/musique/results.json --dataset musique
```

`--dataset all` 可运行三个原有多跳数据集；`--limit` 只限制问题数，检索仍覆盖完整语料。
独立评测脚本计算所保存问题的 Recall@k 和 AllEvidence@k，并显示评测问题数量；不调用模型，
也不执行答案 EM/F1 或 LLM Judge。答案生成与答案质量评测是两个不同步骤。
此入口不包含 Enterprise 专用数据格式，可先使用上述自有 JSON 语料格式。

## 接口约定与评分

Reranker 请求为 `POST {RERANK_BASE_URL}{RERANK_ENDPOINT}`，使用 `Authorization: Bearer ...`。
`RERANK_ENDPOINT` 也可填写完整 URL；`RERANK_BASE_URL` 可以直接填写完整 `/rerank` URL，
默认 endpoint 不会重复拼接。新入口同时接受 `RERANK_MODEL_NAME` 作为模型名兼容项。

```json
{
  "model": "your-reranker-model",
  "query": "证据判定要求及原问题、已找到证据、检索历史",
  "documents": ["标题\n段落正文", "标题\n段落正文"],
  "top_n": 2
}
```

```json
{
  "results": [
    {"index": 1, "relevance_score": 2.0},
    {"index": 0, "relevance_score": -1.0}
  ]
}
```

响应顺序可以变化，适配器按每个批次的 `index` 映射回原候选。
每个候选必须恰好返回一个有限数值 logit；缺项、重复索引、NaN/Infinity 都会报错。
对每个 logit **仅应用一次 sigmoid**：`score = 1 / (1 + exp(-logit))`，使用稳定计算处理大数。
例如 `-1 → 0.2689`，`0 → 0.5`，`2 → 0.8808`。也兼容返回字段 `score`。
这里不做 softmax 或 min-max，不会把 0 到 1 范围内的 logit 误当作已归一化概率。

每轮的候选按 `RERANK_BATCH_SIZE` 分批，要求服务返回全部候选；整个服务请求的并发由
`--concurrency` 控制。原问题、扩展证据和 Agentic 搜索历史都会参与评分输入；未对原文静默截断。
内部保留 `jev_score` 字段以兼容原算法，新入口输出中使用 `score` 和 `raw_logit`。
API 替换发生在 SAG2 的每轮证据评分中，分数会参与词项扩展与最终排序。

JEV 使用原 `state/questions → answers[*].noul` 协议，其分数不额外应用 sigmoid。
用公司 reranker 替换 JEV 会改变实验配置和检索效果，不能视为原论文模型分数复现。

## 答案与输出

答案 LLM 接收原问题及选中的原文片段，并被要求用问题的语言回答、标注 `[1]` 等来源编号，
证据不足时明确说明。`answer_sources` 保存实际提供给生成模型的片段；模型引用的语义正确性
仍需按业务要求评估。`--answer-top-k` 默认 5，且不超过检索的 `--top-k`。
`--answer-max-context-chars` 默认 60000，超过时明确报错，需调整参数以适配模型上下文；
字符数不等于 token 数。没有检索到证据时直接返回说明，不请求答案模型。

输出结构：

```text
output/demo-qa/
  config.json                     # 数据、代码签名与模型配置，不含 Key
  custom/                         # 数据集输入时为 musique 等名称
    details.jsonl                 # 每题完成立即保存
    results.json                  # 按输入顺序汇总
    requests/0/                   # HTTP 请求与响应缓存
      reranker/                   # 或 jev/
      llm/                        # 开启 Agentic 时才有
      answer/                     # 开启生成且有证据时才有
```

`results.json` 中包含 `question`、`passages`、`retrieved_docs`、`api_usage`；
QA 模式还包含 `answer`、`answer_sources`、`answer_model`、`answer_usage`。
API 记录和结果包含发送的业务原文，应保存在部署机器的适当目录。`output/` 已被 Git 忽略。

新运行的输出目录必须尚不存在。继续同一次运行，用 `--resume` 替换 `--output-dir`，
保留相同输入与模型参数：

```bash
uv run python scripts/run_sag2.py --dataset musique --limit 10 --task qa --scorer reranker --resume output/musique-qa
```

已完成问题被跳过；题目中途失败时复用相同 payload 的已成功 HTTP 响应。
更改检索/生成开关、模型、代码或数据时需使用新输出目录，避免混合实验结果。
默认不读取代理环境变量；如确实需要代理，增加 `--trust-env`，内部地址可通过 `NO_PROXY` 排除。
