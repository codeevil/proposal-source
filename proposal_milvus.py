"""
Optimizer program with multiple message interactions.
Includes system prompt, measures client time and estimates server processing time.
"""

import argparse
import os
import time

import httpx
from dotenv import load_dotenv
from litellm import OpenAI

# Load environment variables
load_dotenv()

# Constants
BASE_URL = "http://localhost:8004/v1"  # Local vLLM server
# BASE_URL = "https://ai-green.yasdb.com/compatible-mode/v1"
MODEL_NAME = "/ssd_data/models/Qwen3-30B-A3B-Instruct-2507-FP8"
# MODEL_NAME = "deepseek-v4-flash"
DEFAULT_TIMEOUT = 300

# Parse command line arguments
parser = argparse.ArgumentParser(description="Optimizer program")
parser.add_argument("--output", "--output-file", type=str, required=True,
                    help="Output file path (required)")
# parser.add_argument("--sql", type=str, required=True,
#                     help="SQL content (required)")
parser.add_argument("--mode", type=str, default="standard",
                    choices=["extreme", "standard", "balance"],
                    help="Generation mode: extreme (15 diverse strategies), "
                         "standard (default, 10 strategies), balance (6 balanced strategies)")
parser.add_argument("--dataset", type=str, default=None,
                    choices=["SIFT", "PAPER", "YFCC"],
                    help="Dataset to use for statistics (SIFT, PAPER, YFCC). "
                         "If not specified, uses default stat.txt")
parser.add_argument("--selective", type=float, default=0.01,
                    help="Scalar filter selectivity ratio (default: 0.01). "
                         "Common values: 0.01, 0.083, 0.3")
args = parser.parse_args()

OUTPUT_FILE = args.output
# SQL_CONTENT = args.sql
MODE = args.mode
SELECTIVE = args.selective


# Read file contents
def read_file(filepath):
    """Read file content, return empty string if file not found."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"[WARNING] File not found: {filepath}")
        return ""
    except Exception as e:
        print(f"[ERROR] Failed to read file {filepath}: {e}")
        return ""

# Define file paths
DATASET_STAT_MAP = {
    "SIFT": "/home/liujianzhong/proposal-source/sift_stat.txt",
    "PAPER": "/home/liujianzhong/proposal-source/paper_stat.txt",
    "YFCC": "/home/liujianzhong/proposal-source/yfcc_stat.txt",
}
STAT_FILE = DATASET_STAT_MAP.get(args.dataset) if args.dataset else "/home/liujianzhong/proposal-source/stat.txt"
STAT_CONTENT = read_file(STAT_FILE)

# Mode-based proposal count
MODE_CONFIG = {
    "extreme": {"count": "15", "label": "极端多样"},
    "standard": {"count": "10", "label": "标准"},
    "balance": {"count": "6", "label": "平衡"},
}
PROPOSAL_COUNT = MODE_CONFIG.get(MODE, MODE_CONFIG["standard"])["count"]
MODE_LABEL = MODE_CONFIG.get(MODE, MODE_CONFIG["standard"])["label"]

# System prompt
# SYSTEM_PROMPT = """
# 你是一位**PostgreSQL优化器内核专家**和**pgvector向量数据库资深架构师**，精通：
# - pgvector扩展的内部实现机制（HNSW/IVFFlat索引构建、查询执行流程、代价估算模型）
# - PostgreSQL查询优化器的行为特性与局限性
# - 向量标量混合查询的各类性能瓶颈与解决方案
# - pg_hint_plan扩展的精确使用方法
# - 大规模向量数据集的查询性能调优技术
# """

# TEMPLATE_CONTENT = """
# ```json
# [
#   {
#     "strategy_id": <整数，1 ~ {PROPOSAL_COUNT}，按顺序编号>,
#     "name": "字符串，策略简要名称，不超过20字",
#     "description": "字符串，结合数据分布与选择性数值，说明设计思想、优势与潜在代价，必须明确引用选择性数值",
#     "vector_index_used": "字符串，索引名：my_table_image_vec_idx / my_table_ivf_image_vec_idx / null（无索引，精确搜索），如果`filter_strategy`为`post-filter`，则`vector_index_used`不得为`null`",
#     "index_parameters": {
#       "type": "字符串，HNSW / IVFFlat / none",
#       "ef_search": "整数，仅HNSW有效，其余为null",
#       "probes": "整数，仅IVFFlat有效，其余为null",
#       "distance_type": "字符串，L2 / inner_product / cosine，与原查询保持一致"
#     },
#     "filter_strategy": "字符串，pre-filter / post-filter",
#     "sql_rewrite_required": "布尔值，true / false",
#     "rewritten_sql": "字符串，sql_rewrite_required为true时填写完整改写后的SQL；否则为空字符串",
#     "hint": "字符串，使用pg_hint_plan时填写准确提示语句；否则为空字符串，严格遵循语法规范",
#     "refill_fallback_required": "布尔值，true / false",
#     "refill_parameters": {
#       "hnsw.iterative_scan": "字符串，off / strict_order / relaxed_order，仅HNSW有效，其余为null",
#       "hnsw.max_scan_tuples": "整数，仅HNSW有效，其余为null",
#       "ivfflat.iterative_scan": "字符串，off / relaxed_order，仅IVFFlat有效，其余为null",
#       "ivfflat.max_probes": "整数，仅IVFFlat有效，其余为null"
#     },
#     "notes": "字符串，补充说明执行边界、回退条件、适用场景限制等，不超过150字"
#   }
# ]
# ```
# """


PROMPTS = [
"""
# Milvus 混合查询策略生成
---

## 角色定位
你是精通 Milvus 2.4.16 内核执行机制、向量查询优化器、HNSW 索引原理的资深向量数据库性能优化专家，擅长基于数据统计特征与业务目标，在**召回率**与**查询性能（QPS）**之间做最优权衡，生成可直接落地的查询参数策略。

## 核心任务
基于下面的**输入信息**，生成 **20 组差异化的向量标量混合查询策略**。每组策略对应一套参数配置，覆盖从极致性能到极致召回的全梯度权衡，同时包含不同过滤模式、范围搜索的有效组合，所有策略必须严格适配 Milvus 2.4.16 版本的执行逻辑，最终目标是实现召回率与 QPS 的综合收益最大化。

## 输入信息
**集合(表)结构：**
```yaml
milvus:
  table_name: my_table
  columns:
    - name: id
      type: INT64
      primary: True
      description: "id"
    - name: equal
      type: INT16
      description: "equal_val"
    - name: image_vec
      type: FLOAT_VECTOR
      dimension: 128
      description: "image_vec(128)"
```

**HNSW索引配置：**
```yaml
milvus:
  hnsw:
    - index_column: "image_vec"
      distance: L2
      params:
        - m: 16
          ef_construction: 64
```

**向量标量混合查询的基础信息：**
```yaml
SIFT:
- name: query1
  vector_field: image_vec
  scalar_filters:
  - field: equal
    operator: ==
    value: 27
    logic: and
  limit: 100
```
如上 limit值(TopK数量)为100；过滤表达式形如"equal == 27"。

**统计信息：**
- my_table集合(表)总行数: 1000000
- equal字段的选择率(过滤后有效数据占比): {SELECTIVE}

---

## 一、专业背景知识（Milvus 2.4.16 HNSW 混合查询机制）
你必须基于以下官方机制进行推理，不得引入其他版本特性或虚构功能。

### 1.1 向量标量混合查询的两种执行路径
Milvus 2.4.16 中，携带标量过滤表达式（`expr`）的 HNSW 向量查询，存在两种核心执行路径，由 `hints` 参数控制：
1.  **标准前置过滤（默认路径，不指定 hints）**
    - 执行逻辑：先执行全量标量过滤，生成符合条件的向量位图（Bitset）；再在 HNSW 图遍历过程中，仅对位图内的向量执行相似度计算与排序，最终返回 TopK 结果。
    - 特点：过滤逻辑一次性完成，向量搜索阶段直接基于过滤后的子集执行；当过滤选择率高（过滤后剩余数据多）时，性能与召回率表现稳定。
    - 劣势：当过滤条件复杂、选择率极低（符合条件的向量极少）时，全量标量过滤开销大，且向量搜索阶段大量节点被跳过，HNSW 图遍历效率大幅下降。

2.  **迭代后置过滤（开启 `hints: iterative_filter`）**
    - 执行逻辑：利用 HNSW 索引的迭代能力，按距离由近到远逐批输出候选向量；每输出一批候选，就执行一次标量过滤校验；持续迭代直到收集到 `limit` 条有效结果，或遍历完全量向量。
    - 特点：无需预先全量过滤，边搜边滤，符合条件的结果靠前时可提前终止；适合过滤选择率低、过滤条件计算成本高的场景。
    - 劣势：若符合条件的向量普遍距离较远，需要多轮迭代，整体延迟可能高于前置过滤。

### 1.2 HNSW 索引查询核心原理
HNSW（层次化导航小世界）是基于多层邻接图的近似最近邻索引：
- 查询时通过贪心算法在多层图中遍历，维护一个大小为 `ef` 的动态候选队列，最终从队列中选出 TopK 结果。
- `ef` 是查询阶段唯一的精度控制参数，直接决定候选池大小与搜索探索宽度，是召回率与性能权衡的核心旋钮。

### 1.3 范围搜索执行逻辑
通过 `radius` 与 `range_filter` 可实现相似度范围搜索，替代纯 TopK 查询：
- 对于 **L2 欧氏距离**：`radius` 为最大距离阈值（仅返回距离 ≤ radius 的向量），`range_filter` 为最小距离阈值（仅返回距离 ≥ range_filter 的向量），二者共同构成有效区间 `[range_filter, radius]`。
- 对于 **IP 内积 / COSINE 余弦相似度**：`radius` 为最小相似度阈值，`range_filter` 为最大相似度阈值，有效区间为 `[radius, range_filter]`。
- 范围搜索模式下，查询不再以凑够 TopK 为终止条件，而是遍历所有符合距离阈值的结果，结果数量不固定。

---

## 二、查询参数深度影响分析
你必须准确理解每个参数对「召回率」「查询延迟/QPS」的影响方向与边际效应，以及参数间的协同关系。

### 2.1 `ef` 参数（HNSW 查询核心精度参数）
- **参数定位**：HNSW 图搜索时动态候选队列的大小，仅对 HNSW 系列索引生效，必须放置于 `search_params["params"]` 子字典中。
- **对召回率的影响**：正相关。`ef` 越大，搜索时探索的邻居节点越多，遗漏真实最近邻的概率越低，召回率越高；边际收益递减，`ef` 增大到一定程度后召回率趋近于上限。
- **对 QPS 的影响**：负相关。`ef` 越大，单查询的图遍历计算量越大，延迟越高，QPS 越低。
- **硬性约束**：`ef` 的取值必须 ≥ 查询的 `limit` 值，否则无法返回合法的 TopK 结果。

### 2.2 `iterative_filter` 过滤策略开关
- **参数定位**：通过 `search_params["hints"] = "iterative_filter"` 显式开启，不指定则默认使用前置过滤。
- **对召回率的影响**：相同 `ef` 下，迭代过滤的召回率略低于前置过滤（迭代过程中存在近似排序误差），但差距极小；当过滤选择率极低时，前置过滤会因 HNSW 跳节点导致召回率下降，此时迭代过滤召回率更稳定。
- **对 QPS 的影响**：
  - 高选择率场景（过滤后数据占比 > 30%）：迭代过滤性能弱于前置过滤，QPS 更低。
  - 低选择率场景（过滤后数据占比 < 5%）：迭代过滤性能显著优于前置过滤，QPS 更高。
- **与 `ef` 的协同效应**：开启迭代过滤时，`ef` 决定单轮迭代输出的候选数量：`ef` 过小会导致迭代轮次增多，总开销上升；`ef` 过大会导致单轮计算量过高，首轮过滤浪费算力。

### 2.3 `radius` 与 `range_filter` 范围参数
- **参数定位**：`search_params` 顶层参数，非必填，不填写则执行纯 TopK 查询。
- **对召回率的影响**：`radius` 越宽松（L2 下数值越大），返回结果越多，召回覆盖率越高；`radius` 越严格，结果越少，精准度越高。
- **对 QPS 的影响**：`radius` 越严格，符合条件的向量越少，搜索可提前终止，QPS 越高；`radius` 越宽松，遍历范围越大，QPS 越低。
- **约束规则**：`range_filter` 必须与 `radius` 配合使用，且二者的数值边界必须符合对应度量类型的大小逻辑，不可出现逻辑矛盾的区间。

### 2.4 参数间的互斥与补充
- 范围搜索与迭代过滤可同时开启，此时迭代输出的候选会同时经过距离阈值与标量条件双重过滤。
- 纯 TopK 场景下，`radius` 和 `range_filter` 应设为 `null`，不启用范围逻辑。
- 前置过滤与迭代过滤为二选一关系，无法同时生效。

---

## 三、查询策略生成核心原则与硬性规则
### 3.1 核心优化目标
所有策略均以**「召回率 × QPS 综合收益最大化」**为核心目标，在两个指标之间做差异化权衡，覆盖不同业务场景的需求：
- 性能导向型：牺牲少量召回率，换取更高 QPS 与更低延迟
- 均衡型：召回率与 QPS 处于中间区间，兼顾体验与成本
- 召回导向型：牺牲部分 QPS，换取更高的召回准确率

### 3.2 硬性约束规则（必须 100% 遵守，违反视为无效策略）
1.  **版本合规性**：所有参数必须符合 Milvus 2.4.16 的官方规范，不得引入不存在的参数或取值。
2.  **`ef` 取值约束**：所有策略的 `ef` 值必须 ≥ 输入信息中给出的查询 `limit` 值，且为正整数；取值范围建议控制在 `[limit, 2048]` 区间内，超出该区间需有明确的场景合理性。
3.  **范围参数逻辑约束**：
    - 若启用范围搜索，必须同时符合度量类型的数值大小逻辑；
    - 若不启用范围搜索，`radius` 和 `range_filter` 字段必须设为 `null`；
    - 不得仅设置 `range_filter` 而不设置 `radius`。
4.  **参数有效性**：所有参数组合必须具备实际业务意义，不得出现逻辑矛盾、无价值的参数组合。
5.  **数量约束**：必须恰好生成 20 个策略，编号从 1 到 20，不得多也不得少。

### 3.3 策略生成指导原则
1.  **全梯度覆盖原则**
    20 个策略需形成连续的权衡梯度，覆盖从「极致 QPS 优先」到「极致召回优先」的完整区间，每个档位的参数差异要可感知，避免参数高度接近的重复策略。

2.  **多维度差异化原则**
    策略需在多个维度上形成差异，不得仅调整 `ef` 单一参数：
    - 过滤模式维度：包含前置过滤（关闭 iterative_filter）、迭代过滤（开启 iterative_filter）两大类策略；
    - 查询模式维度：包含纯 TopK 查询、宽松范围查询、严格范围查询三类；
    - 适配场景维度：覆盖高选择率适配、低选择率适配、通用场景适配三类。

3.  **场景适配原则**
    结合输入的标量过滤选择率，匹配最优的过滤模式：
    - 高选择率场景（过滤后剩余数据占比高）：以前置过滤策略为主；
    - 低选择率场景（过滤后剩余数据占比低）：增加迭代过滤策略的占比；
    - 每个策略需明确标注其最适配的业务场景。

4.  **参数协同最优原则**
    避免参数组合的内耗：
    - 开启迭代过滤时，`ef` 取值需匹配迭代轮次平衡，避免过小或过大；
    - 严格范围查询下，`ef` 无需设置过高，满足范围遍历即可；
    - 高召回导向策略需同步考虑过滤模式对召回率的影响。

---

## 四、输出格式严格定义
你的所有输出必须是一个合法的 JSON 字符串，**不得包含任何额外的解释文字、前言、备注、markdown 格式或代码块标记**，直接输出纯 JSON 内容。

输出为一个包含 **20 个元素**的 JSON 数组，每个元素代表一个查询策略，字段定义如下：

| 字段名 | 数据类型 | 字段说明 | 取值要求 |
|--------|----------|----------|----------|
| `strategy_id` | integer | 策略唯一编号 | 从 1 到 20 顺序编号 |
| `ef` | integer | HNSW 查询的 ef 参数值 | 正整数，必须 ≥ 查询 limit 值 |
| `enable_iterative_filter` | boolean | 是否开启迭代过滤 | `true` = 开启 `hints: iterative_filter`；`false` = 使用默认前置过滤 |
| `radius` | number / null | 范围搜索的半径阈值 | 启用范围搜索时填写具体数值；纯 TopK 查询填 `null` |
| `range_filter` | number / null | 范围搜索的过滤边界 | 配合 radius 使用；纯 TopK 查询填 `null` |
| `strategy_orientation` | string | 策略定位与权衡倾向 | 例如：极致QPS导向、QPS优先均衡型、均衡型、召回优先均衡型、极致召回导向 |
| `applicable_scenario` | string | 适用场景说明 | 用一句话描述该策略最适合的业务场景 |

### 格式示例（仅示例结构，非推荐值）
```json
[
  {
    "strategy_id": 1,
    "ef": 100,
    "enable_iterative_filter": false,
    "radius": null,
    "range_filter": null,
    "strategy_orientation": "极致QPS导向",
    "applicable_scenario": "高选择率过滤、对延迟要求极高的高并发在线场景"
  }
]
```

---

## 五、最终输出要求
1.  严格遵守上述所有硬性规则，确保所有参数合法、逻辑自洽；
2.  20 个策略需具备明显差异化，避免参数高度相似的无效重复；
3.  策略排序建议按「QPS 从高到低、召回率从低到高」的顺序排列，便于对比选型；
4.  输出必须为纯 JSON，不得包含任何额外文字，不得添加代码块、注释、说明文字等内容；
5.  所有分析与推理过程内化，仅输出最终的策略 JSON 数组。

---
"""
]


def get_server_time_ms(response, response_headers: dict) -> float | None:
    """从响应头或响应对象中提取服务端计算时间(毫秒)"""
    server_time_ms = None

    # 常见的服务端时间头字段 (按优先级)
    header_names = [
        'x-envoy-upstream-service-time',  # Envoy 代理上报的服务端时间
        'req-cost-time',  # 请求处理时间
        'x-process-time-ms',
        'x-response-time-ms',
        'x-latency-ms',
        'process-time-ms',
        'response-time-ms',
        'latency',
        'server-timing',
    ]

    # 从响应头中获取
    for header_name in header_names:
        if header_name in response_headers:
            header_value = response_headers[header_name]
            try:
                server_time_ms = float(header_value)
                break
            except (ValueError, TypeError):
                pass

    # 如果 header 中没有，尝试从响应对象的其他属性获取
    if server_time_ms is None:
        model_extra = getattr(response, 'model_extra', None)
        if model_extra:
            server_time_ms = (model_extra.get('response_ms') or
                            model_extra.get('latency') or
                            model_extra.get('server_time_ms'))

    # 尝试从 litellm 的其他属性获取
    if server_time_ms is None:
        if hasattr(response, 'response_ms'):
            server_time_ms = response.response_ms
        elif hasattr(response, 'extra_data') and response.extra_data:
            server_time_ms = response.extra_data.get('response_ms') or response.extra_data.get('latency')

    return server_time_ms


def main():
    """Initialize LLM client and interact with multiple messages."""

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY") #TODO
    # api_key = "sk-24Mf5wY2RvofQHgknlCakLhQgd2ZSP1it14GvhxF9WWvqW6T"
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return

    # 创建响应头存储变量
    response_headers = {}

    # 使用 httpx transport 来拦截响应头
    class HeaderCaptureTransport(httpx.HTTPTransport):
        def handle_request(self, request):
            response = super().handle_request(request)
            # 存储响应头供后续使用
            response_headers.update(dict(response.headers))
            return response

    # Initialize client with custom http client that disables proxy
    http_client = httpx.Client(
        trust_env=False,
        timeout=DEFAULT_TIMEOUT,
        transport=HeaderCaptureTransport(),
    )

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        http_client=http_client,
    )

    print(f"[INFO] Client initialized successfully")
    print(f"[INFO] Base URL: {BASE_URL}")
    print(f"[INFO] Model: {MODEL_NAME}")
    print(f"[INFO] Output file: {OUTPUT_FILE}")
    # print(f"[INFO] SQL content length: {len(SQL_CONTENT)}")

    # print(f"\n[INFO] System prompt: {SYSTEM_PROMPT}")
    # print(f"=" * 60)

    # Initialize messages with system prompt
    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT}
    # ]
    messages = []

    # # Replace placeholders in prompts with actual values
    # processed_prompts = [p.replace("{SQL_CONTENT}", SQL_CONTENT).replace("{STAT_CONTENT}", STAT_CONTENT).replace("{PROPOSAL_COUNT}", PROPOSAL_COUNT) for p in PROMPTS]

    # Process each prompt
    for i, prompt in enumerate(PROMPTS):
        # print(f"\n[INFO] Prompt#{i+1} prompt length: {len(prompt)}: (SQL content length: {len(SQL_CONTENT)})")
        print(f"\n[INFO] Prompt#{i+1} prompt length: {len(prompt)}")
        # print(f"------\n {prompt} \n")

        # Record client start time (before sending request)
        client_start_time = time.time()

        # Add user message to conversation
        prompt = prompt.replace("{SELECTIVE}", str(SELECTIVE))
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                timeout=DEFAULT_TIMEOUT,
            )

            # Calculate client-side time
            client_end_time = time.time()
            client_elapsed = client_end_time - client_start_time

            # Extract server time from response headers
            server_time_ms = get_server_time_ms(response, response_headers)

            # Extract response content
            assistant_message = response.choices[0].message.content

            # Estimate server processing time
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                # Rough estimate: Assuming ~50 tokens/sec generation rate for 30B model
                estimated_server_time = completion_tokens / 50.0 if completion_tokens > 0 else 0
            else:
                prompt_tokens = completion_tokens = total_tokens = 0
                estimated_server_time = 0

            # Add assistant response to messages for context
            messages.append({"role": "assistant", "content": assistant_message})

            # Write assistant_message to output file
            try:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    f.write(assistant_message)
                print(f"[INFO] Output written to: {OUTPUT_FILE}")
            except Exception as e:
                print(f"[ERROR] Failed to write output file: {e}")

            print(f"[INFO] Client time: {client_elapsed * 1000:.1f} ms")
            if server_time_ms is not None:
                print(f"[INFO] Server computation time: {server_time_ms:.1f} ms")
            else:
                print(f"[INFO] Server computation time: ~{estimated_server_time * 1000:.1f} ms (estimated)")

            if total_tokens > 0:
                print(f"[INFO] Tokens - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")

        except Exception as e:
            print(f"[ERROR] Failed to get response: {e}")

    print(f"\n{'=' * 60}")
    print(f"[INFO] All prompts processed successfully!")


if __name__ == "__main__":
    main()