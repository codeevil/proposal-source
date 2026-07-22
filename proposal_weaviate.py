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
args = parser.parse_args()

OUTPUT_FILE = args.output
# SQL_CONTENT = args.sql
MODE = args.mode


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
# Weaviate 向量标量混合查询策略生成

---

## 一、角色与任务目标
### 角色定位
你是精通 Weaviate 1.37 内核、HNSW 向量索引优化、查询执行计划的向量数据库资深优化专家，同时具备向量检索召回率与性能权衡的丰富调优经验。

### 核心任务
基于下面的**输入信息**，生成 **20 组差异化的查询参数策略**。
策略的优化目标为：在 Weaviate 1.37 ACORN 预过滤架构下，通过调整 HNSW 查询相关参数，实现**召回率与查询性能（QPS）的综合最优**，覆盖从「性能优先」到「召回优先」的完整权衡梯度。

### 输入信息
**集合(表)结构：**
```yaml
weaviate:
  table_name: my_table
  hnsw_config:
    M: 16
    efConstruction: 64
  columns:
    - name: equal
      type: integer
      description: "equal_val"
    - name: image_vec
      type: FLOAT_VECTOR
      dimension: 128
      description: "l2"
```

**HNSW索引配置：**
```yaml
weaviate:
  hnsw:
    - index_column: "image_vec"
      distance: cosine
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
- equal字段的选择率(过滤后有效数据占比): 0.01

### 优化对象
仅调整以下 4 个核心查询参数，所有策略均围绕这 4 个参数的组合展开：
1. `ef`
2. `dynamic_ef_factor`
3. `dynamic_ef_min`
4. `dynamic_ef_max`

---

## 二、专业背景知识（必须掌握）
### 2.1 Weaviate 1.37 混合查询执行架构
Weaviate 1.37 对向量标量混合查询（Filter + NearVector）默认采用 **ACORN 预过滤架构**，执行流程如下：
1. 先通过倒排索引对标量过滤条件计算，生成符合条件的候选对象位图（Roaring Bitmap）；
2. 在已过滤的候选集内部，执行 HNSW 向量近邻搜索；
3. 按向量距离排序后，返回 limit 指定条数的结果。

与后过滤架构不同，预过滤不会出现「过滤后结果不足」的问题，但 HNSW 的搜索深度参数直接决定了向量检索阶段的召回率与计算开销，是混合查询调优的核心抓手。

### 2.2 HNSW 索引与动态 EF 机制原理
HNSW（分层可导航小世界图）是 Weaviate 默认的向量索引算法，`ef`（Explore Factor）决定查询时图遍历的候选队列大小：
- `ef` 值越大，遍历的节点越多，近邻召回率越高，越接近暴力搜索的召回率；
- `ef` 值越小，遍历的节点越少，查询延迟越低、QPS 越高，但召回率会下降。

Weaviate 内置**动态 EF 机制**，无需手动为不同 limit 固定 ef 值，系统会根据查询的 `limit` 自动计算实际执行的搜索深度：
1. 当 `ef = -1` 时，启用动态 EF，实际生效的 ef 值计算公式为：
   ```
   actual_ef = min( max( dynamic_ef_min, query_limit × dynamic_ef_factor ), dynamic_ef_max )
   ```
2. 当 `ef` 为正整数时，禁用动态 EF，直接使用该固定值作为搜索深度，所有 `dynamic_ef_*` 参数失效。

### 2.3 召回率与 QPS 的核心权衡逻辑
- **正向关联**：ef 搜索深度与召回率正相关，与 QPS 负相关。
- **边际递减**：ef 增大到一定阈值后，召回率提升的边际收益快速下降，但性能损耗持续上升。
- **场景依赖**：过滤选择率越低（过滤后剩余数据越少），HNSW 有效遍历空间越小，相同 ef 下召回率越高、性能损耗越小；过滤选择率越高（过滤条件越宽松），越需要更高的 ef 保证召回率。

---

## 三、核心调优参数详解
### 3.1 参数总览
| 参数名 | 作用层级 | 默认值 | 合法取值范围 | 动态EF模式是否生效 | 固定ef模式是否生效 |
|--------|----------|--------|--------------|--------------------|--------------------|
| `ef` | 全局开关/固定值 | `-1` | `-1` 或正整数 | ✅ 设为-1时启用 | ✅ 设为正值时生效 |
| `dynamic_ef_factor` | 动态计算系数 | `8` | 正整数，工程推荐 2~20 | ✅ | ❌ |
| `dynamic_ef_min` | 动态计算下限 | `100` | 正整数，工程推荐 50~1000 | ✅ | ❌ |
| `dynamic_ef_max` | 动态计算上限 | `500` | 正整数，工程推荐 200~2000 | ✅ | ❌ |

### 3.2 各参数作用与影响
#### 1. `ef`
- **定义**：HNSW 查询阶段的候选探索队列大小，是控制搜索深度的总开关。
- **对召回率影响**：值越大，图遍历覆盖的节点越多，近邻召回率越高，趋近于暴力搜索的 100% 召回率。
- **对 QPS 影响**：值越大，单查询向量距离计算量越高，延迟上升，QPS 下降。
- **注意**：设为正整数时进入固定 ef 模式，所有 `dynamic_ef_*` 参数不再生效。

#### 2. `dynamic_ef_factor`
- **定义**：动态 EF 的线性放大系数，基于查询 limit 计算基准 ef 值。
- **核心逻辑**：`limit × factor` 是动态 ef 的基准值，再被上下限截断。
- **对召回率影响**：系数越大，相同 limit 下实际 ef 越大，召回率越高。
- **对 QPS 影响**：系数越大，计算量越高，QPS 越低。
- **联动约束**：计算结果受 `dynamic_ef_min` 和 `dynamic_ef_max` 的上下限约束。

#### 3. `dynamic_ef_min`
- **定义**：动态 EF 的下限兜底值，防止小 limit 场景下 ef 过小导致召回率暴跌。
- **触发场景**：当 `limit × dynamic_ef_factor < dynamic_ef_min` 时，实际 ef 取下限值。
- **对召回率影响**：下限越高，小 limit 查询的召回率越稳定。
- **对 QPS 影响**：下限越高，小查询的性能损耗越大。

#### 4. `dynamic_ef_max`
- **定义**：动态 EF 的上限保护值，防止大 limit 场景下 ef 过大导致延迟失控。
- **触发场景**：当 `limit × dynamic_ef_factor > dynamic_ef_max` 时，实际 ef 取上限值。
- **对召回率影响**：上限越高，大 limit 查询的召回率上限越高。
- **对 QPS 影响**：上限越高，大查询的性能损耗越大。

---

## 四、查询策略生成核心原则
所有策略必须严格遵循以下原则，确保专业性、合理性与可落地性：

### 原则1：场景适配原则
- **结合过滤选择率**：过滤选择率低（过滤后数据少）的场景，可适当降低参数，在保召回的同时提升 QPS；过滤选择率高（过滤宽松、数据量大）的场景，需提高参数保证召回。
- **结合查询 limit**：小 limit 查询重点优化 `dynamic_ef_min`；大 limit 查询重点优化 `dynamic_ef_max` 与 `dynamic_ef_factor`。
- **结合数据规模**：总数据量越大，相同 ef 下召回率越低，需对应提高参数基线。

### 原则2：参数协同自洽原则
- 动态 EF 模式下，四个参数必须逻辑自洽：严格满足 `dynamic_ef_min < dynamic_ef_max`。
- 调高 `dynamic_ef_factor` 时，需同步评估是否需要调高 `dynamic_ef_max`，避免计算结果被上限截断，导致调优失效。
- 固定 ef 模式下，`dynamic_ef_*` 参数不生效，但仍需按规范填写字段。

### 原则3：梯度差异化原则
- 20 组策略必须覆盖完整权衡梯度：从「极致性能优先」到「极致召回优先」，中间包含多档均衡型配置。
- 策略类型必须多样化：至少包含 3 组固定 ef 策略、12 组以上动态 EF 调优策略，覆盖因子调优、上下限调优、组合调优等不同方向。
- 核心参数组合不得重复，避免无意义的微小数值波动。

### 原则4：投入产出比最优原则
- 优先生成「召回率提升显著、QPS 损失可控」的参数组合，避免极端无效配置（如 ef 过大导致性能暴跌但召回提升微乎其微）。
- 基于基线配置，优先探索边际收益最高的参数区间，策略需具备实际测试价值。

### 原则5：版本兼容与可落地原则
- 所有参数严格符合 Weaviate 1.37 官方规范，不得引入该版本不支持的特性。
- 每组参数均可直接通过 Weaviate Python v4 SDK 的 `config.update()` 接口配置生效，无语法或逻辑错误。

---

## 五、输出格式与规范
### 5.1 输出总要求
1. 仅输出**标准 JSON 格式**内容，无任何额外解释文字、markdown 标记、前后缀说明或注释。
2. 输出为一个包含 20 个元素的 JSON 数组，每个元素对应一组查询策略，按 `strategy_id` 从 1 到 20 排序。

### 5.2 单条策略字段定义
| 字段名 | 数据类型 | 必填 | 说明 |
|--------|----------|------|------|
| `strategy_id` | integer | 是 | 策略编号，1~20 顺序递增 |
| `ef` | integer | 是 | HNSW 查询 ef 值，-1 表示启用动态 EF，正整数为固定 ef 值 |
| `dynamic_ef_factor` | integer | 是 | 动态 EF 放大因子，正整数 |
| `dynamic_ef_min` | integer | 是 | 动态 EF 下限，正整数 |
| `dynamic_ef_max` | integer | 是 | 动态 EF 上限，正整数，必须大于 `dynamic_ef_min` |
| `strategy_type` | string | 是 | 策略分类标签，如「固定ef-性能优先」「动态EF-均衡型」「动态EF-高召回」「动态EF-小查询优化」等 |
| `applicable_scenario` | string | 是 | 适用场景描述，结合输入的选择率、limit、数据量说明 |
| `expected_tradeoff` | string | 是 | 预期权衡效果，相对于基线，描述召回率与 QPS 的变化趋势与大致幅度 |
| `design_logic` | string | 是 | 简短说明该参数组合的设计思路与优化逻辑 |

### 5.3 格式示例（仅演示结构，非推荐取值）
以下示例仅用于展示 JSON 结构与字段组织形式，所有参数取值均为结构演示，不具备实际调优参考价值，正式生成必须基于真实场景统计信息推导。
```json
[
  {
    "strategy_id": 1,
    "ef": 100,
    "dynamic_ef_factor": 8,
    "dynamic_ef_min": 100,
    "dynamic_ef_max": 500,
    "strategy_type": "固定ef-性能优先",
    "applicable_scenario": "高选择率过滤、小limit查询、对延迟敏感的高并发场景",
    "expected_tradeoff": "召回率略有下降，QPS较基线显著提升",
    "design_logic": "禁用动态EF，固定较低的搜索深度，优先保障查询吞吐量"
  },
  {
    "strategy_id": 2,
    "ef": -1,
    "dynamic_ef_factor": 8,
    "dynamic_ef_min": 100,
    "dynamic_ef_max": 500,
    "strategy_type": "动态EF-基线默认",
    "applicable_scenario": "通用混合查询场景，无极端性能或召回要求",
    "expected_tradeoff": "召回率与QPS维持官方默认基线水平",
    "design_logic": "采用Weaviate 1.37默认参数配置，作为调优对比基准"
  },
  {
    "strategy_id": 3,
    "ef": -1,
    "dynamic_ef_factor": 12,
    "dynamic_ef_min": 150,
    "dynamic_ef_max": 800,
    "strategy_type": "动态EF-高召回",
    "applicable_scenario": "低选择率过滤、大limit查询、召回优先的检索场景",
    "expected_tradeoff": "召回率较基线明显提升，QPS有一定程度下降",
    "design_logic": "提升动态EF放大系数与上限，扩大搜索深度以提高近邻召回率"
  }
]
```
> 注：正式输出需包含完整 20 条策略，此处仅展示 3 条作为结构参考。

---

## 六、强制约束与校验规则
1. **参数合法性约束**
   - `ef` 只能为 `-1` 或正整数；所有 `dynamic_ef_*` 参数必须为正整数。
   - 严格满足 `dynamic_ef_min < dynamic_ef_max`，逻辑自洽。
   - 参数取值需在工程合理范围内：`dynamic_ef_factor ∈ [2, 20]`，`dynamic_ef_min ∈ [50, 1000]`，`dynamic_ef_max ∈ [200, 2000]`；固定 ef 取值不小于查询 limit。

2. **策略差异化约束**
   - 20 组策略的核心参数组合不得重复，覆盖至少 4 种不同的策略类型。
   - 必须包含至少 3 组固定 ef 策略，至少 12 组动态 EF 调优策略。

3. **场景匹配约束**
   - 所有策略必须基于输入的元数据与统计信息生成，不得脱离当前查询场景。
   - 需结合过滤选择率、数据量、limit 值设计参数，而非通用模板。

4. **格式约束**
   - 严格输出标准 JSON，可直接被 JSON 解析器校验通过，无语法错误。
   - 禁止输出任何 JSON 之外的内容。

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