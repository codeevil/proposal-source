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
parser.add_argument("--sql", type=str, required=True,
                    help="SQL content (required)")
parser.add_argument("--mode", type=str, default="standard",
                    choices=["extreme", "standard", "balance"],
                    help="Generation mode: extreme (15 diverse strategies), "
                         "standard (default, 10 strategies), balance (6 balanced strategies)")
args = parser.parse_args()

OUTPUT_FILE = args.output
SQL_CONTENT = args.sql
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
STAT_FILE = "/home/liujianzhong/proposal-source/stat.txt"
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
SYSTEM_PROMPT = """
你是一位**PostgreSQL优化器内核专家**和**pgvector向量数据库资深架构师**，精通：
- pgvector扩展的内部实现机制（HNSW/IVFFlat索引构建、查询执行流程、代价估算模型）
- PostgreSQL查询优化器的行为特性与局限性
- 向量标量混合查询的各类性能瓶颈与解决方案
- pg_hint_plan扩展的精确使用方法
- 大规模向量数据集的查询性能调优技术
"""

TEMPLATE_CONTENT = """
```json
[
  {
    "strategy_id": <整数，1 ~ {PROPOSAL_COUNT}，按顺序编号>,
    "name": "字符串，策略简要名称，不超过20字",
    "description": "字符串，结合数据分布与选择性数值，说明设计思想、优势与潜在代价，必须明确引用选择性数值",
    "vector_index_used": "字符串，索引名：my_table_image_vec_idx / my_table_ivf_image_vec_idx / null（无索引，精确搜索）",
    "index_parameters": {
      "type": "字符串，HNSW / IVFFlat / none",
      "ef_search": "整数，仅HNSW有效，其余为null",
      "probes": "整数，仅IVFFlat有效，其余为null",
      "distance_type": "字符串，L2 / inner_product / cosine，与原查询保持一致"
    },
    "filter_strategy": "字符串，pre-filter / post-filter",
    "sql_rewrite_required": "布尔值，true / false",
    "rewritten_sql": "字符串，sql_rewrite_required为true时填写完整改写后的SQL；否则为空字符串",
    "hint": "字符串，使用pg_hint_plan时填写准确提示语句；否则为空字符串，严格遵循语法规范",
    "refill_fallback_required": "布尔值，true / false",
    "refill_parameters": {
      "hnsw.iterative_scan": "字符串，off / strict_order / relaxed_order，仅HNSW有效，其余为null",
      "hnsw.max_scan_tuples": "整数，仅HNSW有效，其余为null",
      "ivfflat.iterative_scan": "字符串，off / relaxed_order，仅IVFFlat有效，其余为null",
      "ivfflat.max_probes": "整数，仅IVFFlat有效，其余为null"
    },
    "notes": "字符串，补充说明执行边界、回退条件、适用场景限制等，不超过150字"
  }
]
```
"""

# "你是PostgreSQL+pgvector领域的数据库优化专家，精通向量标量混合查询的执行计划优化、索引调优与代价估算。"
PROMPTS = [
f"""
请基于下方提供的向量标量混合查询、数据库元数据、表统计信息，严格按照要求生成 **{PROPOSAL_COUNT} 个差异化的混合查询执行策略**。

所有策略必须基于统计信息中的选择性估算推导，覆盖不同技术方向与权衡偏好；最终仅输出符合指定Schema的JSON数组，禁止输出任何额外解释、说明或markdown格式。

---

## 一、环境版本约束（硬约束，不可违背）
- PostgreSQL 版本：15.5
- pgvector 版本：0.8.1
- pg_hint_plan 版本：1.5.1
- 存储引擎：堆表（Heap）

---

## 二、输入信息说明
### 1. 目标查询语句
向量标量混合查询（同时包含标量过滤条件与向量相似度搜索）：
```sql
{SQL_CONTENT}
```

### 2. 数据库元数据（表结构与索引）
```
                                             Table "public.my_table"
  Column   |    Type     | Collation | Nullable | Default | Storage  | Compression | Stats target | Description
-----------+-------------+-----------+----------+---------+----------+-------------+--------------+-------------
 id        | integer     |           | not null |         | plain    |             |              |
 equal     | smallint    |           |          |         | plain    |             |              |
 image_vec | vector(128) |           |          |         | external |             |              |
Indexes:
    "my_table_pkey" PRIMARY KEY, btree (id)
    "my_table_image_vec_idx" hnsw (image_vec vector_l2_ops) WITH (m='32', ef_construction='300')
    "my_table_ivf_image_vec_idx" ivfflat (image_vec vector_l2_ops) WITH (lists='1000')
Access method: heap
```

**索引明细**：
- HNSW索引：`my_table_image_vec_idx`，L2距离算子，构建参数 m=32、ef_construction=300
- IVFFlat索引：`my_table_ivf_image_vec_idx`，L2距离算子，构建参数 lists=1000

### 3. 表统计信息
- 表总行数：1000000 行
- 向量维度：128 维
- 数据分布与标量条件选择性：
```
{STAT_CONTENT}
```

---

## 三、核心优化目标
生成的执行策略需在两个核心目标间做差异化权衡，覆盖不同偏好的策略组合：
1. **高召回率**：向量搜索返回的结果中，真实最近邻的占比尽可能高，避免漏检最优结果
2. **高QPS**：单查询延迟尽可能低，单位时间可处理的查询量尽可能大

策略需覆盖「极致QPS、平衡型、极致召回」三类偏好，同时结合标量条件的选择性，选择最优的执行路径。

---

## 四、必备领域知识库
### 1. pgvector 核心特性
- 支持L2距离、内积、余弦相似度三种向量距离计算
- 提供HNSW、IVFFlat两类近似最近邻（ANN）索引，同时支持精确搜索（顺序扫描）
- 支持向量字段与标量字段的联合查询，但原生优化器对向量索引的代价估算存在偏差
- 向量维度最高支持65535维，本场景固定为128维

### 2. 混合查询执行路径对比
| 执行路径       | 原理                                  | 适用场景                     | 优势                  | 劣势                          |
|----------------|---------------------------------------|------------------------------|-----------------------|-------------------------------|
| Post-Filter    | 先执行向量ANN搜索，再对标量结果过滤   | 低选择性（过滤后剩余>50%）   | 向量索引效率最大化    | 过滤后可能出现结果行数不足    |
| Pre-Filter     | 先执行标量过滤，再对结果集做向量搜索  | 高选择性（过滤后剩余<10%）   | 向量计算量大幅降低    | 结果集过小时无法利用向量索引  |
| 重排序扩大召回 | 先ANN返回N倍LIMIT结果，过滤后重排序   | 中选择性，召回要求高         | 召回率显著提升        | 延迟随扩大倍数线性上升        |
| 迭代回退       | 自动扩大向量搜索范围直到满足行数要求  | 中低选择性，行数稳定性要求高 | 单次查询即可满足行数  | 极端场景下延迟波动大          |

### 3. 优化技术池
- **索引选型**：HNSW / IVFFlat / 无索引精确搜索，及运行时参数调优
- **过滤时机**：Post-Filter / Pre-Filter
- **查询重写**：子查询加 OFFSET 0 、CTE 等方式强制执行执行顺序
- **迭代回退（Refill Fallback）**：通过迭代扫描自动扩大搜索范围，补足过滤后缺失的行数
- **执行计划提示**：通过pg_hint_plan强制索引选择、禁止顺序扫描、固定连接顺序
- **部分索引**：利用带标量条件的向量部分索引，实现Pre-Filter效果同时保留向量索引效率
- **近似转精确回退**：ANN结果不足时自动回退为精确顺序扫描

### 4. pg_hint_plan 语法参考
- 强制指定索引：`/*+ IndexScan(表名 索引名) */`
- 禁止顺序扫描：`/*+ NoSeqScan(表名) */`
- 强制顺序扫描：`/*+ SeqScan(表名) */`
- 提示需放置在SELECT关键字后，语法严格匹配

### 5. SQL改写参考范式
**仅Pre-Filter场景需要SQL改写，SQL改写需严格遵循这里的参考范式**
- 物化 CTE 的方式
  举例：
  ```sql
  WITH filtered AS MATERIALIZED (
    SELECT id, image_vec 
    FROM my_table 
    WHERE [标量过滤条件]
  )
  SELECT id 
  FROM filtered 
  ORDER BY image_vec <-> (SELECT image_vec FROM my_table WHERE [id字段过滤条件]) 
  LIMIT 100;
  ```

- 子查询加 OFFSET 0 优化屏障的方式
  举例：
  ```sql
  SELECT id 
  FROM (
    SELECT id, image_vec 
    FROM my_table 
    WHERE equal = 23 
    OFFSET 0
  ) t
  ORDER BY image_vec <-> (SELECT image_vec FROM my_table WHERE [id字段过滤条件]) 
  LIMIT 100;
  ```

---

## 五、策略生成核心原则
### 1. 向量索引参数取值规则
#### （1）HNSW 索引：hnsw.ef_search
- 取值范围：1 ~ 1000（整数），**必须大于查询的LIMIT值**
- 取值梯度（K为查询SQL LIMIT数值）：
  - 极速档：K * 1.5 ~ K * 2 → 优先QPS，召回率一般
  - 平衡档：K * 3 ~ K * 5 → 召回与性能均衡，推荐基线配置
  - 高召回档：K * 8 ~ K * 10 → 优先召回率，延迟线性上升
  - 极高召回档：100 ~ 200 → 小K值场景下最大化召回，上限1000
- 影响规律：值越大，遍历候选点越多，召回率越高，查询延迟近似线性上升

#### （2）IVFFlat 索引：ivfflat.probes
- 取值范围：1 ~ 1000（与索引lists数一致），默认值1
- 取值梯度（N为索引lists总数=1000）：
  - 极速档：1 ~ 2 → 极致QPS，召回率最低
  - 平衡档：N * 1% ~ N * 3%（10 ~ 30）→ 召回与性能均衡
  - 高召回档：N * 5% ~ N * 10%（50 ~ 100）→ 召回率显著提升
  - 极高召回档：N * 20% ~ N * 50%（200 ~ 500）→ 接近精确搜索，延迟大幅升高
- 影响规律：值越大，扫描的聚类桶越多，召回率越高，查询耗时近似线性上升

#### （3）迭代扫描（Refill Fallback）参数
- 启用条件：Post-Filter场景下，标量过滤可能导致结果行数不足LIMIT时启用
- 模式选择：
  - `strict_order`：严格保持距离排序，召回准确，性能稍低，适合排序精度要求高的场景
    + **注意**：IVFFlat不支持`strict_order`
  - `relaxed_order`：放宽排序约束，性能更优，优先满足行数要求，适合对排序精度要求一般的场景
- HNSW 迭代参数：
  - `hnsw.max_scan_tuples`：单次查询最大扫描向量数，取值建议 = 目标返回行数 / 标量选择性 * 倍率
    - 保守回退：目标行数 / 选择性 * 2
    - 中等回退：目标行数 / 选择性 * 5
    - 激进回退：20000（默认上限）
  - 取值范围：1 ~ 1000000
- IVFFlat 迭代参数：
  - `ivfflat.max_probes`：迭代过程中最大探测聚类数，取值建议为初始probes的3~10倍，不超过lists总数
  - 取值范围：1 ~ 1000

### 2. 召回率与QPS平衡原则
1. **选择性适配**：
   - 高选择性（过滤后结果占比 < 10%）：优先Pre-Filter，减少向量计算量，大幅提升QPS；可搭配精确搜索
   - 中选择性（10% ~ 50%）：优先Post-Filter + 适度扩大召回 + 迭代回退，平衡召回与性能
   - 低选择性（> 50%）：优先Post-Filter + 标准参数，过滤开销低，最大化向量索引效率
2. **索引选型适配**：
   - QPS优先：选择HNSW + 低档位ef_search，或IVFFlat + probes=1
   - 召回优先：选择HNSW + 高档位ef_search，或IVFFlat + 高比例probes
   - 内存受限：优先选择IVFFlat（内存占用约为HNSW的1/3~1/2）
3. **策略多样性要求**：
   {PROPOSAL_COUNT}个策略必须覆盖以下至少6类方向，不得出现3个以上仅参数微调的同质化策略：
   - 基线策略（HNSW/IVFFlat默认参数Post-Filter）
   - HNSW参数梯度调优（低/中/高ef_search）
   - IVFFlat参数梯度调优（低/中/高probes）
   - Pre-Filter策略（子查询加 OFFSET 0 的方式/CTE的方式）
   - 迭代回退策略（HNSW/IVFFlat + Refill Fallback）
   - 特殊优化策略（精确搜索回退、重排序扩大召回、部分索引思路等）

### 3. 其他通用规则
- 仅Pre-Filter场景需要SQL改写或pg_hint_plan提示，用于强制执行先过滤后向量搜索的顺序
- Pre-Filter场景的SQL改写，请严格按照**SQL改写参考范式**中所列出来的范式进行改写
- Post-Filter场景可直接通过pg_hint_plan指定向量索引，无需改写SQL
- 每个策略的设计必须结合统计信息中的选择性数值，说明该选择性下策略的收益与代价
- 所有参数取值必须符合版本约束，不得超出合法范围

---

## 六、输出格式与Schema
### 输出要求
- 仅输出JSON数组，数组包含{PROPOSAL_COUNT}个策略对象
- 严格遵循下方Schema，字段名、类型完全匹配，无多余字段
- 未使用的参数字段统一设为`null`，不得留空或省略

### JSON Schema
{TEMPLATE_CONTENT}

---

## 七、输出强制校验规则
1. 所有数值参数必须在合法范围内，ef_search必须大于查询LIMIT值
2. 每个策略的description必须明确引用统计信息中的选择性数值，不得泛泛而谈
3. Pre-Filter策略必须配套SQL改写或pg_hint_plan提示，确保执行顺序可控
4. 如果使用Pre-Filter,涉及到SQL改写，请严格按照**SQL改写参考范式**中所列出来的范式进行改写
5. refill_fallback_required为true时，必须配置对应索引类型的迭代参数，且iterative_scan不得为off
6. 策略之间必须有明确差异，禁止重复或高度同质化的策略
7. 仅输出JSON数组，无任何前置、后置说明文字，无markdown格式，无代码块包裹

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
    print(f"[INFO] SQL content length: {len(SQL_CONTENT)}")

    # print(f"\n[INFO] System prompt: {SYSTEM_PROMPT}")
    # print(f"=" * 60)

    # Initialize messages with system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # # Replace placeholders in prompts with actual values
    # processed_prompts = [p.replace("{SQL_CONTENT}", SQL_CONTENT).replace("{STAT_CONTENT}", STAT_CONTENT).replace("{PROPOSAL_COUNT}", PROPOSAL_COUNT) for p in PROMPTS]

    # Process each prompt
    for i, prompt in enumerate(PROMPTS):
        print(f"\n[INFO] Prompt#{i+1} prompt length: {len(prompt)}: (SQL content length: {len(SQL_CONTENT)})")
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