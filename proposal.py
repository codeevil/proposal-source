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
# parser.add_argument("--stat", type=str, default="",
#                     help="Statistics content (optional, default: empty)")
args = parser.parse_args()

OUTPUT_FILE = args.output
SQL_CONTENT = args.sql
# STAT_CONTENT = args.stat


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
PROPOSAL_COUNT = "15"

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
{
  "strategy_id": <1 - PROPOSAL_COUNT>,
  "name": "策略简要名称",
  "description": "结合本数据分布（选择性约36.4%）说明该策略的设计思想、优点及潜在代价，务必引用选择性数值。",
  "vector_index_used": "索引名，如 my_table_image_vec_idx 或 my_table_ivf_image_vec_idx 或 null（表示不使用向量索引，需精确搜索）",
  "index_parameters": {
    "type": "HNSW 或 IVFFlat 或 none",
    "ef_search": <仅HNSW,HNSW搜索时动态参数，若有>,
    "probes": <仅IVFFlat,IVFFlat探测数，若设定>,
    "distance_type": "L2 / inner_product / cosine"
  },
  "filter_strategy": "pre-filter / post-filter",
  "sql_rewrite_required": true/false,
  "rewritten_sql": "若需要改写，提供完整 SQL；若不需要，设为空字符串",
  "hint": "若使用 pg_hint_plan，给出准确提示字符串；否则为空字符串。提示需严格遵循 pg_hint_plan 语法。",
  "refill_fallback_required": true/false,
  "refill_parameters": {
    "hnsw.iterative_scan": <仅HNSW>,
    "hnsw.max_scan_tuples": <仅HNSW>,
    "ivfflat.iterative_scan": <仅IVFFlat>,
    "ivfflat.max_probes": <仅IVFFlat>
  },
  "notes": "补充说明，如执行边界、回退条件、对统计信息的依赖、适用场景限制等，不超过150字。"
}
```
"""

PROMPTS = [
f"""
请根据下面提供的**向量标量混合查询**、**数据库元数据**和**表统计信息**，生成 **{PROPOSAL_COUNT} 个混合查询执行策略**。
- 每个策略必须明确基于统计信息中的选择性估算
- 策略覆盖关键技术方向（例如：索引选择差异、过滤时机差异、查询重写差异、近似/精确回退、参数调优、部分索引等）。
- 所有策略使用统一 JSON 数组输出，严格遵守给定的 Schema，不要输出任何额外的解释文字。

*PostgreSQL版本为15.5，pgvector版本为0.8.1，pg_hint_plan版本为1.5.1。*

## 目标

向量标量混合查询：同时包含标量过滤条件与向量相似度搜索。

### 生成查询执行策略的目标
为混合查询生成的查询执行策略，应综合满足：
1. 高召回率
2. 高QPS（性能要求）

### 查询语句
```sql
{SQL_CONTENT}
```

## 数据库元数据与统计信息

### my_table 表结构

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

- 向量索引：
  - `my_table_image_vec_idx`：HNSW 索引，使用 L2 距离，参数：m=32, ef_construction=300
  - `my_table_ivf_image_vec_idx`：IVFFlat 索引，使用 L2 距离，参数：lists=1000
- 存储引擎：堆表


### 统计信息 ###

- my_table表总行数： 1000000行
- 向量字段(embddding)维数：128维
- 数据分布：
	```
    {STAT_CONTENT}
	```

## 必备知识

1. **pgvector扩展核心特性**：
   - 支持L2距离、内积、余弦相似度三种向量距离计算
   - 提供HNSW、IVFFlat两种主要向量索引类型及其适用场景
   - 支持精确搜索(Sequential Scan)和近似最近邻搜索(ANN)
   - 支持向量维度最高可达65535维(不同版本略有差异)
   - 支持向量与标量字段的联合查询

2. **混合查询性能瓶颈**：
   - 先过滤后向量搜索：标量过滤后结果集小但无法有效利用向量索引
   - 先向量搜索后过滤：向量索引高效但可能返回大量不满足标量条件的结果
   - 联合索引限制：pgvector不支持向量字段与标量字段的B树联合索引
   - 执行计划偏差：PostgreSQL优化器对向量索引的代价估算不准确

3. **优化技术池**：
   - 索引选择：HNSW vs IVFFlat 及参数调优
   - 过滤时机：pre-filter, post-filter
   - 查询重写：子查询/CTE 强制执行边界、LATERAL JOIN 逐行驱动
   - 近似回退 (Refill Fallback)：逐步扩大 ANN 的 LIMIT 直到满足标量过滤后所需行数
   - 执行计划提示：pg_hint_plan 强制索引扫描、禁止 SeqScan 等
   - 索引提示：使用pg_hint_plan强制优化器选择正确的执行计划
   - 联合查询重写：使用CTE、子查询、LATERAL JOIN等语法优化执行顺序
   - 重排序技术：先ANN搜索返回更多结果，再进行标量过滤和重排序

## 输出内容要求

1. 使用JSON格式，描述查询计划/策略的具体内容
2. 查询计划/策略的具体内容，参考如下：
	- 使用什么向量索引
	- 使用的向量索引的参数
        + HNSW索引:
            * hnsw.ef_search : 搜索时探索的邻居数
        + IVFFlat索引:
            * ivfflat.probes : 搜索时访问的聚类中心数量
	- 使用post-filter还是pre-filter
	- 是否需要SQL改写(SQL Rewrite)
	- 如果需要SQL改写，改写后的SQL内容
	- 基于pg_hint_plant的HINT等
	- 是否需要Refill Fallback（通过迭代扫描来实现）
	- 如果需要迭代扫描，相关的参数值设置，具体包括：
        + HNSW索引:
            * hnsw.iterative_scan : 迭代扫描模式,取值范围: off、strict_order、relaxed_order
            * hnsw.max_scan_tuples : 单次查询最多扫描的向量数量,取值范围: 1~1000000; 默认值:20000
        + IVFFlat索引:
            * ivfflat.iterative_scan : 迭代扫描模式,取值范围: off、strict_order、relaxed_order
            * ivfflat.max_probes : 单次查询最多探测的聚类中心数量,取值范围: 1~nlist; 默认值:65535
3. 查询策略的具体内容中可以包含概要描述
4. 查询计划/策略的具体内容中可以包含其他你认为必要的内容
5. 一般情况下,仅在使用Pre-Filter的情况下，需要使用到HINT或SQL改写。即：通过HINT或SQL改写(或者两者结合)来实现Pre-Filter
6. 不同的参数取值组合，被视为不同的查询计划/策略。你可以通过不同的参数取值组合来生成更多的可能更有的查询策略。

**输出内容模板**
{TEMPLATE_CONTENT}
注：仅输出JSON数组，不附带任何说明。

## 遵循思路和原则
为**向量标量混合查询**生成查询执行策略，请遵循如下思路和原则：
1. 仅在使用pre-filter方式时，**可能**需要做SQL Rewrite
2. 在使用post-filter方式时，可以在HINT中指定使用的索引

**输出内容中参数取值的参考：（请遵循这里的参考）**
- hnsw.ef_search: 取值范围：1 ~ 1000（整数），且必须大于查询的 LIMIT 值。影响：
    * 值越大：搜索遍历的候选点越多，召回率越高，但查询延迟线性上升。
    * 值越小：查询速度越快（QPS越高），但漏检最近邻的概率升高。
- ivfflat.probes: 默认值：1，取值范围：1 ~ lists总数。影响：
    * 值越大：扫描的桶越多，召回率越高，但查询耗时近似线性上升。
    * 值越小：扫描的桶越少，召回率越低，但查询耗时近似线性下降。

""",
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