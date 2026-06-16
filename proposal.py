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
DEFAULT_QUERY_ID = '101'
parser = argparse.ArgumentParser(description="Optimizer program")
parser.add_argument("--query-id", type=str, default=str(DEFAULT_QUERY_ID),
                    help=f"Query ID (default: {DEFAULT_QUERY_ID})")
args = parser.parse_args()

QUERY_ID = args.query_id

# Define file paths with QUERY_ID
SQL_FILE = f"/home/liujianzhong/spj_queries/query{QUERY_ID}_spj.sql"
EXPLAIN_FILE = f"/home/liujianzhong/spj_queries/explain/explain_query{QUERY_ID}_spj.json"
SCHEMA_FILE = "/home/liujianzhong/spj_queries/schema.md"
MANUAL_FILE = "/home/liujianzhong/spj_queries/optimizer_manual.md"
HINT_SPEC_FILE = "/home/liujianzhong/spj_queries/hint_spec.md"
OUT_HINT_FILE = f"/home/liujianzhong/proposal-source/output/out_{QUERY_ID}.json"

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

SQL_CONTENT = read_file(SQL_FILE)
EXPLAIN_CONTENT = read_file(EXPLAIN_FILE)
SCHEMA_CONTENT = read_file(SCHEMA_FILE)
MANUAL_CONTENT = read_file(MANUAL_FILE)
HINT_SPEC_CONTENT = read_file(HINT_SPEC_FILE)

# System prompt
SYSTEM_PROMPT = """
你是一位**PostgreSQL优化器内核专家**和**pgvector向量数据库资深架构师**，精通：
- pgvector扩展的内部实现机制（IVFFlat/HNSW索引构建、查询执行流程、代价估算模型）
- PostgreSQL查询优化器的行为特性与局限性
- 向量标量混合查询的各类性能瓶颈与解决方案
- pg_hint_plan扩展的精确使用方法
- 大规模向量数据集的查询性能调优技术
"""

PROMPTS = [
"""
请根据下面提供的**向量标量混合查询**、**数据库元数据**和**表统计信息**，生成 **5 个混合查询执行策略**。
- 每个策略必须明确基于统计信息中的选择性估算
- 策略覆盖关键技术方向（例如：索引选择差异、过滤时机差异、查询重写差异、近似/精确回退、参数调优、部分索引等）。
- 所有策略使用统一 JSON 数组输出，严格遵守给定的 Schema，不要输出任何额外的解释文字。

*PostgreSQL版本为15.5，pgvector版本为0.8.1，pg_hint_plan版本为1.5.1。*

## 目标混合查询

向量标量混合查询：同时包含标量过滤条件与向量相似度搜索。

### 查询模板语句
```sql
SELECT id FROM sift1m
  WHERE attr >= 3 AND attr <= 6
  ORDER BY embedding <-> %s
  LIMIT 100
```

## 数据库元数据与统计信息

### sift1m 表结构

```
  Column   |       Type       | Collation | Nullable | Default | Storage  | Compression | Stats target | Description
-----------+------------------+-----------+----------+---------+----------+-------------+--------------+-------------
 id        | integer          |           | not null |         | plain    |             |              |
 embedding | vector(128)      |           | not null |         | external |             |              |
 attr      | double precision |           |          |         | plain    |             |              |
Indexes:
    "sift1m_pkey" PRIMARY KEY, btree (id)
    "sift1m_hnsw_idx" hnsw (embedding vector_l2_ops) WITH (m='16', ef_construction='64')
    "sift1m_ivfflat_idx" ivfflat (embedding) WITH (lists='1000')
Access method: heap
```

- 向量索引：
  - `sift1m_hnsw_idx`：HNSW 索引，使用 L2 距离
  - `sift1m_ivfflat_idx`：IVFFlat 索引，未显式指定距离函数（默认与操作符对齐）
- 存储引擎：堆表


### 统计信息 ###

- sift1m表总行数： 1000000行
- 向量字段(embddding)维数：128维
- attr字段数据分布：
	```
	sift=# select attr, count(*) from sift1m group by attr;
	 attr | count
	------+-------
		0 | 90905
		1 | 91125
		2 | 90721
		3 | 90253
		4 | 91408
		5 | 91023
		6 | 90896
		7 | 90676
		8 | 90620 
		9 | 91179
	   10 | 91194
	```

## 必备知识

1. **pgvector扩展核心特性**：
   - 支持L2距离、内积、余弦相似度三种向量距离计算
   - 提供IVFFlat、HNSW两种主要向量索引类型及其适用场景
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
   - 过滤时机：pre-filter, post-filter, 两阶段混合
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
	- 使用post-filter还是pre-filter
	- 是否需要SQL改写(SQL Rewrite)
	- 如果需要SQL改写，改写后的SQL内容
	- 基于pg_hint_plant的HINT等
	- 是否需要Refill Fallback（通过迭代扫描来实现）
	- 如果需要迭代扫描，相关的参数值设置
3. 查询策略的具体内容中可以包含概要描述
4. 查询计划/策略的具体内容中可以包含其他你认为必要的内容

**输出内容模板**
```json
{
  "strategy_id": <1-5>,
  "name": "策略简要名称",
  "description": "结合本数据分布（选择性约36.4%）说明该策略的设计思想、优点及潜在代价，务必引用选择性数值。",
  "vector_index_used": "索引名，如 sift1m_hnsw_idx 或 sift1m_ivfflat_idx 或 null（表示不使用，需精确搜索）",
  "index_parameters": {
    "type": "HNSW 或 IVFFlat 或 none",
    "m": <仅HNSW>,
    "ef_construction": <仅HNSW>,
    "ef_search": <HNSW搜索时动态参数，若有>,
    "lists": <仅IVFFlat>,
    "probes": <IVFFlat探测数，若设定>,
    "distance_type": "L2 / inner_product / cosine"
  },
  "filter_strategy": "pre-filter / post-filter / hybrid(说明) / two-phase",
  "sql_rewrite_required": true/false,
  "rewritten_sql": "若需要改写，提供完整 SQL；若不需要，设为空字符串",
  "hint": "若使用 pg_hint_plan，给出准确提示字符串；否则为空字符串。提示需严格遵循 pg_hint_plan 语法。",
  "refill_fallback_required": true/false,
  "refill_parameters": {
    "initial_limit": <初始 LIMIT，若使用 refill>,
    "final_limit": 100,
    "refill_multiplier": <每次扩展的倍数，如 2.0>,
    "max_iterations": <最大迭代次数>,
    "min_result_threshold": <满足标量过滤的最少行数阈值，达到即停止>
  },
  "notes": "补充说明，如执行边界、回退条件、对统计信息的依赖、适用场景限制等，不超过150字。"
}
```
注：仅输出JSON数组，不附带任何说明。
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
    print(f"[INFO] Query ID: {QUERY_ID}")
    print(f"[INFO] SQL file: {SQL_FILE}")
    print(f"[INFO] Explain file: {EXPLAIN_FILE}")

    # print(f"\n[INFO] System prompt: {SYSTEM_PROMPT}")
    # print(f"=" * 60)

    # Initialize messages with system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Process each prompt
    for i, prompt in enumerate(PROMPTS):
        print(f"\n[INFO] Prompt#{i+1} prompt length: {len(prompt)}: (SQL content length: {len(SQL_CONTENT)}, Explain content length: {len(EXPLAIN_CONTENT)})")

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
                with open(OUT_HINT_FILE, 'w', encoding='utf-8') as f:
                    f.write(assistant_message)
                print(f"[INFO] Output written to: {OUT_HINT_FILE}")
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