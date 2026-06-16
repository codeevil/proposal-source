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
You are a **PostgreSQL Optimizer Kernel Expert** and **Senior pgvector Vector Database Architect**, proficient in:
- Internal implementation mechanisms of the pgvector extension (IVFFlat/HNSW index construction, query execution flow, cost estimation model)
- Behavioral characteristics and limitations of the PostgreSQL query optimizer
- Various performance bottlenecks and solutions for vector-scalar hybrid queries
- Precise usage methods of the pg_hint_plan extension
- Query performance tuning techniques for large-scale vector datasets
"""

PROMPTS = [
"""
Generate **5 hybrid query execution strategies** based on the **vector-scalar hybrid query**, **database metadata**, and **table statistics** provided below.
- Each strategy must be explicitly based on selectivity estimation from the statistics
- Strategies should cover key technical directions (e.g., index selection differences, filtering timing differences, query rewriting differences, approximate/exact fallback, parameter tuning, partial indexes, etc.)
- All strategies must be output as a unified JSON array, strictly adhering to the given Schema. Do not output any additional explanatory text.

*PostgreSQL version 15.5, pgvector version 0.8.1, pg_hint_plan version 1.5.1.*

## Target Hybrid Query

Vector-scalar hybrid query: A query that contains both scalar filter conditions and vector similarity search.

### Query Template Statement
```sql
SELECT id FROM sift1m
  WHERE attr >= 3 AND attr <= 6
  ORDER BY embedding <-> %s
  LIMIT 100
```

## Database Metadata and Statistics

### sift1m Table Structure

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

- Vector indexes:
  - `sift1m_hnsw_idx`: HNSW index using L2 distance
  - `sift1m_ivfflat_idx`: IVFFlat index with no explicitly specified distance function (defaults to align with the operator)
- Storage engine: Heap table


### Statistics

- Total rows in sift1m table: 1,000,000 rows
- Dimension of vector field (embedding): 128 dimensions
- Data distribution of attr field:
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

## Prerequisite Knowledge

1. **Core Features of pgvector Extension**:
   - Supports three vector distance calculations: L2 distance, inner product, and cosine similarity
   - Provides two main vector index types: IVFFlat and HNSW, along with their applicable scenarios
   - Supports exact search (Sequential Scan) and approximate nearest neighbor search (ANN)
   - Supports vector dimensions up to 65535 (varies slightly between versions)
   - Supports joint queries on vector and scalar fields

2. **Hybrid Query Performance Bottlenecks**:
   - Filter first then vector search: Small result set after scalar filtering but cannot effectively utilize vector indexes
   - Vector search first then filter: Efficient vector indexing but may return a large number of results that do not meet scalar conditions
   - Joint index limitation: pgvector does not support B-tree joint indexes on vector and scalar fields
   - Execution plan deviation: PostgreSQL optimizer has inaccurate cost estimation for vector indexes

3. **Optimization Technique Pool**:
   - Index selection: HNSW vs IVFFlat and parameter tuning
   - Filtering timing: pre-filter, post-filter, two-phase hybrid
   - Query rewriting: Subquery/CTE to enforce execution boundaries, LATERAL JOIN for row-by-row driving
   - Approximate fallback (Refill Fallback): Gradually expand the LIMIT of ANN until the required number of rows after scalar filtering is met
   - Execution plan hints: pg_hint_plan to force index scans, disable SeqScan, etc.
   - Index hints: Use pg_hint_plan to force the optimizer to select the correct execution plan
   - Joint query rewriting: Use CTE, subquery, LATERAL JOIN and other syntax to optimize execution order
   - Reordering technique: First perform ANN search to return more results, then apply scalar filtering and reordering

## Output Content Requirements

1. Use JSON format to describe the specific content of the query plan/strategy
2. The specific content of the query plan/strategy should refer to the following:
	- Which vector index to use
	- Parameters of the used vector index
	- Whether to use post-filter or pre-filter
	- Whether SQL rewrite is required
	- If SQL rewrite is required, the rewritten SQL content
	- Hints based on pg_hint_plan, etc.
	- Whether Refill Fallback (implemented via iterative scanning) is required
	- If iterative scanning is required, the relevant parameter value settings
3. The specific content of the query strategy may include a summary description
4. The specific content of the query plan/strategy may include other content you deem necessary

**Output Content Template**
```json
{
  "strategy_id": <1-5>,
  "name": "Brief strategy name",
  "description": "Explain the design idea, advantages and potential costs of this strategy in combination with this data distribution (selectivity approximately 36.4%). Be sure to cite the selectivity value.",
  "vector_index_used": "Index name, e.g., sift1m_hnsw_idx or sift1m_ivfflat_idx or null (indicates not used, exact search required)",
  "index_parameters": {
    "type": "HNSW or IVFFlat or none",
    "m": <HNSW only>,
    "ef_construction": <HNSW only>,
    "ef_search": <Dynamic parameter during HNSW search, if any>,
    "lists": <IVFFlat only>,
    "probes": <Number of IVFFlat probes, if set>,
    "distance_type": "L2 / inner_product / cosine"
  },
  "filter_strategy": "pre-filter / post-filter / hybrid(with explanation) / two-phase",
  "sql_rewrite_required": true/false,
  "rewritten_sql": "Provide complete SQL if rewrite is required; set to empty string if not needed",
  "hint": "Provide accurate hint string if using pg_hint_plan; otherwise empty string. Hints must strictly follow pg_hint_plan syntax.",
  "refill_fallback_required": true/false,
  "refill_parameters": {
    "initial_limit": <Initial LIMIT if using refill>,
    "final_limit": 100,
    "refill_multiplier": <Multiplier for each expansion, e.g., 2.0>,
    "max_iterations": <Maximum number of iterations>,
    "min_result_threshold": <Minimum row count threshold for scalar filtering, stop when reached>
  },
  "notes": "Supplementary notes, such as execution boundaries, fallback conditions, dependency on statistics, applicable scenario limitations, etc. No more than 150 words."
}
```
Note: Only output the JSON array without any additional explanations.
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