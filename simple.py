"""
Simple LLM interaction demo.
Initialize, send prompt, receive and print LLM response.
"""

import os

import httpx
from dotenv import load_dotenv
from litellm import OpenAI

# Load environment variables
load_dotenv()

# Constants
BASE_URL = "http://localhost:8004/v1"  # Local vLLM server
MODEL_NAME = "/ssd_data/models/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_TIMEOUT = 60


def main():
    """Initialize LLM client and interact with it."""

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return

    # Initialize client with custom http client that disables proxy
    # This avoids SOCKS proxy issues with localhost
    http_client = httpx.Client(
        trust_env=False,  # Disable proxy from environment
        timeout=DEFAULT_TIMEOUT,
    )

    client = OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        http_client=http_client,
    )

    print(f"[INFO] Client initialized successfully")
    print(f"[INFO] Base URL: {BASE_URL}")
    print(f"[INFO] Model: {MODEL_NAME}")

    # Send prompt
    prompt_text = "中国的首都是那个城市"

    print(f"\n[INFO] Sending prompt: {prompt_text}")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            timeout=DEFAULT_TIMEOUT,
        )

        # Extract response content
        result = response.choices[0].message.content

        print(f"\n[INFO] Received response:")
        print("-" * 50)
        print(result)
        print("-" * 50)

        # Print usage stats if available
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            print(f"\n[INFO] Token usage:")
            print(f"  Prompt tokens: {usage.prompt_tokens}")
            print(f"  Completion tokens: {usage.completion_tokens}")
            print(f"  Total tokens: {usage.total_tokens}")

    except Exception as e:
        print(f"[ERROR] Failed to get response: {e}")


if __name__ == "__main__":
    main()