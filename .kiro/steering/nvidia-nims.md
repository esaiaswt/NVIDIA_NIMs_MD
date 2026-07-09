---
inclusion: fileMatch
fileMatchPattern: "*nim*,*nvidia*,*NIM*,*NVIDIA*"
---

# NVIDIA NIMs Integration Guide

When implementing code that accesses NVIDIA NIM models, follow these patterns and guidelines.

## Overview

NVIDIA NIMs are available at https://build.nvidia.com. They expose an OpenAI-compatible API at `https://integrate.api.nvidia.com/v1`. Any model listed on build.nvidia.com can be used by specifying its model identifier (e.g., `nvidia/nemotron-3-ultra-550b-a55b`).

## API Key Management

- **NEVER** hardcode the API key in source code
- **ALWAYS** store the key in a `.env` file as `NVIDIA_API_KEY`
- **ALWAYS** ensure `.gitignore` includes `.env` — never commit secrets to any repository
- Users obtain their key by registering at https://build.nvidia.com/

Example `.env`:
```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Python — OpenAI SDK

Dependencies: `openai`, `python-dotenv`

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

completion = client.chat.completions.create(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_budget": 16384,
    },
    stream=True,
)

for chunk in completion:
    if not chunk.choices:
        continue
    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
    if reasoning:
        print(reasoning, end="")
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## Python — LangChain

Dependencies: `langchain-nvidia-ai-endpoints`, `python-dotenv`

```python
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

client = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key=os.environ["NVIDIA_API_KEY"],
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    reasoning_budget=16384,
    chat_template_kwargs={"enable_thinking": True},
)

for chunk in client.stream([{"role": "user", "content": "Hello"}]):
    if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
        print(chunk.additional_kwargs["reasoning_content"], end="")
    print(chunk.content, end="")
```

## Key Integration Notes

- The API is **OpenAI-compatible** — use the standard OpenAI SDK with a custom `base_url`
- Replace the `model` parameter with any model ID from https://build.nvidia.com
- Reasoning models support `reasoning_budget` and `chat_template_kwargs.enable_thinking`
- Streaming is recommended for long-running inference calls
- Always load the API key from environment variables using `python-dotenv` or equivalent
- For non-reasoning models, omit `extra_body` / `reasoning_budget` / `chat_template_kwargs`

## Common Model IDs

Browse https://build.nvidia.com for the full catalog. Examples:
- `nvidia/nemotron-3-ultra-550b-a55b` — reasoning model
- `meta/llama-3.3-70b-instruct` — instruction-following
- `nvidia/llama-3.1-nemotron-ultra-253b-v1` — large reasoning model

## Security Checklist

Before committing any NIM integration code:
1. Confirm `.env` is in `.gitignore`
2. Confirm no API key literals exist in source files
3. Confirm environment variable loading is in place
4. Never log or print the API key value
