# NVIDIA NIMs Integration Guide

A quick-start reference for integrating [NVIDIA NIM](https://build.nvidia.com) models into your Python applications using the OpenAI-compatible API.

## What Are NVIDIA NIMs?

NVIDIA NIMs (NVIDIA Inference Microservices) expose large language models through an OpenAI-compatible endpoint at:

```
https://integrate.api.nvidia.com/v1
```

Any model listed on [build.nvidia.com](https://build.nvidia.com) can be accessed by specifying its model ID.

## Getting Started

### 1. Get an API Key

Register at [build.nvidia.com](https://build.nvidia.com/) and generate an API key.

### 2. Store the Key Securely

Create a `.env` file in your project root:

```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> **Never** commit your API key. Make sure `.gitignore` includes `.env`.

### 3. Install Dependencies

**OpenAI SDK approach:**

```bash
pip install openai python-dotenv
```

**LangChain approach:**

```bash
pip install langchain-nvidia-ai-endpoints python-dotenv
```

## Usage Examples

### OpenAI SDK

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

### LangChain

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

## Common Model IDs

| Model | Use Case |
|-------|----------|
| `nvidia/nemotron-3-ultra-550b-a55b` | Reasoning |
| `meta/llama-3.3-70b-instruct` | Instruction-following |
| `nvidia/llama-3.1-nemotron-ultra-253b-v1` | Large reasoning |

Browse the full catalog at [build.nvidia.com](https://build.nvidia.com).

## Integration Notes

- The API is **OpenAI-compatible** — use the standard OpenAI SDK with a custom `base_url`.
- Reasoning models support `reasoning_budget` and `enable_thinking`.
- Streaming is recommended for long-running inference.
- For non-reasoning models, omit `extra_body` / `reasoning_budget` / `chat_template_kwargs`.

## Security Checklist

- [ ] `.env` is listed in `.gitignore`
- [ ] No API key literals in source files
- [ ] Environment variable loading is in place
- [ ] API key is never logged or printed

## License

This project is provided as-is for reference purposes.
