---
inclusion: fileMatch
fileMatchPattern: "*nim*,*nvidia*,*NIM*,*NVIDIA*"
---

# NVIDIA NIMs Integration Guide

When implementing code that accesses NVIDIA NIM models, follow these patterns and guidelines.

## Overview

NVIDIA NIMs are available at https://build.nvidia.com. They expose an OpenAI-compatible API at `https://integrate.api.nvidia.com/v1`. Any model listed on build.nvidia.com can be used by specifying its model identifier (e.g., `nvidia/nemotron-3-ultra-550b-a55b`).

## Documentation References

- Models overview: https://docs.api.nvidia.com/nim/docs/models
- API reference: https://docs.api.nvidia.com/nim/reference/models-1
- LLM APIs: https://docs.api.nvidia.com/nim/reference/llm-apis
- Retrieval APIs: https://docs.api.nvidia.com/nim/reference/retrieval-apis
- Visual APIs: https://docs.api.nvidia.com/nim/reference/visual-models-apis
- Multimodal APIs: https://docs.api.nvidia.com/nim/reference/multimodal-apis
- Healthcare APIs: https://docs.api.nvidia.com/nim/reference/healthcare-apis

## API Categories & Endpoints

| Category | Endpoint | Description |
|----------|----------|-------------|
| LLM | `POST /v1/chat/completions` | Chat, QA, summarization, code generation |
| Retrieval (Embeddings) | `POST /v1/embeddings` | Text/image embeddings |
| Retrieval (Reranking) | `POST /v1/ranking` | Passage reranking |
| Visual Models | `POST /v1/...` (model-specific) | Image/video generation, object detection |
| Multimodal | `POST /v1/chat/completions` | Vision + text understanding |
| Healthcare | Various model-specific endpoints | Drug discovery, protein structure, genomics |
| Route Optimization | `POST /v1/...` (cuOpt) | Vehicle routing optimization |
| Climate Simulation | `POST /v1/...` (FourCastNet, CorrDiff) | Weather forecasting |

## Chat Completions API Parameters

All LLM and multimodal models use the OpenAI-compatible chat completions endpoint.

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | string | required | — | Model identifier (e.g., `nvidia/nemotron-3-ultra-550b-a55b`) |
| `messages` | array | required | — | Conversation messages (system/user/assistant roles) |
| `temperature` | number | 1 | 0–1 | Sampling randomness |
| `top_p` | number | 0.95 | 0–1 | Nucleus sampling |
| `max_tokens` | integer | 16384 | 1–32768 | Max output tokens |
| `reasoning_effort` | enum | "high" | none/medium/high | Controls reasoning mode (reasoning models only) |
| `reasoning_budget` | integer | 16384 | -1–32768 | Max reasoning tokens (-1 disables budget) |
| `seed` | integer | — | 0–max | Deterministic sampling |
| `stream` | boolean | true | — | Enable SSE streaming |
| `stop` | string/array/null | — | — | Stop sequences |

Notes:
- Do not modify both `temperature` and `top_p` in the same call
- `reasoning_effort` and `reasoning_budget` apply only to reasoning models
- For non-reasoning models, omit reasoning parameters entirely

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

## Model Catalog

### LLM Models

**NVIDIA Nemotron (Reasoning):**
- `nvidia/nemotron-3-ultra-550b-a55b` — flagship reasoning model
- `nvidia/nemotron-3-super-120b-a12b` — large reasoning
- `nvidia/nemotron-3-nano-30b-a3b` — efficient reasoning
- `nvidia/llama-3.1-nemotron-ultra-253b-v1` — large reasoning
- `nvidia/llama-3.3-nemotron-super-49b-v1` — mid-size reasoning
- `nvidia/llama-3.3-nemotron-super-49b-v1.5` — mid-size reasoning (updated)
- `nvidia/llama-3.1-nemotron-nano-8b-v1` — compact reasoning
- `nvidia/nvidia-nemotron-nano-9b-v2` — compact reasoning v2

**NVIDIA Specialized:**
- `nvidia/nemotron-mini-4b-instruct` — lightweight instruction
- `nvidia/usdcode` — USD/3D code generation
- `nvidia/riva-translate-4b-instruct-v1_1` — translation
- `nvidia/gliner-pii` — PII entity extraction
- `nvidia/nemoguard-jailbreak-detect` — jailbreak detection
- `nvidia/nemotron-content-safety-reasoning-4b` — content safety

**Meta Llama:**
- `meta/llama-3.3-70b-instruct` — instruction-following
- `meta/llama-3.1-70b-instruct` — instruction-following
- `meta/llama-3.1-8b-instruct` — compact instruction
- `meta/llama-3.2-1b-instruct` — tiny instruction
- `meta/llama-3.2-3b-instruct` — small instruction
- `meta/llama2-70b` — legacy

**DeepSeek:**
- `deepseek-ai/deepseek-v4-flash` — fast inference
- `deepseek-ai/deepseek-v4-pro` — high quality

**Google:**
- `google/gemma-2-2b-it` — compact
- `google/gemma-7b` — mid-size
- `google/codegemma-7b` — code generation

**Microsoft:**
- `microsoft/phi-4-mini-instruct` — compact instruction
- `microsoft/phi-4-mini-flash-reasoning` — compact reasoning

**MiniMax:**
- `minimaxai/minimax-m2.5` — general purpose
- `minimaxai/minimax-m2.7` — general purpose (updated)

**Mistral:**
- `mistralai/mistral-nemotron` — NVIDIA-tuned Mistral
- `mistralai/mixtral-8x7b-instruct` — MoE instruction
- `mistralai/mixtral-8x22b-instruct` — large MoE instruction

**Moonshot:**
- `moonshotai/kimi-k2-instruct` — instruction-following
- `moonshotai/kimi-k2-thinking` — reasoning

**OpenAI:**
- `openai/gpt-oss-20b` — open-source GPT
- `openai/gpt-oss-120b` — large open-source GPT

**Qwen:**
- `qwen/qwen2.5-coder-32b-instruct` — code generation
- `qwen/qwen3.5-122b-a10b` — general purpose
- `qwen/qwen3-coder-480b-a35b-instruct` — large code generation
- `qwen/qwen3-next-80b-a3b-instruct` — instruction-following
- `qwen/qwen3-next-80b-a3b-thinking` — reasoning
- `qwen/qwq-32b` — reasoning

**Others:**
- `abacusai/dracarys-llama-3.1-70b-instruct`
- `bytedance/seed-oss-36b-instruct`
- `sarvamai/sarvam-m`
- `stepfun-ai/step-3-5-flash`
- `stockmark/stockmark-2-100b-instruct`
- `upstage/solar-10.7b-instruct`
- `z-ai/glm4.7`, `z-ai/glm5.1`, `z-ai/glm-5.2`

### Retrieval Models — Embeddings

- `nvidia/embed-qa-4` — general QA embedding
- `nvidia/nv-embed-v1` — general embedding
- `nvidia/nv-embedqa-e5-v5` — QA-optimized embedding
- `nvidia/nv-embedcode-7b-v1` — code embedding
- `nvidia/llama-nemotron-embed-1b-v2` — efficient embedding
- `nvidia/llama-nemotron-embed-vl-1b-v2` — vision+language embedding
- `nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1` — VLM embedding
- `nvidia/llama-3.2-nemoretriever-300m-embed-v1` — tiny embedding
- `nvidia/llama-3.2-nemoretriever-300m-embed-v2` — tiny embedding v2
- `nvidia/llama-3.2-nv-embedqa-1b-v2` — QA embedding
- `nvidia/nvclip` — CLIP-style text/image embedding
- `baai/bge-m3` — multilingual embedding
- `snowflake/arctic-embed-l` — general embedding

### Retrieval Models — Reranking

- `nvidia/nv-rerankqa-mistral-4b-v3` — QA reranking
- `nvidia/rerank-qa-mistral-4b` — QA reranking (legacy)
- `nvidia/llama-nemotron-rerank-1b-v2` — efficient reranking
- `nvidia/llama-nemotron-rerank-vl-1b-v2` — vision+language reranking
- `nvidia/llama-3.2-nemoretriever-500m-rerank-v2` — compact reranking
- `nvidia/llama-3.2-nv-rerankqa-1b-v1` — QA reranking
- `nvidia/llama-3.2-nv-rerankqa-1b-v2` — QA reranking v2

### Visual Models

**Image Generation:**
- `black-forest-labs/flux.1-dev` — high-quality image generation
- `black-forest-labs/flux.1-schnell` — fast image generation
- `black-forest-labs/flux.2-klein-4b` — compact image generation
- `google/diffusiongemma-26b-a4b-it` — diffusion model
- `stabilityai/stable-diffusion-3-medium` — Stable Diffusion 3
- `stabilityai/stable-diffusion-xl` — SDXL
- `stabilityai/stable-video-diffusion` — video generation

**Vision Understanding:**
- `google/gemma-3-27b-it` — vision+language
- `google/gemma-3n-e2b-it`, `google/gemma-3n-e4b-it` — efficient vision
- `google/gemma-4-31b-it` — latest vision
- `nvidia/nv-dinov2` — image features
- `nvidia/nv-grounding-dino` — object detection
- `nvidia/bevformer` — BEV perception
- `nvidia/cosmos-predict1-7b` — world model
- `hive/ai-generated-image-detection` — AI image detection
- `hive/deepfake-image-detection` — deepfake detection

### Multimodal Models

- `meta/llama-3.2-11b-vision-instruct` — vision+language (11B)
- `meta/llama-3.2-90b-vision-instruct` — vision+language (90B)
- `meta/llama-4-maverick-17b-128e-instruct` — Llama 4 multimodal
- `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` — compact vision+language
- `nvidia/nemotron-nano-12b-v2-vl` — vision+language
- `nvidia/nemotron-3-content-safety` — multimodal content safety
- `google/paligemma` — image understanding
- `moonshotai/kimi-k2.5` — multimodal reasoning
- `qwen/qwen3.5-397b-a17b` — large multimodal
- `black-forest-labs/flux.1-kontext-dev` — image editing

### Healthcare Models

- `deepmind/alphafold2` — protein structure prediction
- `deepmind/alphafold2-multimer` — multi-chain protein structure
- `openfold/openfold2` — protein folding
- `openfold/openfold3` — latest protein folding
- `meta/esmfold` — alignment-free protein structure
- `meta/esm2-650m` — protein embeddings
- `arc/evo2-40b` — DNA sequence generation
- `nvidia/genmol` — molecular generation
- `nvidia/molmim` — molecule generation
- `nvidia/vista3d` — 3D medical imaging
- `mit/diffdock` — molecular docking
- `mit/boltz2` — structure prediction
- `ipd/proteinmpnn` — protein sequence design
- `ipd/rfdiffusion` — protein structure generation
- `colabfold/msa-search` — multiple sequence alignment

### Safety & Guardrails Models

- `nvidia/llama-3.1-nemoguard-8b-content-safety` — content safety
- `nvidia/llama-3.1-nemoguard-8b-topic-control` — topic control
- `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` — safety guard
- `nvidia/nemotron-3.5-content-safety` — content safety (multimodal)
- `nvidia/nemotron-content-safety-reasoning-4b` — reasoning safety
- `nvidia/nemoguard-jailbreak-detect` — jailbreak detection
- `meta/llama-guard-4-12b` — Llama Guard

### Route Optimization

- `nvidia/cuOpt` — vehicle routing optimization

### Climate Simulation

- `nvidia/fourcastnet` — weather forecasting
- `nvidia/corrdiff` — high-resolution weather

## Security Checklist

Before committing any NIM integration code:
1. Confirm `.env` is in `.gitignore`
2. Confirm no API key literals exist in source files
3. Confirm environment variable loading is in place
4. Never log or print the API key value
