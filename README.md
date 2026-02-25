# LLM Parallelism Demo

Demo notebook from Agentic Coding Fitness Week 3 session.

Shows how to run a 72B parameter model (Qwen2.5 72B Instruct) across 2x A100 80GB GPUs when it doesn't fit on a single one.

## What's inside

- `demo_model_parallelism.ipynb` — step-by-step notebook comparing different parallelism strategies:
  - Naive tensor parallelism (manual layer splitting)
  - `device_map="auto"` (HuggingFace's automatic approach)
  - vLLM tensor parallelism (production-grade)
  - Benchmarks for all three

## Requirements

- 2x NVIDIA A100 80GB (or similar multi-GPU setup)
- HuggingFace account with access to Qwen2.5-72B-Instruct
- PyTorch, transformers, vllm
