# Awesome Local AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of resources for running AI locally on consumer hardware — LLMs, image generation, and AI agents without cloud dependencies.

Running AI locally means privacy, no subscriptions, and full control. This list covers the tools, guides, and communities that make it practical.

## Contents

- [Getting Started](#getting-started)
- [Hardware Guides](#hardware-guides)
- [Inference Engines](#inference-engines)
- [User Interfaces](#user-interfaces)
- [Models](#models)
  - [Language Models](#language-models)
  - [Image Generation Models](#image-generation-models)
- [Image Generation](#image-generation)
- [AI Agents](#ai-agents)
- [Advanced Topics](#advanced-topics)
- [Communities](#communities)
- [Contributing](#contributing)

---

## Getting Started

New to local AI? Start here.

- [Run Your First Local LLM](https://insiderllm.com/guides/run-first-local-llm/) - Zero to chatting in 10 minutes with Ollama
- [Ollama Quickstart](https://github.com/ollama/ollama#quickstart) - Official getting started guide
- [LM Studio Download](https://lmstudio.ai/) - Visual interface, no command line needed
- [LocalLLaMA Wiki](https://www.reddit.com/r/LocalLLaMA/wiki/index/) - Community-maintained knowledge base
- [What is Quantization?](https://insiderllm.com/guides/llm-quantization-explained/) - Understanding Q4, Q5, Q8 and why they matter
- [Model Formats Explained](https://insiderllm.com/guides/model-formats-explained-gguf-gptq-awq-exl2/) - GGUF vs GPTQ vs AWQ vs EXL2

## Hardware Guides

Figuring out what hardware you need (or what to do with what you have).

### VRAM Requirements

- [How Much VRAM Do You Need?](https://insiderllm.com/guides/vram-requirements-local-llms/) - Model size to VRAM mapping
- [What Can You Run on 4GB VRAM](https://insiderllm.com/guides/what-can-you-run-4gb-vram/) - GTX 1650, 1050 Ti users
- [What Can You Run on 8GB VRAM](https://insiderllm.com/guides/what-can-you-run-8gb-vram/) - RTX 3060 Ti, 4060 class
- [What Can You Run on 12GB VRAM](https://insiderllm.com/guides/what-can-you-run-12gb-vram/) - RTX 3060 12GB, 4070
- [What Can You Run on 16GB VRAM](https://insiderllm.com/guides/what-can-you-run-16gb-vram/) - RTX 4060 Ti 16GB, 4080
- [What Can You Run on 24GB VRAM](https://insiderllm.com/guides/what-can-you-run-24gb-vram/) - RTX 3090, 4090
- [GPU Benchmarks for LLM Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) - Community benchmark database

### Buying Guides

- [GPU Buying Guide for Local AI](https://insiderllm.com/guides/gpu-buying-guide-local-ai/) - Price/performance analysis
- [Used RTX 3090 Buying Guide](https://insiderllm.com/guides/used-rtx-3090-buying-guide/) - The value king for 24GB VRAM
- [Used GPU Buying Guide](https://insiderllm.com/guides/used-gpu-buying-guide-local-ai/) - eBay, Marketplace, what to look for
- [Budget AI PC Under $500](https://insiderllm.com/guides/budget-local-ai-pc-500/) - Used Optiplex + GPU strategy
- [Tom's Hardware GPU Hierarchy](https://www.tomshardware.com/reviews/gpu-hierarchy,4388.html) - General GPU rankings

### Platform Comparisons

- [Mac vs PC for Local AI](https://insiderllm.com/guides/mac-vs-pc-local-ai/) - Unified memory vs discrete GPU
- [Running LLMs on Mac M-Series](https://insiderllm.com/guides/running-llms-mac-m-series/) - M1/M2/M3/M4 complete guide
- [AMD vs NVIDIA for Local AI](https://insiderllm.com/guides/amd-vs-nvidia-local-ai-rocm/) - ROCm reality check
- [Laptop vs Desktop for Local AI](https://insiderllm.com/guides/laptop-vs-desktop-local-ai/) - Portability tradeoffs
- [CPU-Only LLMs](https://insiderllm.com/guides/cpu-only-llms-what-actually-works/) - Running models without a GPU

## Inference Engines

The software that actually runs the models.

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The foundational CPU/GPU inference engine, supports GGUF
- [Ollama](https://github.com/ollama/ollama) - User-friendly wrapper around llama.cpp with model management
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput serving for production deployments
- [ExLlamaV2](https://github.com/turboderp/exllamav2) - Fastest single-user NVIDIA inference, EXL2 format
- [MLX](https://github.com/ml-explore/mlx) - Apple's framework optimized for M-series Macs
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings for llama.cpp
- [candle](https://github.com/huggingface/candle) - Rust ML framework with LLM support
- [llama.cpp vs Ollama vs vLLM](https://insiderllm.com/guides/llamacpp-vs-ollama-vs-vllm/) - When to use each

## User Interfaces

GUIs and web interfaces for interacting with local models.

### Desktop Applications

- [LM Studio](https://lmstudio.ai/) - Polished desktop app with built-in model browser
- [GPT4All](https://gpt4all.io/) - Simple desktop app, good for beginners
- [Jan](https://jan.ai/) - Open-source ChatGPT alternative with local models
- [Msty](https://msty.app/) - Mac-native LLM interface

### Web Interfaces

- [Open WebUI](https://github.com/open-webui/open-webui) - ChatGPT-style interface, works with Ollama
- [text-generation-webui](https://github.com/oobabooga/text-generation-webui) - The Swiss Army knife of local AI UIs
- [SillyTavern](https://github.com/SillyTavern/SillyTavern) - Frontend for chat/roleplay with local models
- [LibreChat](https://github.com/danny-avila/LibreChat) - Multi-provider chat interface
- [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) - RAG-focused interface with document upload

### Guides

- [Ollama vs LM Studio](https://insiderllm.com/guides/ollama-vs-lm-studio/) - Comparison of the two most popular tools
- [Open WebUI Setup Guide](https://insiderllm.com/guides/open-webui-setup-guide/) - Installation and configuration
- [Text Generation WebUI Guide](https://insiderllm.com/guides/text-generation-webui-oobabooga-guide/) - Power user setup
- [LM Studio Tips & Tricks](https://insiderllm.com/guides/lm-studio-tips-and-tricks/) - Hidden features

## Models

### Language Models

#### Model Libraries

- [Ollama Library](https://ollama.com/library) - Curated models ready to run with `ollama pull`
- [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - Benchmark comparisons
- [TheBloke on HuggingFace](https://huggingface.co/TheBloke) - Quantized models in every format
- [bartowski on HuggingFace](https://huggingface.co/bartowski) - High-quality GGUF quantizations

#### Model Families

- [Llama 3](https://github.com/meta-llama/llama3) - Meta's flagship open model (1B to 405B)
- [Qwen 2.5](https://github.com/QwenLM/Qwen2.5) - Alibaba's strong multilingual models
- [Mistral](https://mistral.ai/technology/) - Efficient 7B-12B models
- [DeepSeek](https://github.com/deepseek-ai) - Strong reasoning and coding models
- [Phi](https://huggingface.co/microsoft/phi-4) - Microsoft's small-but-capable models
- [Gemma](https://ai.google.dev/gemma) - Google's open models

#### Model Guides

- [Llama 3 Guide](https://insiderllm.com/guides/llama-3-guide-every-size/) - Every size from 1B to 405B
- [Qwen Models Guide](https://insiderllm.com/guides/qwen-models-guide/) - Qwen 3, Qwen 2.5 Coder, Qwen-VL
- [DeepSeek Models Guide](https://insiderllm.com/guides/deepseek-models-guide/) - R1, V3, Coder
- [Mistral & Mixtral Guide](https://insiderllm.com/guides/mistral-mixtral-guide/) - 7B, Nemo, Mixtral MoE
- [Best Models Under 3B](https://insiderllm.com/guides/best-models-under-3b-parameters/) - For edge devices

#### By Use Case

- [Best Models for Coding](https://insiderllm.com/guides/best-local-coding-models-2026/) - Code completion and generation
- [Best Models for Math & Reasoning](https://insiderllm.com/guides/best-local-llms-math-reasoning/) - DeepSeek R1, Qwen thinking
- [Best Models for Writing](https://insiderllm.com/guides/best-local-llms-writing-creative-work/) - Creative and content work
- [Best Models for Chat](https://insiderllm.com/guides/best-local-llms-chat-conversation/) - Conversational assistants

### Image Generation Models

- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) - The original open image model
- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) - Higher resolution, better quality
- [Flux](https://github.com/black-forest-labs/flux) - Best open image model for prompt following
- [SD 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) - Stability's latest
- [Civitai](https://civitai.com/) - Community checkpoints, LoRAs, and embeddings

## Image Generation

Interfaces and tools for running image generation locally.

### Interfaces

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Node-based workflow, supports everything
- [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Classic web UI, huge extension ecosystem
- [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) - A1111 fork with better performance
- [Fooocus](https://github.com/lllyasviel/Fooocus) - Simplified UI, Midjourney-like experience
- [SD.Next](https://github.com/vladmandic/sdnext) - A1111 fork with AMD/Intel support
- [InvokeAI](https://github.com/invoke-ai/InvokeAI) - Professional-grade creative tool

### Guides

- [Stable Diffusion Locally](https://insiderllm.com/guides/stable-diffusion-locally-getting-started/) - Complete getting started guide
- [Flux Locally](https://insiderllm.com/guides/flux-locally-complete-guide/) - Running Flux on consumer hardware
- [ComfyUI vs A1111 vs Fooocus](https://insiderllm.com/guides/comfyui-vs-automatic1111-vs-fooocus/) - Which UI to choose

### Extensions & Tools

- [ControlNet](https://github.com/lllyasviel/ControlNet) - Precise image control
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) - Style and subject transfer
- [AnimateDiff](https://github.com/guoyww/AnimateDiff) - Video generation from SD
- [Upscayl](https://github.com/upscayl/upscayl) - AI image upscaling

## AI Agents

Running autonomous AI agents locally.

### Frameworks

- [OpenClaw](https://github.com/open-claw/open-claw) - Open-source AI agent framework
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) - Autonomous GPT-4 agent
- [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for LLM apps
- [Haystack](https://github.com/deepset-ai/haystack) - NLP framework for agents

### Guides

- [OpenClaw Setup Guide](https://insiderllm.com/guides/openclaw-setup-guide/) - Local AI agent installation
- [OpenClaw Security Guide](https://insiderllm.com/guides/openclaw-security-guide/) - Hardening autonomous agents
- [Best Models for OpenClaw](https://insiderllm.com/guides/best-local-models-openclaw/) - Which Ollama models work for agents
- [OpenClaw vs Commercial Agents](https://insiderllm.com/guides/openclaw-vs-commercial-ai-agents/) - Local vs Lindy, Rabbit, etc.

## Advanced Topics

Going deeper into local AI.

### RAG & Document Search

- [Local RAG Guide](https://insiderllm.com/guides/local-rag-search-documents-private-ai/) - Search your documents with private AI
- [Chroma](https://github.com/chroma-core/chroma) - Open-source embedding database
- [Qdrant](https://github.com/qdrant/qdrant) - Vector similarity search engine
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook's similarity search library

### Fine-Tuning

- [Fine-Tuning on Consumer Hardware](https://insiderllm.com/guides/fine-tuning-local-lora-qlora/) - LoRA and QLoRA guide
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - Streamlined fine-tuning tool
- [Unsloth](https://github.com/unslothai/unsloth) - 2x faster fine-tuning
- [PEFT](https://github.com/huggingface/peft) - HuggingFace parameter-efficient fine-tuning

### Voice & Multimodal

- [Voice Chat with Local LLMs](https://insiderllm.com/guides/voice-chat-local-llms-whisper-tts/) - Whisper + TTS setup
- [Whisper](https://github.com/openai/whisper) - OpenAI's speech recognition
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Fast Whisper inference
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Text-to-speech synthesis
- [Piper](https://github.com/rhasspy/piper) - Fast local TTS

### Coding Assistants

- [Continue](https://github.com/continuedev/continue) - VS Code/JetBrains AI assistant
- [Tabby](https://github.com/TabbyML/tabby) - Self-hosted code completion
- [Aider](https://github.com/paul-gauthier/aider) - AI pair programming in terminal
- [Codeium](https://codeium.com/) - Free AI code completion (cloud + local options)

### Troubleshooting

- [Local AI Troubleshooting Guide](https://insiderllm.com/guides/local-ai-troubleshooting-guide/) - Fix common problems
- [Ollama Troubleshooting Guide](https://insiderllm.com/guides/ollama-troubleshooting-guide/) - Ollama-specific fixes
- [Context Length Explained](https://insiderllm.com/guides/context-length-explained/) - Why long conversations crash

## Communities

Where to get help and stay updated.

### Reddit

- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) - The main hub for local AI discussion
- [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/) - Image generation community
- [r/Ollama](https://www.reddit.com/r/ollama/) - Ollama-specific discussion
- [r/Oobabooga](https://www.reddit.com/r/Oobabooga/) - text-generation-webui community

### Discord

- [LocalLLaMA Discord](https://discord.gg/localllama) - Active chat community
- [Ollama Discord](https://discord.gg/ollama) - Official Ollama server
- [ComfyUI Discord](https://discord.gg/comfyui) - ComfyUI community
- [Stable Diffusion Discord](https://discord.gg/stablediffusion) - Image generation community

### Other

- [HuggingFace Discussions](https://huggingface.co/discussions) - Model-specific discussions
- [llama.cpp GitHub Discussions](https://github.com/ggerganov/llama.cpp/discussions) - Technical questions
- [LocalAI Slack](https://localai.io/) - LocalAI community

## Contributing

Contributions welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

- Add resources that are genuinely useful for running AI locally
- Include a brief description explaining why the resource is valuable
- Verify links are working and resources are actively maintained
- Keep the list organized and avoid duplicates

---

## License

[![CC0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the contributors have waived all copyright and related rights to this work.
