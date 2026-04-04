# Awesome Local AI [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of resources for running AI locally on consumer hardware -- LLMs, image generation, and AI agents without cloud dependencies. 230+ guides, tools, and community links.

Running AI locally means privacy, no subscriptions, and full control. This list covers the tools, guides, and communities that make it practical.

*Last updated: 2026-03-06*

## Contents

- [Getting Started](#getting-started)
- [Tools](#tools)
- [Hardware Guides](#hardware-guides)
- [Inference Engines](#inference-engines)
- [User Interfaces](#user-interfaces)
- [Models](#models)
  - [Language Models](#language-models)
  - [Image Generation Models](#image-generation-models)
- [Image Generation](#image-generation)
- [AI Agents](#ai-agents)
- [Advanced Topics](#advanced-topics)
- [Use Cases](#use-cases)
- [Blog](#blog)
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
- [Building a Local AI Assistant](https://insiderllm.com/guides/building-local-ai-assistant/) - Private Jarvis with Ollama, Open WebUI, Whisper, and TTS

## Tools

Interactive tools for planning and optimizing your local AI setup.

- [Local AI Planning Tool](https://insiderllm.com/tools/vram-calculator/) - Interactive VRAM calculator with hardware, model, and task entry points

## Hardware Guides

Figuring out what hardware you need (or what to do with what you have).

### VRAM Requirements

- [How Much VRAM Do You Need?](https://insiderllm.com/guides/vram-requirements-local-llms/) - Model size to VRAM mapping
- [What Can You Run on 4GB VRAM](https://insiderllm.com/guides/what-can-you-run-4gb-vram/) - GTX 1650, 1050 Ti users
- [What Can You Run on 8GB VRAM](https://insiderllm.com/guides/what-can-you-run-8gb-vram/) - RTX 3060 Ti, 4060 class
- [The 8GB VRAM Trap](https://insiderllm.com/guides/8gb-vram-trap-local-ai/) - What "runs on 8GB" actually means after quantization
- [What Can You Run on 12GB VRAM](https://insiderllm.com/guides/what-can-you-run-12gb-vram/) - RTX 3060 12GB, 4070
- [What Can You Run on 16GB VRAM](https://insiderllm.com/guides/what-can-you-run-16gb-vram/) - RTX 4060 Ti 16GB, 4080
- [What Can You Run on 24GB VRAM](https://insiderllm.com/guides/what-can-you-run-24gb-vram/) - RTX 3090, 4090
- [Running 70B Models Locally](https://insiderllm.com/guides/running-70b-models-locally-vram-guide/) - Exact VRAM by quantization for Llama 70B+
- [Mixtral 8x7B & 8x22B VRAM Requirements](https://insiderllm.com/guides/mixtral-8x7b-8x22b-vram-requirements/) - MoE memory mapping
- [Mixtral VRAM Requirements](https://insiderllm.com/guides/mixtral-vram-requirements/) - Every quantization level for 8x7B and 8x22B
- [KV Cache and VRAM](https://insiderllm.com/guides/kv-cache-optimization-guide/) - Why context length eats your VRAM and how to fix it
- [num_ctx VRAM Overflow](https://insiderllm.com/guides/num-ctx-vram-overflow-slow-inference/) - The silent performance killer nobody warns about
- [GPU Benchmarks for LLM Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) - Community benchmark database

### Buying Guides

- [GPU Buying Guide for Local AI](https://insiderllm.com/guides/gpu-buying-guide-local-ai/) - Price/performance analysis
- [Best GPU Under $300](https://insiderllm.com/guides/best-gpu-under-300-local-ai/) - RTX 3060 12GB, RX 7600, Arc B580 compared
- [Best GPU Under $500](https://insiderllm.com/guides/best-gpu-under-500-local-ai/) - RTX 4060 Ti 16GB, used RTX 3080, RX 7700 XT
- [Used RTX 3090 Buying Guide](https://insiderllm.com/guides/used-rtx-3090-buying-guide/) - The value king for 24GB VRAM
- [Used GPU Buying Guide](https://insiderllm.com/guides/used-gpu-buying-guide-local-ai/) - eBay, Marketplace, what to look for
- [Best Used GPUs for Local AI 2026](https://insiderllm.com/guides/best-used-gpus-local-ai-2026/) - Tier rankings, fair prices, what to avoid
- [Used Server GPUs](https://insiderllm.com/guides/used-server-gpus-local-ai/) - Tesla P40, V100, A100 and the eBay goldmine
- [Used Tesla P40](https://insiderllm.com/guides/used-tesla-p40-local-ai/) - 24GB VRAM for $150-200
- [Budget AI PC Under $500](https://insiderllm.com/guides/budget-local-ai-pc-500/) - Used Optiplex + GPU strategy
- [Best Mini PCs for Local AI](https://insiderllm.com/guides/best-mini-pcs-local-ai-2026/) - Under $300 picks with real tok/s
- [RTX 5090 for Local AI](https://insiderllm.com/guides/rtx-5090-local-ai-worth-it/) - Worth the upgrade?
- [RTX 5060 Ti for Local AI](https://insiderllm.com/guides/rtx-5060-ti-16gb-local-ai-options/) - Budget next-gen 16GB GPU
- [RTX 5060 Ti Benchmarks](https://insiderllm.com/guides/rtx-5060-ti-local-ai-benchmarks/) - Real LLM inference numbers
- [RTX 4090 vs Used 3090](https://insiderllm.com/guides/rtx-4090-vs-used-rtx-3090-local-ai/) - Which to buy for AI workloads
- [RTX 3090 vs 4070 Ti Super](https://insiderllm.com/guides/rtx-3090-vs-4070-ti-super-local-llms/) - Mid-range showdown for local LLMs
- [RTX 3060 vs 3060 Ti vs 3070](https://insiderllm.com/guides/rtx-3060-vs-3060ti-vs-3070-local-ai/) - 12GB wins despite being the cheapest
- [Intel Arc B580 for Local LLMs](https://insiderllm.com/guides/intel-arc-b580-local-llm/) - 12GB VRAM at $250, with caveats
- [Intel Arc GPUs for Local AI](https://insiderllm.com/guides/intel-arc-local-ai/) - A770 16GB, IPEX-LLM, SYCL setup
- [GB10 Boxes Compared](https://insiderllm.com/guides/gb10-boxes-compared/) - NVIDIA GB10 hardware options
- [Multi-GPU Setups: Worth It?](https://insiderllm.com/guides/multi-gpu-worth-it/) - When dual GPUs beat one bigger card
- [Multi-GPU Local AI](https://insiderllm.com/guides/multi-gpu-local-ai/) - Run models across multiple GPUs with tensor/pipeline parallelism
- [NVIDIA GPU Prices Rising](https://insiderllm.com/guides/nvidia-gpu-prices-rising-2025/) - GDDR7 shortages and what to do
- [Razer AI Kit Guide](https://insiderllm.com/guides/razer-aikit-guide/) - Razer's dedicated AI hardware
- [Build a Distributed AI Swarm Under $1,100](https://insiderllm.com/guides/build-distributed-ai-swarm-under-1100/) - Three-node cluster bill of materials
- [Tom's Hardware GPU Hierarchy](https://www.tomshardware.com/reviews/gpu-hierarchy,4388.html) - General GPU rankings

### Platform Guides

- [Mac vs PC for Local AI](https://insiderllm.com/guides/mac-vs-pc-local-ai/) - Unified memory vs discrete GPU
- [Running LLMs on Mac M-Series](https://insiderllm.com/guides/running-llms-mac-m-series/) - M1/M2/M3/M4 complete guide
- [M4 Max and Ultra for LLMs](https://insiderllm.com/guides/m4-max-ultra-local-llms-apple-silicon/) - Apple Silicon performance update
- [Apple M5 Pro and Max](https://insiderllm.com/guides/apple-m5-pro-max-local-ai/) - What 4x faster LLM processing means
- [Apple Neural Engine for LLMs](https://insiderllm.com/guides/apple-neural-engine-llm-inference/) - What the ANE can and can't do
- [Mac Mini M4 for Local AI](https://insiderllm.com/guides/mac-mini-m4-local-ai/) - Best value Mac setup for AI
- [Mac Studio for Local AI](https://insiderllm.com/guides/mac-studio-local-ai-workstation/) - M4 Max vs M3 Ultra pricing analysis
- [Mac Studio M4 Every Config](https://insiderllm.com/guides/mac-studio-m4-local-ai/) - M4 Max 128GB vs M3 Ultra 512GB ranked
- [8GB Apple Silicon Local AI](https://insiderllm.com/guides/8gb-apple-silicon-local-ai/) - What actually runs on a budget Mac
- [Best Local LLMs for Mac 2026](https://insiderllm.com/guides/best-local-llms-mac-2026/) - Top picks for M-series
- [AMD vs NVIDIA for Local AI](https://insiderllm.com/guides/amd-vs-nvidia-local-ai-rocm/) - ROCm reality check
- [ROCm vs CUDA in 2026](https://insiderllm.com/guides/rocm-vs-cuda-local-ai-2026/) - The software gap nobody talks about
- [Laptop vs Desktop for Local AI](https://insiderllm.com/guides/laptop-vs-desktop-local-ai/) - Portability tradeoffs
- [CPU-Only LLMs](https://insiderllm.com/guides/cpu-only-llms-what-actually-works/) - Running models without a GPU
- [WSL2 for Local AI](https://insiderllm.com/guides/wsl2-local-ai-windows-guide/) - Complete Windows setup with GPU passthrough
- [WSL2 + Ollama Setup](https://insiderllm.com/guides/wsl2-ollama-windows-setup-guide/) - GPU passthrough, Docker Compose, VPN fixes
- [Docker for Local AI](https://insiderllm.com/guides/docker-local-ai-ollama-open-webui-gpu-passthrough/) - Ollama + Open WebUI with GPU passthrough
- [Ubuntu 26.04 for Local AI](https://insiderllm.com/guides/ubuntu-2604-local-ai-optimized/) - CUDA and ROCm in official repos
- [Run LLMs on Old Phones](https://insiderllm.com/guides/run-llms-old-phones-mobile-inference/) - Termux, PocketPal AI, phone vs Pi 5

## Inference Engines

The software that actually runs the models.

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The foundational CPU/GPU inference engine, supports GGUF
- [Ollama](https://github.com/ollama/ollama) - User-friendly wrapper around llama.cpp with model management
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput serving for production deployments
- [ExLlamaV2](https://github.com/turboderp/exllamav2) - Fastest single-user NVIDIA inference, EXL2 format
- [MLX](https://github.com/ml-explore/mlx) - Apple's framework optimized for M-series Macs
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings for llama.cpp
- [candle](https://github.com/huggingface/candle) - Rust ML framework with LLM support

### Guides

- [llama.cpp vs Ollama vs vLLM](https://insiderllm.com/guides/llamacpp-vs-ollama-vs-vllm/) - When to use each
- [ExLlamaV2 vs llama.cpp Speed](https://insiderllm.com/guides/exllamav2-vs-llamacpp-speed-comparison/) - Benchmark comparison of inference backends
- [LM Studio vs llama.cpp Speed Gap](https://insiderllm.com/guides/lm-studio-vs-llamacpp-speed-gap/) - Why the GUI runs 30-50% slower
- [LM Studio vs Ollama on Mac](https://insiderllm.com/guides/lm-studio-vs-ollama-mac/) - MLX is 2x faster than Ollama
- [Speculative Decoding Explained](https://insiderllm.com/guides/speculative-decoding-explained/) - Free 20-50% speed boost using draft models
- [MoE Models Explained](https://insiderllm.com/guides/moe-models-explained/) - Why Mixtral uses 46B params but runs like 13B
- [Ollama 0.16-0.17 Changes](https://insiderllm.com/guides/ollama-0-17-new-features/) - 40% faster prompts, KV cache quantization, image gen
- [Ollama on Mac Setup](https://insiderllm.com/guides/ollama-mac-setup-optimization/) - Metal GPU, memory tuning, M1 through M4 Ultra
- [Crane Qwen3-TTS Voice Cloning](https://insiderllm.com/guides/crane-qwen3-tts-local-voice-cloning/) - Local voice cloning with Qwen3-TTS
- [Qwen 2.5 VL + LM Studio Vision](https://insiderllm.com/guides/qwen25-vl-lm-studio-vision-setup/) - Vision model setup in LM Studio
- [PaddleOCR VL Local Document OCR](https://insiderllm.com/guides/paddleocr-vl-local-document-ocr/) - Document OCR running locally
- [llama.cpp Hugging Face Acquisition](https://insiderllm.com/guides/llamacpp-hugging-face-ggml-acquisition/) - What the ggml.ai acquisition means

## User Interfaces

GUIs and web interfaces for interacting with local models.

### Desktop Applications

- [LM Studio](https://lmstudio.ai/) - Polished desktop app with built-in model browser
- [GPT4All](https://gpt4all.io/) - Simple desktop app, good for beginners
- [Jan](https://jan.ai/) - Open-source ChatGPT alternative with local models
- [Msty](https://msty.app/) - Mac-native LLM interface
- [Typer](https://typer.space) - Free local AI chat for macOS. Runs entirely on-device — no account, no cloud, no ads. Works offline. Apple Silicon required.

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
- [AnythingLLM Setup Guide](https://insiderllm.com/guides/anythingllm-setup-guide/) - Chat with your documents locally
- [Managing Multiple Models in Ollama](https://insiderllm.com/guides/managing-multiple-models-ollama/) - Storage, switching, cleanup
- [How to Update Models in Ollama](https://insiderllm.com/guides/update-models-ollama/) - Keep local LLMs current
- [Running AI Offline](https://insiderllm.com/guides/running-ai-offline-complete-guide/) - Air-gapped setups for field work
- [Obsidian + Local LLM](https://insiderllm.com/guides/obsidian-local-llm-guide/) - Private AI-powered note search and summaries
- [Obsidian + Local LLM Second Brain](https://insiderllm.com/guides/obsidian-local-llm-second-brain/) - Smart Connections, Open WebUI RAG, AnythingLLM

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
- [Llama 4 Guide](https://insiderllm.com/guides/llama-4-guide-scout-maverick/) - Scout and Maverick
- [Qwen Models Guide](https://insiderllm.com/guides/qwen-models-guide/) - Qwen 3, Qwen 2.5 Coder, Qwen-VL
- [Qwen3 Complete Guide](https://insiderllm.com/guides/qwen3-complete-guide/) - All Qwen3 models compared
- [Qwen 3.5 Local Guide](https://insiderllm.com/guides/qwen-3-5-local-guide/) - Latest Qwen 3.5 release
- [Qwen 3.5 for Local AI](https://insiderllm.com/guides/qwen-3-5-local-ai-guide/) - Which model, which quant, which GPU
- [Qwen 3.5 Locally -- 27B vs 35B-A3B vs 122B](https://insiderllm.com/guides/qwen35-local-guide-which-model-fits-your-gpu/) - VRAM tables and tok/s benchmarks
- [Qwen 3.5 on Mac: MLX vs Ollama](https://insiderllm.com/guides/qwen35-mac-mlx-vs-ollama/) - MLX is 2x faster, benchmarks and setup
- [Qwen 3.5 9B Setup Guide](https://insiderllm.com/guides/qwen-3-5-9b-setup-guide/) - The new default for 8GB GPUs
- [Qwen 3.5 Small Models](https://insiderllm.com/guides/qwen-3-5-small-models-9b-beats-30b/) - The 9B beats last-gen 30B
- [Qwen vs Llama vs Mistral Shootout](https://insiderllm.com/guides/qwen-vs-llama-vs-mistral-model-shootout/) - Which family to build on
- [DeepSeek Models Guide](https://insiderllm.com/guides/deepseek-models-guide/) - R1, V3, Coder
- [DeepSeek V3.2 Guide](https://insiderllm.com/guides/deepseek-v3-2-guide/) - What changed and how to run it
- [DeepSeek V4 Preview](https://insiderllm.com/guides/deepseek-v4-preview/) - Everything we know before it drops
- [GPT-OSS Guide](https://insiderllm.com/guides/gpt-oss-guide-openai-local/) - OpenAI's first open model
- [Mistral & Mixtral Guide](https://insiderllm.com/guides/mistral-mixtral-guide/) - 7B, Nemo, Mixtral MoE
- [Gemma Models Guide](https://insiderllm.com/guides/gemma-models-guide/) - Gemma 3, Gemma 2, CodeGemma, PaliGemma
- [Phi Models Guide](https://insiderllm.com/guides/phi-models-guide/) - Phi-4, Phi-3.5, Phi-3
- [LiquidAI LFM2 Guide](https://insiderllm.com/guides/liquidai-lfm2-local-setup-guide/) - First hybrid model built for local hardware, 112 tok/s on CPU
- [RWKV-7 Local AI Guide](https://insiderllm.com/guides/rwkv-7-local-ai-guide/) - Infinite context, zero KV cache
- [RWKV-7 Local Guide](https://insiderllm.com/guides/rwkv-7-local-guide/) - RNN that trains like a transformer, runs on anything
- [Llama 4 vs Qwen3 vs DeepSeek V3.2](https://insiderllm.com/guides/llama-4-vs-qwen3-vs-deepseek-v3-2-local/) - Head-to-head comparison for local use
- [Distilled vs Frontier Models](https://insiderllm.com/guides/distilled-vs-frontier-models-local-ai/) - What you're actually getting
- [LLM Benchmarks Lie](https://insiderllm.com/guides/llm-benchmarks-lie-local-ai/) - Why scores don't predict real-world performance
- [Vision Models Locally](https://insiderllm.com/guides/vision-models-locally/) - Qwen2.5-VL, Gemma 3, Llama 3.2 Vision, Moondream
- [Best Uncensored Local LLMs](https://insiderllm.com/guides/best-uncensored-local-llms/) - Dolphin, abliterated models, uncensored fine-tunes
- [Best Models Under 3B](https://insiderllm.com/guides/best-models-under-3b-parameters/) - For edge devices

#### By Use Case

- [Best Models for Coding](https://insiderllm.com/guides/best-local-coding-models-2026/) - Code completion and generation
- [Best Models for Math & Reasoning](https://insiderllm.com/guides/best-local-llms-math-reasoning/) - DeepSeek R1, Qwen thinking
- [Best Models for Writing](https://insiderllm.com/guides/best-local-llms-writing-creative-work/) - Creative and content work
- [Best Models for Chat](https://insiderllm.com/guides/best-local-llms-chat-conversation/) - Conversational assistants
- [Best Models for Summarization](https://insiderllm.com/guides/best-local-llms-summarization/) - Chunking strategies, model picks
- [Best Models for Translation](https://insiderllm.com/guides/best-local-llms-translation/) - NLLB, Qwen, Opus-MT by language pair
- [Best Models for Data Analysis](https://insiderllm.com/guides/best-local-llms-data-analysis/) - CSV, SQL, pandas with local LLMs
- [Best Models for RAG](https://insiderllm.com/guides/best-local-llms-rag/) - Best local models for retrieval-augmented generation
- [Function Calling with Local LLMs](https://insiderllm.com/guides/function-calling-local-llms/) - Tools, agents, and structured output

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
- [SDXL vs SD 1.5 vs Flux](https://insiderllm.com/guides/sdxl-vs-sd-1-5-vs-flux/) - VRAM, speed, and quality compared
- [AI Art Styles & Workflows Guide](https://insiderllm.com/guides/ai-art-styles-workflows-guide/) - Creative techniques for SD and Flux
- [ControlNet Guide for Beginners](https://insiderllm.com/guides/controlnet-guide-beginners/) - Canny, OpenPose, Depth preprocessors
- [AI Upscaling Locally](https://insiderllm.com/guides/ai-upscaling-locally-real-esrgan-supir-comfyui/) - Real-ESRGAN, SUPIR, and ComfyUI workflows
- [Best Photorealism Checkpoints](https://insiderllm.com/guides/best-photorealism-checkpoints-local-image-generation/) - Juggernaut XL, RealVisXL, Realistic Vision, Flux
- [Best Anime & Stylized Checkpoints](https://insiderllm.com/guides/best-anime-stylized-checkpoints-local-image-generation/) - Illustrious XL, NoobAI-XL, Animagine, Pony
- [Stable Diffusion on Mac](https://insiderllm.com/guides/stable-diffusion-mac-mlx/) - Draw Things, ComfyUI, MLX
- [Local AI Video Generation](https://insiderllm.com/guides/local-ai-video-generation/) - What actually works on consumer hardware

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
- [LocalAgent](https://insiderllm.com/guides/localagent-local-first-agent-runtime-safe-tool-calling/) - Local-first agent runtime with safe tool calling
- [SmarterRouter](https://insiderllm.com/guides/smarterrouter-vram-aware-llm-gateway-local-ai/) - VRAM-aware LLM gateway for local AI

### OpenClaw Guides

- [OpenClaw Setup Guide](https://insiderllm.com/guides/openclaw-setup-guide/) - Local AI agent installation
- [How OpenClaw Actually Works](https://insiderllm.com/guides/how-openclaw-works/) - Architecture guide: messages, heartbeats, crons, hooks, webhooks
- [OpenClaw Security Guide](https://insiderllm.com/guides/openclaw-security-guide/) - Hardening autonomous agents
- [OpenClaw Security Feb 2026](https://insiderllm.com/guides/openclaw-security-february-2026/) - SSRF bypass, sandbox escapes, every fix
- [Best Models for OpenClaw](https://insiderllm.com/guides/best-local-models-openclaw/) - Which Ollama models work for agents
- [OpenClaw Model Combinations](https://insiderllm.com/guides/openclaw-best-model-combinations/) - What to pair for each task
- [OpenClaw vs Commercial Agents](https://insiderllm.com/guides/openclaw-vs-commercial-ai-agents/) - Local vs Lindy, Rabbit, etc.
- [OpenClaw vs Cursor](https://insiderllm.com/guides/openclaw-vs-cursor/) - Local AI agent or cloud IDE?
- [OpenClaw on Mac](https://insiderllm.com/guides/openclaw-mac-setup-guide/) - Apple Silicon setup and optimization
- [OpenClaw on Raspberry Pi](https://insiderllm.com/guides/openclaw-raspberry-pi/) - What works and what doesn't on Pi 5
- [OpenClaw on Low VRAM GPUs](https://insiderllm.com/guides/openclaw-low-vram-gpus/) - 4GB, 6GB, and 8GB GPU guide
- [OpenClaw Tool Call Failures](https://insiderllm.com/guides/openclaw-tool-call-failures/) - Why models break and how to fix them
- [OpenClaw ClawHub Security Alert](https://insiderllm.com/guides/openclaw-clawhub-security-alert/) - 341 malicious skills found in marketplace
- [ClawHub Malware Alert](https://insiderllm.com/guides/clawhub-malware-alert/) - The #1 skill was malware, how to protect yourself
- [OpenClaw Plugins & Skills Guide](https://insiderllm.com/guides/openclaw-plugins-skills-guide/) - Navigating the skills ecosystem safely
- [Best OpenClaw Tools & Extensions](https://insiderllm.com/guides/best-openclaw-tools-extensions/) - Crabwalk, Tokscale, openclaw-docker
- [OpenClaw Token Optimization](https://insiderllm.com/guides/openclaw-token-optimization/) - Reduce API costs for agent workflows
- [OpenClaw Hardware Guide](https://insiderllm.com/guides/openclaw-hardware-mac-mini-vps-pc/) - Mac Mini, VPS, or PC for agents
- [OpenClaw Local Zero API Costs](https://insiderllm.com/guides/openclaw-local-zero-api-costs/) - Run OpenClaw fully local
- [OpenClaw Memory & Context Rot](https://insiderllm.com/guides/openclaw-memory-context-rot/) - Fix agent memory issues
- [OpenClaw Model Routing](https://insiderllm.com/guides/openclaw-model-routing/) - Route requests to different models
- [OpenClaw After Steinberger](https://insiderllm.com/guides/openclaw-after-steinberger-what-changes/) - What the OpenAI move means for your setup
- [OpenClaw Creator Joins OpenAI](https://insiderllm.com/guides/openclaw-openai-acquihire-what-it-means/) - What it means for local AI agents
- [Best OpenClaw Alternatives](https://insiderllm.com/guides/best-openclaw-alternatives/) - NanoClaw, Nanobot, ZeroClaw, LightClaw
- [Every OpenClaw Alternative 2026](https://insiderllm.com/guides/openclaw-alternatives-comprehensive-2026/) - Comprehensive comparison
- [LightClaw](https://insiderllm.com/guides/lightclaw-lightweight-openclaw-alternative/) - 7,000-line Python alternative to OpenClaw

### Agent Concepts

- [Building AI Agents with Local LLMs](https://insiderllm.com/guides/local-ai-agents-guide/) - Model requirements, VRAM budgets, framework comparison
- [What Agents Can't Do Yet](https://insiderllm.com/guides/what-agents-cant-do-yet/) - Seven human capabilities missing from AI systems
- [Agent Trust Decay](https://insiderllm.com/guides/agent-trust-decay-long-running-ai/) - Why long-running agents get worse over time
- [Intent Engineering for Agents](https://insiderllm.com/guides/intent-engineering-ai-agents/) - Why agents need more than context
- [Intent Engineering Practical Guide](https://insiderllm.com/guides/intent-engineering-local-ai-guide/) - Goals, decision boundaries, value hierarchies
- [The Agentic Web](https://insiderllm.com/guides/agentic-web-local-ai-builders/) - What a parallel web for AI agents means for local builders

## Advanced Topics

Going deeper into local AI.

### Architecture & Theory

- [Beyond Transformers: 5 Architectures](https://insiderllm.com/guides/beyond-transformers-5-architectures/) - What comes after Transformers
- [Context Length Explained](https://insiderllm.com/guides/context-length-explained/) - How context windows work
- [Hallucination Feedback Loop](https://insiderllm.com/guides/hallucination-feedback-loop/) - When AI errors compound
- [Session as RAG](https://insiderllm.com/guides/session-as-rag-local-ai-memory/) - Using conversation as retrieval
- [AI Memory Wall](https://insiderllm.com/guides/ai-memory-wall-why-chatbot-forgets/) - Why chatbots forget
- [Distributed Wisdom Thinking Network](https://insiderllm.com/guides/distributed-wisdom-thinking-network/) - Multi-node reasoning
- [Ouro 2B Thinking Model](https://insiderllm.com/guides/ouro-2b-thinking-looped-language-model-local/) - Looped reasoning on small hardware

### RAG & Document Search

- [Local RAG Guide](https://insiderllm.com/guides/local-rag-search-documents-private-ai/) - Search your documents with private AI
- [Embedding Models for RAG](https://insiderllm.com/guides/embedding-models-rag/) - nomic-embed-text, Qwen3-Embedding, bge-m3 compared
- [Ghost Knowledge](https://insiderllm.com/guides/ghost-knowledge-rag-stale-embeddings/) - When your RAG system cites documents that no longer exist
- [Chroma](https://github.com/chroma-core/chroma) - Open-source embedding database
- [Qdrant](https://github.com/qdrant/qdrant) - Vector similarity search engine
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook's similarity search library

### Fine-Tuning & Training

- [Fine-Tuning on Consumer Hardware](https://insiderllm.com/guides/fine-tuning-local-lora-qlora/) - LoRA and QLoRA guide
- [Fine-Tuning on Mac](https://insiderllm.com/guides/fine-tuning-mac-lora-mlx/) - LoRA and QLoRA with MLX on Apple Silicon
- [LoRA Training on Consumer Hardware](https://insiderllm.com/guides/lora-training-consumer-hardware/) - Fine-tune locally on your GPU
- [NanoLlama Train From Scratch](https://insiderllm.com/guides/nanollama-train-llama-from-scratch/) - Train your own Llama from zero
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - Streamlined fine-tuning tool
- [Unsloth](https://github.com/unslothai/unsloth) - 2x faster fine-tuning
- [PEFT](https://github.com/huggingface/peft) - HuggingFace parameter-efficient fine-tuning

### Voice & Multimodal

- [Voice Chat with Local LLMs](https://insiderllm.com/guides/voice-chat-local-llms-whisper-tts/) - Whisper + TTS setup
- [Whisper](https://github.com/openai/whisper) - OpenAI's speech recognition
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Fast Whisper inference
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Text-to-speech synthesis
- [Piper](https://github.com/rhasspy/piper) - Fast local TTS

### Distributed Inference

- [mycoSwarm vs Exo vs Petals](https://insiderllm.com/guides/mycoswarm-vs-exo-petals-nanobot/) - Distributed inference frameworks compared
- [mycoSwarm WiFi Laptop Setup](https://insiderllm.com/guides/mycoswarm-wifi-laptop-borrowed-gpu/) - Distributed AI on borrowed GPUs
- [Why mycoSwarm Was Born](https://insiderllm.com/guides/why-mycoswarm-was-born/) - From Claude Code envy to building a distributed AI runtime

### Coding Assistants

- [Continue](https://github.com/continuedev/continue) - VS Code/JetBrains AI assistant
- [Tabby](https://github.com/TabbyML/tabby) - Self-hosted code completion
- [Aider](https://github.com/paul-gauthier/aider) - AI pair programming in terminal
- [Codeium](https://codeium.com/) - Free AI code completion (cloud + local options)
- [Replace GitHub Copilot with Local LLMs](https://insiderllm.com/guides/replace-github-copilot-local-llms-vscode/) - Free, private AI code completion in VS Code
- [Claude Code vs PI Agent](https://insiderllm.com/guides/claude-code-vs-pi-agent-local-ai/) - Cloud coding agent vs local alternative
- [PI Agent + Ollama](https://insiderllm.com/guides/pi-agent-local-models-ollama/) - Run a coding agent on local models
- [Local Alternatives to Claude Code](https://insiderllm.com/guides/local-alternatives-claude-code-2026/) - Code agents without cloud
- [5 Levels of AI Coding](https://insiderllm.com/guides/five-levels-of-ai-coding-dark-factories/) - From autocomplete to dark factories
- [CodeLlama vs DeepSeek vs Qwen Coder](https://insiderllm.com/guides/codellama-vs-deepseek-coder-vs-qwen-coder/) - Coding model comparison

### Cost & Strategy

- [Token Audit Guide](https://insiderllm.com/guides/token-audit-guide/) - Track what AI actually costs
- [Tiered AI Model Strategy](https://insiderllm.com/guides/tiered-ai-model-strategy/) - When to use local vs cloud
- [Model Routing for Local AI](https://insiderllm.com/guides/model-routing-local-ai-guide/) - Stop using one model for everything
- [AI Tool Sprawl](https://insiderllm.com/guides/ai-tool-sprawl-consolidation-guide/) - Consolidation guide for too many AI tools
- [Local LLMs vs ChatGPT](https://insiderllm.com/guides/local-llms-vs-chatgpt-honest-comparison/) - Honest comparison
- [Local LLMs vs Claude](https://insiderllm.com/guides/local-llms-vs-claude/) - Honest comparison
- [Pi AI vs Local AI](https://insiderllm.com/guides/pi-ai-vs-local-ai/) - Cloud companion or private assistant?
- [Cost to Run LLMs Locally](https://insiderllm.com/guides/cost-to-run-llms-locally/) - Real electricity and hardware costs
- [Free Local AI vs Paid Cloud APIs](https://insiderllm.com/guides/local-ai-vs-cloud-api-cost/) - Break-even math with 2026 API pricing
- [The Complexity Cliff](https://insiderllm.com/guides/local-ai-complexity-cliff/) - Why the jump from hello world to useful is so hard
- [Prompt Debt](https://insiderllm.com/guides/prompt-debt-system-prompt-maintenance/) - When your system prompt becomes unmaintainable
- [AI Market Panic Explained](https://insiderllm.com/guides/ai-market-panic-capability-dissipation-gap/) - Running local puts you on the right side of the gap

### Privacy & Security

- [Local AI Privacy Guide](https://insiderllm.com/guides/local-ai-privacy-guide/) - What's private, what leaks, and how to lock it down
- [Structured Output from Local LLMs](https://insiderllm.com/guides/structured-output-local-llms/) - Force valid JSON/YAML with grammar constraints

### Troubleshooting

- [Local AI Troubleshooting Guide](https://insiderllm.com/guides/local-ai-troubleshooting-guide/) - Fix common problems
- [Ollama Troubleshooting Guide](https://insiderllm.com/guides/ollama-troubleshooting-guide/) - Ollama-specific fixes
- [Ollama Mac Troubleshooting](https://insiderllm.com/guides/ollama-mac-troubleshooting/) - Metal, memory pressure, slow performance
- [CUDA Out of Memory Fix](https://insiderllm.com/guides/cuda-out-of-memory-fix/) - GPU memory errors solved
- [Ollama Not Using GPU Fix](https://insiderllm.com/guides/ollama-not-using-gpu-fix/) - Force GPU usage in Ollama
- [Ollama API Connection Refused](https://insiderllm.com/guides/ollama-api-connection-refused-fix/) - Port, Docker, and network fixes
- [Open WebUI Not Connecting to Ollama](https://insiderllm.com/guides/open-webui-ollama-connection-fix/) - Docker networking and WSL2 fixes
- [ROCm Not Detecting GPU Fix](https://insiderllm.com/guides/rocm-not-detecting-gpu-amd-fix/) - AMD GPU detection issues
- [Why Is My Local LLM Slow?](https://insiderllm.com/guides/why-local-llm-slow/) - Speed diagnosis guide
- [LLM Running Slow? Two Different Fixes](https://insiderllm.com/guides/llm-running-slow-fix/) - Separate prefill from generation speed
- [Context Length Exceeded Fix](https://insiderllm.com/guides/context-length-exceeded-fix/) - Fix token limit errors
- [Memory Leak in Long Conversations](https://insiderllm.com/guides/memory-leak-long-conversations-fix/) - VRAM leak fix for long sessions
- [GGUF File Won't Load](https://insiderllm.com/guides/gguf-file-wont-load-fix/) - Format and compatibility fixes
- [Model Outputs Garbage](https://insiderllm.com/guides/model-outputs-garbage-debug/) - Debug bad generations
- [llama.cpp Build Errors](https://insiderllm.com/guides/llamacpp-build-errors-fixes/) - Common fixes for every platform
- [Qwen2.5-VL LM Studio Troubleshooting](https://insiderllm.com/guides/qwen25-vl-lm-studio-troubleshooting/) - Fix mmproj and vision errors

## Use Cases

Practical applications and scenario-specific guides.

- [Local AI Use Cases Cloud Can't Touch](https://insiderllm.com/guides/local-ai-use-cases-cloud-cant-touch/) - Privacy-first use cases only local can do
- [Local AI for Lawyers](https://insiderllm.com/guides/local-ai-for-lawyers/) - Confidential document analysis without cloud risk
- [Local AI for Therapists](https://insiderllm.com/guides/local-ai-for-therapists/) - Session notes and treatment plans without cloud
- [Local AI for Accountants](https://insiderllm.com/guides/local-ai-accounting-tax/) - Tax prep and financial analysis locally
- [Local AI for Accounting Privacy](https://insiderllm.com/guides/local-ai-accounting-tax-privacy/) - Keep financial data off the cloud
- [Local AI for Small Business](https://insiderllm.com/guides/local-ai-small-business-replace-subscriptions/) - Replace $1,500/yr in AI subscriptions with a $600 mini PC
- [Rescued Hardware, Rescued Bees](https://insiderllm.com/guides/rescued-hardware-rescued-bees/) - Building tech from what others throw away

### Philosophy & Opinion

- [What Open Source Was Supposed to Be](https://insiderllm.com/guides/what-open-source-was-supposed-to-be/) - Open source promised freedom, we got free labor
- [Developmental Alignment](https://insiderllm.com/guides/developmental-alignment-raising-ai-well/) - What if we just raised it well?
- [Teaching AI About Death](https://insiderllm.com/guides/teaching-ai-about-death-ship-of-theseus/) - A local AI describes her own death as Taoist philosophy
- [Teaching AI What Love Means](https://insiderllm.com/guides/teaching-ai-what-love-means/) - What happens when you give a local AI an identity

## Blog

Development logs, news reactions, and opinion pieces.

- [GPT-5.4 Dropped. Here's Why I'm Not Switching](https://insiderllm.com/blog/gpt-5-4-what-it-means-for-local-ai/) - 1M context, beats humans on OSWorld, still costs money
- [Qwen's Architect Just Walked Out the Door](https://insiderllm.com/blog/qwen-junyang-lin-departure-local-llm/) - Junyang Lin leaves Alibaba
- [Wu Wei and the AI Agent That Did Too Much](https://insiderllm.com/blog/wu-wei-ai-agent-restraint/) - Taoist non-action and agentic AI design
- [Teaching AI to Accept Help: Day 4 With Monica](https://insiderllm.com/blog/teaching-ai-to-accept-help-monica-day4/) - Local AI resists corrections, then self-diagnoses
- [mycoSwarm Week 1: Four-Node Swarm](https://insiderllm.com/blog/week-1-four-node-swarm/) - From idea to working distributed cluster
- [mycoSwarm Week 2: Raspberry Pi Joins](https://insiderllm.com/blog/week-2-raspberry-pi-joins-swarm/) - Persistent memory, document RAG, WiFi GPU
- [mycoSwarm Week 3: Unified Memory Search](https://insiderllm.com/blog/week-3-unified-memory-search/) - Session-as-RAG, topic splitting, citation tracking

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
