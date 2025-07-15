<div align="center">

# LLM Compression Toolkit

  <p>
    <a><img width="100%" src="assets/banner.jpg" alt="LLM compressor banner"></a>
  </p>

A lightweight, modular toolkit for compressing Large Language Models (LLMs) using **pruning**, **quantization**, and other efficient compression strategies — designed for both research and deployment.

</div>

---

## 🚀 Features

- ✅ Support **Unstructured pruning**
    + Magnitude alogithm
- ✅ Support **Post-training quantization(PTQ)**
    + INT / FP / MX / NVFP4 format
    + 4-bit / 8-bit Fake quantizer support
    + Symmetric / Asymmetric quantization
    + RTN / GPTQ / AWQ / AWQ+ / SpinQuant algorithms
    + Per-tensor / Per-token / Per-channel / Per-block scaling options
- ✅ **Model Profiling**
    + Percentile(99%) / Max / Fake Quantized Max / SQNR for operations
- ✅ Support **Mixed-Precision Quantization(MPQ)**
- ✅ **Plug-and-play integration** with Hugging Face Transformers
- ✅ **Tinychat** with Compressed Instruct-model

---

## 🏗️ Supported Small Language Models

 - OPT-125M, OPT-350M, OPT-1.7B
 - BLOOM-560M, BLOOM-1.1B, BLOOM-1.7B, BLOOM-3B
 - Phi-1.5, Phi-2.0
 - Qwen-2.5-0.5B-Instruct, Qwen-3.0-1.7B, Qwen-3.0-4B
 - Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct
 - Gemma-2B-Instruct, Gemma-2-2B-Instruct, Gemma-3-1B-Instruct, Gemma-3-4B-Instruct

---

## 📚 Documentation

- [Getting Started](docs/getting_started.md)
- [Pruning Guide](docs/pruning.md)
- [Quantization Guide](docs/quantization.md)

---

## 🛡 License

GNU GENERAL PUBLIC LICENSE 3.0 License © 2025

---

## 🌐 Acknowledgements

This project builds upon my personal research 🐯
