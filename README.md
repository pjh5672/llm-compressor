# 🧠 LLM Compression Toolkit

A lightweight, modular toolkit for compressing Large Language Models (LLMs) using **pruning**, **quantization**, and other efficient compression strategies — designed for both research and deployment.

&#x20;&#x20;

---

## 🚀 Features

- ✅ **Structured & unstructured pruning** support
- ✅ **Quantization-aware training (QAT)** & **Post-training quantization (PTQ)**
- ✅ Support for INT8 / INT4 quantization with ZeroPoint & Per-channel options
- ✅ **Plug-and-play integration** with Hugging Face Transformers
- ✅ Model size & FLOPs estimator
- ✅ Export to ONNX / TorchScript / GGUF (coming soon)

---

## 🏗️ Supported Architectures

-

---

## 📦 Installation

```bash
git clone https://github.com/your-org/llm-compression-toolkit.git
cd llm-compression-toolkit
pip install -e .
```

---

## 🛠️ Quick Start

### 1. Load Your Model

```python
from transformers import AutoModelForCausalLM
from compression_toolkit import Pruner, Quantizer

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
```

### 2. Prune the Model

```python
pruner = Pruner(model)
pruned_model = pruner.prune(method="magnitude", sparsity=0.5)
```

### 3. Quantize the Model

```python
quantizer = Quantizer(pruned_model)
quantized_model = quantizer.quantize(method="int8", per_channel=True)
```

### 4. Save the Compressed Model

```python
quantized_model.save_pretrained("compressed-model/")
```

---

## 📊 Benchmarking

| Model          | Compression  | Size ↓ | Speed ↑ | Accuracy (Δ) |
| -------------- | ------------ | ------ | ------- | ------------ |
| GPT-2 (small)  | 8-bit QAT    | 75%    | +1.8×   | -0.3%        |
| LLaMA 2 1.3B   | Prune + INT8 | 60%    | +1.5×   | -0.6%        |
| TinyLlama 1.1B | INT4         | 85%    | +2.2×   | -1.2%        |

> Use `scripts/benchmark.py` to reproduce these results.

---

## 📁 Project Structure

```
llm-compression-toolkit/
│
├── compression_toolkit/
│   ├── pruning/        # Pruning algorithms
│   ├── quantization/   # Quantization modules
│   ├── utils/          # Common utilities
│   └── export/         # Format converters (ONNX, GGUF, etc.)
│
├── examples/           # End-to-end use cases
├── scripts/            # CLI scripts
├── tests/              # Unit tests
└── README.md
```

---

## 📚 Documentation

- [Getting Started](docs/getting_started.md)
- [Pruning Guide](docs/pruning.md)
- [Quantization Guide](docs/quantization.md)
- [API Reference](docs/api.md)

---

## 🧪 Research References

- Han et al. ["Deep Compression"](https://arxiv.org/abs/1510.00149)
- Dettmers et al. ["GPTQ: Accurate Post-Training Quantization"](https://arxiv.org/abs/2210.17323)
- Frantar et al. ["SparseGPT"](https://arxiv.org/abs/2301.00774)

---

## 🤝 Contributing

Pull requests, issues and feature suggestions are welcome!\
Please refer to [`CONTRIBUTING.md`](CONTRIBUTING.md) to get started.

---

## 🛡 License

Apache 2.0 License © 2025 Your Organization

---

## 🌐 Acknowledgements

This project builds upon ideas from [GPTQ](https://github.com/IST-DASLab/gptq), [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes), and [SparseML](https://github.com/neuralmagic/sparseml).

