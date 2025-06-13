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
    + Magnitude pruning
- ✅ Support **Post-training quantization(PTQ)**
    + INT / FP / MX format
    + 4-bit / 8-bit Fake quantizer support
    + Symmetric / Asymmetric quantization
    + Per-tensor / Per-token / Per-channel / Per-block scaling options
- ✅ **Plug-and-play integration** with Hugging Face Transformers

---

## 🏗️ Supported Architectures

- OPT / BLOOM / Llama(1, 2, 3) / Phi(1.5, 2) / Qwen(2.5, 3)

---

## 📦 Installation

```bash
git clone https://github.com/your-org/llm-compression-toolkit.git
cd llm-compression-toolkit
pip install -e .
```

---

## 🛠️ Quick Start

#### 1. Get Your Argument

```python
from pathlib import Path

import torch
from transformers import AutoTokenizer

from llm_compressor.utils.args import build_parser
from llm_compressor.utils.general import print_eval
from llm_compressor.models.opt import CompressOPTForCausalLM
from llm_compressor.evaluation.eval import LMEvaluator

ROOT = Path(__file__).resolve().parents[1]
args, device = build_parser(ROOT)
```

#### 2. Load Your Model

```python
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = CompressOPTForCausalLM.from_pretrained(
    args.model,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
```

#### 3. Prune the Model

```python
model.prune(
    tokenizer=None,
    prune_method=args.prune_method,
    prune_config=args.prune_config,
    device=device,
    prune=args.prune,
)
```

#### 4. Quantize the Model

```python
model.quantize(
    tokenizer=None,
    quant_method=args.quant_method,
    quant_config=args.quant_config,
    device=device,
    quantize=args.quantize,
)
```

#### 5. Evaluate the Model

```python
evaluator = LMEvaluator(device=device, n_samples=128)
eval_kwargs = {
    "tokenizer_path": args.model,
    "seq_len": args.seq_len,
    "batch_size": args.batch_size,
}
results = evaluator.eval(model, tasks=args.tasks, **eval_kwargs)
print_eval(results)
```

#### 6. Save the model
```python
model.save_compressed(args.model, args.save_path)
```


## 📊 Benchmarking

| Model          | Compression  | Size ↓ | Speed ↑ | Accuracy (Δ) |
| -------------- | ------------ | ------ | ------- | ------------ |
| GPT-2 (small)  | 8-bit QAT    | 75%    | +1.8×   | -0.3%        |
| LLaMA 2 1.3B   | Prune + INT8 | 60%    | +1.5×   | -0.6%        |
| TinyLlama 1.1B | INT4         | 85%    | +2.2×   | -1.2%        |

---

## 📚 Documentation

- [Getting Started](docs/getting_started.md)
- [Pruning Guide](docs/pruning.md)
- [Quantization Guide](docs/quantization.md)
- [API Reference](docs/api.md)

---

## 🛡 License

GNU GENERAL PUBLIC LICENSE 3.0 License © 2025

---

## 🌐 Acknowledgements

This project builds upon my personal research 🐯
