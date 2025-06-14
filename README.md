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

As for the argument(args), you can following the information below.
```
options: 
    -h, --help          show this help message and exit
    --model             Path to HF model
    --exp-name          Name to project
    --quantize          Enable to quantize model
    --quant-method      Quantization method
    --weight            Quantization config for weight, following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. (e.g. 'int4-g[-1]-zp-rw' means int4-asymetric-per_token quant)
    --act-in            Quantization config for input activation, following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. (e.g. 'int8-g[-1]-zp-rw' means int8-asymetric-per_token quant)
    --act-out           Quantization config for output activation, following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. (e.g. 'int8-g[-1]-zp-rw' means int8-asymetric-per_token quant)
    --head              Quantization config for head weight, following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. (e.g. 'int8-g[-1]-zp-rw' means int8-asymetric-per_token quant)
    --prune             Enable to prune model
    --prune-method      Prune method
    --sparsity          Sparsity ratio
    --calib-data        Calibration dataset
    --calib-num         Number of calibration dataset
    --save-path         Path to save compressed model
    --tasks             Evaluation tasks
    --seq-len           Sequence length for calibration and evaluation
    --batch-size        Evaluation batch size
    --device            cuda devices, i.e. 0 or 0,1,2,3 or cpu
    --seed              Inference seed
```

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
    tokenizer=tokenizer,
    prune_method=args.prune_method,
    prune_config=args.prune_config,
    device=device,
    prune=args.prune,
)
```

#### 4. Quantize the Model

```python
quant_kwargs = {
    "n_samples": 128,
    "seq_len": 512,
}
model.quantize(
    tokenizer=tokenizer,
    quant_method=args.quant_method,
    quant_config=args.quant_config,
    device=device,
    quantize=args.quantize,
    **quant_kwargs
)
```

#### 5. Evaluate the Model

```python
evaluator = LMEvaluator(device=device, n_samples=128)
eval_kwargs = {
    "tokenizer_path": args.model,
    "seq_len": 512,
    "batch_size": 1,
    "check_sparsity": True,
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
