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
- ✅ Support **Profiling**
    + Percentile(99%) / Max / Fake Quantized Max / SQNR / Kurtosis for operations
- ✅ **Plug-and-play integration** with Hugging Face Transformers
- ✅ **Tinychat** with Compressed Instruct-model

---

## 🏗️ Supported Algorithms for SLMs

| Model | RTN | GPTQ | AWQ | AWQ+ | SpinQuant |
| :---: | :---: | :---: | :---: | :---: | :---: |
| OPT | ✅ | ✅ | ✅ | ✅ | ❌ | 
| BLOOM | ✅ | ✅ | ✅ | ✅ | ❌ | 
| Llama1 | ✅ | ✅ | ✅ | ✅ | ✅ | 
| Llama2 | ✅ | ✅ | ✅ | ✅ | ✅ | 
| Llama3.x | ✅ | ✅ | ✅ | ✅ | ✅ | 
| Phi1.5 | ✅ | ✅ | ✅ | ✅ | ❌ | 
| Phi2 | ✅ | ✅ | ✅ | ✅ | ❌ | 
| Qwen2.x | ✅ | ✅ | ✅ | ✅ | ❌ | 
| Qwen3 | ✅ | ✅ | ✅ | ✅ | ❌ | 

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
    --profile           Enable to profile model
    --quantize          Enable to quantize model
    --quant-method      Quantization method
    --weight            Quantization config for weight, following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. (e.g. 'int4-g[-1]-zp-rw' means int4-asymetric-per_token quant)
    --act-in            Quantization config for input activation, following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. (e.g. 'int8-g[-1]-zp-rw' means int8-asymetric-per_token quant)
    --act-out           Quantization config for output activation, following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. (e.g. 'int8-g[-1]-zp-rw' means int8-asymetric-per_token quant)
    --head              Quantization config for head weight, following pattern of [type]-[group_size]-[zero_point]-[quant_wise]. (e.g. 'int8-g[-1]-zp-rw' means int8-asymetric-per_token quant)
    --rotation-path     Path to rotation matrix for spinquant
    --prune             Enable to prune model
    --prune-method      Prune method
    --sparsity          Sparsity ratio
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
from llm_compressor.evaluation.eval import LMEvaluator
from llm_compressor.models.llama import CompressLlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
args, device = build_parser(ROOT)
```

#### 2. Load Your Model

```python
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = CompressLlamaForCausalLM.from_pretrained(
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

#### 4. Profile the Model

```python
if args.profile:
    model.profile(
        quant_config=args.quant_config,
        device=device,
        save_path=args.exp_dir,
    )
    args.qparser.disable_profile(args.quant_config)
```

#### 5. Quantize the Model

```python
quant_kwargs = {
    "n_samples": args.calib_num,
    "seq_len": args.seq_len,
    "rotation_path": args.rotation_path,
}
model.quantize(
    tokenizer=tokenizer,
    quant_method=args.quant_method,
    quant_config=args.quant_config,
    device=device,
    quantize=args.quantize,
    **quant_kwargs,
)
```

#### 6. Evaluate the Model

```python
evaluator = LMEvaluator(
    model=model, 
    n_samples=128, 
    is_check_sparsity=args.prune
)
eval_kwargs = {
    "tokenizer_path": args.model,
    "seq_len": args.seq_len,
    "batch_size": args.batch_size,
}
results = evaluator.eval(tasks=args.tasks, **eval_kwargs)
print_eval(results)
```

#### 7. Save the Model
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
