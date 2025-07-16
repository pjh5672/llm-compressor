# 🦅 Getting Started

- [Installation](#-installation)

- [Quick Start](#-quick-start)
  - [1. Get Your Argument](#1-get-your-argument)
  - [2. Load Your Model](#2-load-your-model)
  - [3. Prune the Model](#3-prune-the-model)
  - [4. Quantize the Model](#4-quantize-the-model)
  - [5. Evaluate the Model](#5-evaluate-the-model)
  - [6. Save the Model](#6-save-the-model)

- [How to Profile Model](#)


### 📦 Installation

```bash
git clone https://github.com/your-org/llm-compression-toolkit.git
cd llm-compression-toolkit
pip install -e .
```

---

### 🛠️ Quick Start


#### 1. Get Your Argument

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

#### 4. Quantize the Model

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

#### 5. Evaluate the Model

```python
evaluator = LMEvaluator(
    model=model, 
    device=device,
    n_samples=None
)
eval_kwargs = {
    "tokenizer_path": args.model,
    "seq_len": args.seq_len,
    "batch_size": args.batch_size,
    "is_check_sparsity": args.prune,
}
results = evaluator.eval(tasks=args.tasks, **eval_kwargs)
print_eval(results)
```

#### 6. Save the Model
```python
model.save_compressed(args.model, args.save_path)
```

---


### How to Profile Model

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

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = CompressLlamaForCausalLM.from_pretrained(
    args.model,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

model.profile(
    quant_config=args.quant_config,
    device=device,
    save_path=args.exp_dir,
)

# Generate below profile results(stats.csv) to disk

                             Op Name,         PC99%,           Max,      QDQ(Max),          SQNR
    layers.0.self_attn.q_proj.weight,      0.095703,       0.58203,       0.58203,         47.75
    layers.0.self_attn.k_proj.weight,       0.12451,       0.64453,       0.64453,          41.5
    layers.0.self_attn.v_proj.weight,      0.023926,       0.06543,       0.06543,          40.5
    layers.0.self_attn.o_proj.weight,      0.028809,       0.30273,       0.30273,          50.5
    ....

        layers.15.mlp.up_proj.output,        1.8359,            16,            16,            62
       layers.15.mlp.down_proj.input,       0.55469,           112,           112,            78
      layers.15.mlp.down_proj.output,        1.2578,           158,           158,          71.5
                       lm_head.input,        5.3125,         44.75,         44.75,            80
                      lm_head.output,        8.0625,        28.125,        28.125,            80

```