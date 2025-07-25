# 🧩 Pruning

### 🐬 Support Algorithms
- Magnitude
- Wanda
- RIA

### 🤖 API Usage

```python
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = CompressLlamaForCausalLM.from_pretrained(
    args.model,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

model.prune(
    tokenizer=tokenizer,
    prune_method=args.prune_method,
    prune_config=args.prune_config,
    device=device,
    prune=args.prune,
)
```

#### PPL Evaluation 
- sparsity 30%

| Model | BF16 | Magnitude | Wanda | SparseGPT | RIA |
| :---: | :---: | :---: | :---: | :---: | :---: |
| OPT-125M | 42.1965 | 51.5627 | 43.3303 | 44.3394 | 44.0486 |
| OPT-350M | 34.2866 | 42.0008 | 36.3439 | 35.7843 | 36.5450 |
| OPT-1.3B | 21.9675 | 34.9381 | 21.8975 | 22.3482 | 22.2084 |
| BLOOM-560M | 35.7978 | 41.2604 | 37.1605 | 37.5585 | 37.2499 |
| BLOOM-1.1B | 25.0659 | 27.2390 | 26.1514 | 26.3445 | 26.2328 |
| BLOOM-1.7B | 21.4909 | 23.3717 | 22.2842 | 22.1721 | 22.2066 |
| BLOOM-3B | 18.9668 | 19.8218 | 19.7028 | 19.5193 | 19.4788 |
| Phi-1.5 | 33.1104 | 37.4572 | 34.7241 | 34.6615 | 34.3258 |
| Phi-2.0 | 13.6094 | 15.6271 | 14.3896 | 14.3117 | 14.2094 |
| Gemma-2B-Instruct | 307.1260 | 316.1830 | 303.644 | 295.7415 | 357.876 |
| Gemma-2-2B-Instruct | 29.5611 | 49.8580 | 30.9540 | 32.5232 | 31.5795 |
| Gemma-3-1B-Instruct | 61.1776 | 123.1650 | 70.5157 | 70.9543 | 70.4440 |
| Gemma-3-4B-Instruct | 64.0732 | 99.6258 | 72.1694 | 73.6765 | 76.2284 |
| Qwen-2.5-0.5B-Instruct | 19.1299 | 31.7231 | 21.0839 | 20.8793 | 20.9355 |
| Qwen-3.0-1.7B | 21.8734 | 34.2607 | 22.9871 | 22.8272 | 22.5880 |
| Qwen-3.0-4B | 18.1153 | 18.4853 | 19.2171 | 19.4969 | 18.4201 |
| Llama-3.2-1B-Instruct | 19.0228 | 33.8913 | 21.8935 | 21.5101 | 21.2587 |
| Llama-3.2-3B-Instruct | 15.8560 | 22.1732 | 17.0738 | 17.1330 | 17.0796 |
