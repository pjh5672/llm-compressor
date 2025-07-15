# ✨ Quantization

### 🦜 Support Algorithms
- RTN(Rount-to-Nearest)
- SmoothQuant
- GPTQ
- AWQ
- AWQ+
- SpinQuant

### 🦁 API Usage

```python
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = CompressLlamaForCausalLM.from_pretrained(
    args.model,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

model.quantize(
    tokenizer=tokenizer,
    quant_method=args.quant_method,
    quant_config=args.quant_config,
    device=device,
    quantize=args.quantize,
    n_samples=args.calib_num,
    seq_len=args.seq_len,
    rotation_path=args.rotation_path,
)
```

#### PPL Evaluation

| Model | BF16 | RTN | SmoothQuant | GPTQ | AWQ | AWQ+ | SpinQuant | GPTAQ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| OPT-125M | 42.1965 | 47.8449 |  | 47.8277 | 47.3636 | 46.5827 | ❌ |  |
| OPT-350M | 34.2866 | 38.5774 |  | 36.1052 | 37.6461 | 36.4058 | ❌ |  |
| OPT-1.3B | 21.9675 | 23.3960 |  | 22.1323 | 22.6504 | 22.3655 | ❌ |  |
| BLOOM-560M | 35.7978 | 38.6716 |  | 37.7014 | 39.2614 | 38.3677 | ❌ |  |
| BLOOM-1.1B | 25.0659 | 26.5992 |  |25.7862 | 27.1075 | 26.1620 | ❌ |  |
| BLOOM-1.7B | 21.4909 | 22.7827 |  | 22.0410 | 23.1552 | 22.3943 | ❌ |  |
| BLOOM-3B | 18.9668 | 19.9891 |  | 19.3825 | 20.3097 | 19.7093 | ❌ |  |
| Phi-1.5 | 33.1104 | 35.1125 |  | 34.5224 | 54.3485 | 53.5180 | ❌ |  |
| Phi-2.0 | 13.6094 | 14.6068 |  | 15.0878 | 83.0253 | 83.8705 | ❌ |  |
| Gemma-2B-Instruct |  |  |  |  |  |  | ❌ |  |
| Gemma-2-2B-Instruct |  |  |  |  |  |  | ❌ |  |
| Gemma-3-1B-Instruct |  |  |  |  |  |  | ❌ |  |
| Gemma-3-4B-Instruct |  |  |  |  |  |  | ❌ |  |
| Qwen-2.5-0.5B-Instruct |  |  |  |  |  |  | ❌ |  |
| Qwen-3.0-1.7B |  |  |  |  |  |  | ❌ |  |
| Qwen-3.0-4B |  |  |  |  |  |  | ❌ |  |
| Llama-3.2-1B-Instruct | 19.0228 | 25.5581 |   | 22.8635 | 24.3941 | 22.6888 | 21.1071 |  |
| Llama-3.2-3B-Instruct | 15.8560 | 17.6223 |   | 17.0927 | 18.1077 | 17.3577 | 16.9951 |  |
