from pathlib import Path

import torch
from transformers import AutoTokenizer

from llm_compressor.utils.args import build_parser
from llm_compressor.models.llama import CompressLlamaForCausalLM


############### Init Arguments ###############
ROOT = Path(__file__).resolve().parents[1]
args, device = build_parser(ROOT)


############### Model Definition ###############
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = CompressLlamaForCausalLM.from_pretrained(
    args.model,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

############### Model Pruning ###############
model.prune(
    tokenizer=tokenizer,
    prune_method=args.prune_method,
    prune_config=args.prune_config,
    device=device,
    prune=args.prune,
)

############### Model Quantization ###############
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

############### Interact with Instruct-Model ###############
prompt = "What does Llama eat?"
response = model.generate_text(prompt=prompt, 
                               tokenizer=tokenizer, 
                               seq_len=args.seq_len, 
                               max_new_tokens=512,
                               temperature=0.0, 
                               top_k=5)
print(response)