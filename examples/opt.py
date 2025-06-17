from pathlib import Path

import torch
from transformers import AutoTokenizer

from llm_compressor.utils.args import build_parser
from llm_compressor.utils.general import print_eval
from llm_compressor.evaluation.eval import LMEvaluator
from llm_compressor.models.opt import CompressOPTForCausalLM


############### Init Arguments ###############
ROOT = Path(__file__).resolve().parents[1]
args, device = build_parser(ROOT)

############### Model Definition ###############
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = CompressOPTForCausalLM.from_pretrained(
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
}
model.quantize(
    tokenizer=tokenizer,
    quant_method=args.quant_method,
    quant_config=args.quant_config,
    device=device,
    quantize=args.quantize,
    **quant_kwargs,
)

############### Model Evaluation ###############
evaluator = LMEvaluator(model=model, n_samples=128)
eval_kwargs = {
    "tokenizer_path": args.model,
    "seq_len": args.seq_len,
    "batch_size": args.batch_size,
    "check_sparsity": args.prune,
}
results = evaluator.eval(tasks="ppl", **eval_kwargs)
print_eval(results)

############### Model Saving ###############
if args.save_path is not None:
    model.save_compressed(args.save_path)
