from pathlib import Path

import torch
from transformers import AutoTokenizer

from llm_compressor.utils.args import build_parser
from llm_compressor.utils.general import print_eval
from llm_compressor.models.opt import CompressOPTForCausalLM
from llm_compressor.evaluation.eval import LMEvaluator


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

############### Model Compression ###############
model.quantize(
    None, 
    quant_method="rtn", 
    quant_config=args.quant_config, 
    device=device,
    quantize=args.quantize
)

############### Model Evaluation ###############
evaluator = LMEvaluator(device=device, n_samples=128)
eval_kwargs = {
    "tokenizer_path": args.model,
    "seq_len": args.seq_len,
    "batch_size": args.batch_size,
}
results = evaluator.eval(model, tasks=args.tasks, **eval_kwargs)
print_eval(results)
    