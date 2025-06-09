import os
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llm_compressor.utils.args import build_parser
from llm_compressor.utils.general import (LOGGER, print_args, print_eval, init_seeds, file_date)
from llm_compressor.utils.torch_utils import select_device
from llm_compressor.evaluation.eval import LMEvaluator

ROOT = Path(__file__).resolve().parents[1]
args = build_parser(ROOT)
LOGGER.add(args.exp_dir / f"{file_date()}.log", level="DEBUG")

print_args(
    args=args, exclude_keys=("exp_dir", "save_dir", "q_config"), logger=LOGGER
)
init_seeds(args.seed, deterministic=True)
device = select_device(device=args.device, batch_size=args.batch_size)
assert device.type != "cpu", "can not support CPU mode."

############### Model Definition ###############
config = AutoConfig.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    attn_implementation="eager",
    torch_dtype=config.torch_dtype,
    device_map="cpu",
)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(args.model)  # noqa: F841

############### Model Evaluation ###############
LOGGER.info(f"Evaluating compressed model from {args.model.split(os.sep)[-1]}")
evaluator = LMEvaluator(device=device, n_samples=128)
eval_kwargs = {
    "tokenizer_path": args.model,
    "seq_len": args.seq_len,
    "batch_size": args.batch_size,
}
results = evaluator.eval(model, tasks=args.tasks, **eval_kwargs)
print_eval(results, logger=LOGGER)
    