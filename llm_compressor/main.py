import os
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from evaluation.eval import LMEvaluator
from utils.args import build_parser
from utils.general import LOGGER, print_args, print_eval, init_seeds, file_date
from utils.torch_utils import select_device

ROOT = Path(__file__).resolve().parents[0]

"""
Here, we need to define AutoAWQ-like compressed model wrapping from HF_Model

print(f"Loading model from: {args.hf_model_path}")
model = AutoAWQForCausalLM.from_pretrained(
    args.hf_model_path,
    device_map=args.device_map,
)
tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)

print(f"Quantizing model with config: {quant_config}")
model.quantize(
    tokenizer,
    quant_config=quant_config,
    max_calib_samples=args.max_calib_samples,
    max_calib_seq_len=args.max_calib_seq_len,
)

print(f"Saving quantized model to: {args.local_save_path}")
model.save_quantized(args.local_save_path)
tokenizer.save_pretrained(args.local_save_path)

print(f"Quantized model '{args.quant_name}' saved successfully.")
"""

if __name__ == "__main__":
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
