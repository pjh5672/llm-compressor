import os
from pathlib import Path

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from evaluation.eval import LMEvaluator
from utils.args import build_parser
from utils.general import LOGGER, print_args, print_eval, init_seeds, file_date
from utils.torch_utils import select_device

ROOT = Path(__file__).resolve().parents[0]
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
GLOBAL_RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))


def main(args):
    if GLOBAL_RANK in {-1, 0}:
        LOGGER.add(args.exp_dir / f"{file_date()}.log", level="DEBUG")
        print_args(
            args=args, exclude_keys=("exp_dir", "save_dir", "q_config"), logger=LOGGER
        )

    device = select_device(device=args.device, batch_size=args.batch_size)
    assert device.type != "cpu", "can not support CPU mode."

    init_seeds(args.seed + GLOBAL_RANK, deterministic=True)

    ############### Model Definition ###############
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
    ppl_tasks = ["wikitext2", "ptb", "c4"]
    results = evaluator.eval_ppl(
        model=model, tokenizer_path=args.model, datasets=ppl_tasks, seq_len=2024
    )
    qa_tasks = [
        "lambada",
        "hellaswag",
        "winogrande",
        "piqa",
        "truthfulqa",
        "openbookqa",
        "boolq",
        "arc_easy",
        "arc_challenge",
    ]
    results.update(
        evaluator.eval_QA(model=model, tasks=qa_tasks, batch_size=args.batch_size)
    )
    print_eval(results, logger=LOGGER)


if __name__ == "__main__":
    main(args=build_parser(ROOT))
