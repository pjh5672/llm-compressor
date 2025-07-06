from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from llm_compressor.utils.args import build_parser
from llm_compressor.evaluation.eval import LMEvaluator
from llm_compressor.models.llama import CompressLlamaForCausalLM


############### Init Arguments ###############
ROOT = Path(__file__).resolve().parents[1]
args, device = build_parser(ROOT)

############### Model Definition ###############
model_path = "d:\\models\\llama-3.2-1b-it"

config = AutoConfig.from_pretrained(model_path)
setattr(config, "head_dim", 4)
setattr(config, "hidden_size", 32)
setattr(config, "intermediate_size", 16)
setattr(config, "num_attention_heads", 4)
setattr(config, "num_key_value_heads", 4)
setattr(config, "num_hidden_layers", 1)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = CompressLlamaForCausalLM(
    config=config,
)
model.save_pretrained("tiny_llama")