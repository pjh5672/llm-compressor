#!/bin/bash

# sh scripts/eval_quant_bloom.sh
# sh scripts/eval_quant_llama.sh
# sh scripts/eval_quant_opt.sh
sh scripts/eval_quant_phi.sh
sh scripts/eval_quant_qwen2.sh
sh scripts/eval_quant_qwen3.sh

sh scripts/eval_prune_bloom.sh
sh scripts/eval_prune_llama.sh
sh scripts/eval_prune_opt.sh
sh scripts/eval_prune_phi.sh
sh scripts/eval_prune_qwen2.sh
sh scripts/eval_prune_qwen3.sh
