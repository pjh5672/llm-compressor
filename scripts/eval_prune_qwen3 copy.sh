#!/bin/bash

# bf16 model
python examples/qwen3.py \
    --model d:\\models\\qwen3-1.7b \
    --exp qwen3-prune-eval \
    --seq-len 2048 \
    --task ppl \
    --device 0
    
python examples/qwen3.py \
    --model d:\\models\\qwen3-1.7b \
    --exp qwen3-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 2048 \
    --task ppl \
    --device 0

##########################################################

# bf16 model
python examples/qwen3.py \
    --model d:\\models\\qwen3-4b \
    --exp qwen3-prune-eval \
    --seq-len 2048 \
    --task ppl \
    --device 0
    
python examples/qwen3.py \
    --model d:\\models\\qwen3-4b \
    --exp qwen3-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 2048 \
    --task ppl \
    --device 0
