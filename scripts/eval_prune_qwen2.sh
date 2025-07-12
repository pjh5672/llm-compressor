#!/bin/bash

# bf16 model
python examples/qwen2.py \
    --model d:\\models\\qwen2.5-0.5b-it \
    --exp qwen2-prune-eval \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0
    
python examples/qwen2.py \
    --model d:\\models\\qwen2.5-0.5b-it \
    --exp qwen2-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.5 \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0
