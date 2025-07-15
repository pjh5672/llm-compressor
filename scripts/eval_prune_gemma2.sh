#!/bin/bash

# bf16 model
python examples/gemma2.py \
    --model d:\\models\\gemma-2-2b-it \
    --exp gemma2-prune-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
python examples/gemma2.py \
    --model d:\\models\\gemma-2-2b-it \
    --exp gemma2-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --task ppl \
    --device 0
