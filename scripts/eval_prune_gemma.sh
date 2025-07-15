#!/bin/bash

# bf16 model
python examples/gemma.py \
    --model d:\\models\\gemma-2b-it \
    --exp gemma-prune-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
python examples/gemma.py \
    --model d:\\models\\gemma-2b-it \
    --exp gemma-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --task ppl \
    --device 0
