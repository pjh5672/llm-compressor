#!/bin/bash

# bf16 model
python examples/gemma3.py \
    --model d:\\models\\gemma-3-1b-it \
    --exp gemma3-prune-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
python examples/gemma3.py \
    --model d:\\models\\gemma-3-1b-it \
    --exp gemma3-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --task ppl \
    --device 0

###########################################################

# bf16 model
python examples/gemma3.py \
    --model d:\\models\\gemma-3-4b-it \
    --exp gemma3-prune-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
python examples/gemma3.py \
    --model d:\\models\\gemma-3-4b-it \
    --exp gemma3-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --task ppl \
    --device 0
