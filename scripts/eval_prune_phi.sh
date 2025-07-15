#!/bin/bash

# bf16 model
python examples/phi.py \
    --model d:\\models\\phi-1.5 \
    --exp phi2-prune-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
python examples/phi.py \
    --model d:\\models\\phi-1.5 \
    --exp phi2-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --task ppl \
    --device 0

##############################################################

# bf16 model
python examples/phi.py \
    --model d:\\models\\phi-2 \
    --exp phi2-prune-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/phi.py \
    --model d:\\models\\phi-2 \
    --exp phi2-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --task ppl \
    --device 0
