#!/bin/bash

# bf16 model
python examples/llama.py \
    --model d:\\models\\llama-3.2-1b-it \
    --exp llama-prune-eval \
    --seq-len 512 \
    --tasks ppl \
    --device 0
    
python examples/llama.py \
    --model d:\\models\\llama-3.2-1b-it \
    --exp llama-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --tasks ppl \
    --device 0

####################################################################

# bf16 model
python examples/llama.py \
    --model d:\\models\\llama-3.2-3b-it \
    --exp llama-prune-eval \
    --seq-len 512 \
    --tasks ppl \
    --device 0
    
python examples/llama.py \
    --model d:\\models\\llama-3.2-3b-it \
    --exp llama-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --tasks ppl \
    --device 0
