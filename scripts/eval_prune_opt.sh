#!/bin/bash

# bf16 model
python examples/opt.py \
    --model d:\\models\\opt-125m \
    --exp opt-prune-eval \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0
    
python examples/opt.py \
    --model d:\\models\\opt-125m \
    --exp opt-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.5 \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

########################################################

# bf16 model
python examples/opt.py \
    --model d:\\models\\opt-350m \
    --exp opt-prune-eval \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0
    
python examples/opt.py \
    --model d:\\models\\opt-350m \
    --exp opt-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.5 \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

########################################################

# bf16 model
python examples/opt.py \
    --model d:\\models\\opt-1.3b \
    --exp opt-prune-eval \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/opt.py \
    --model d:\\models\\opt-1.3b \
    --exp opt-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.5 \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0
