#!/bin/bash

# bf16 model
python examples/bloom.py \
    --model d:\\models\\bloom-560m \
    --exp bloom-prune-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
python examples/bloom.py \
    --model d:\\models\\bloom-560m \
    --exp bloom-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --task ppl \
    --device 0

############################################

# bf16 model
python examples/bloom.py \
    --model d:\\models\\bloom-1.1b \
    --exp bloom-prune-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
python examples/bloom.py \
    --model d:\\models\\bloom-1.1b \
    --exp bloom-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --task ppl \
    --device 0

############################################

# bf16 model
python examples/bloom.py \
    --model d:\\models\\bloom-1.7b \
    --exp bloom-prune-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
python examples/bloom.py \
    --model d:\\models\\bloom-1.7b \
    --exp bloom-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --task ppl \
    --device 0

############################################

# bf16 model
python examples/bloom.py \
    --model d:\\models\\bloom-3b \
    --exp bloom-prune-eval \
    --seq-len 512 \
    --task ppl \
    --device 0

python examples/bloom.py \
    --model d:\\models\\bloom-3b \
    --exp bloom-prune-eval \
    --prune \
    --prune-method magnitude \
    --sparsity 0.3 \
    --seq-len 512 \
    --task ppl \
    --device 0
