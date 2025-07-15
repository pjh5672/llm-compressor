#!/bin/bash

# bf16 model
python examples/gemma.py \
    --model d:\\models\\gemma-2b-it \
    --exp gemma-quant-eval \
    --seq-len 2048 \
    --task ppl \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/gemma.py \
    --model d:\\models\\gemma-2b-it \
    --exp gemma-quant-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 2048 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/gemma.py \
    --model d:\\models\\gemma-2b-it \
    --exp gemma-quant-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 2048 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/gemma.py \
    --model d:\\models\\gemma-2b-it \
    --exp gemma-quant-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 2048 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/gemma.py \
    --model d:\\models\\gemma-2b-it \
    --exp gemma-quant-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 2048 \
    --task ppl \
    --device 0
