#!/bin/bash

# bf16 model
python examples/gemma2.py \
    --model d:\\models\\gemma-2-2b-it \
    --exp gemma2-quant-eval \
    --seq-len 512 \
    --tasks ppl \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/gemma2.py \
    --model d:\\models\\gemma-2-2b-it \
    --exp gemma2-quant-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/gemma2.py \
    --model d:\\models\\gemma-2-2b-it \
    --exp gemma2-quant-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/gemma2.py \
    --model d:\\models\\gemma-2-2b-it \
    --exp gemma2-quant-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/gemma2.py \
    --model d:\\models\\gemma-2-2b-it \
    --exp gemma2-quant-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0
