#!/bin/bash

# bf16 model
python examples/phi.py \
    --model d:\\models\\phi-2 \
    --exp-name phi2-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
# W4A8 per-row symmetric RTN quant.
python examples/phi.py \
    --model d:\\models\\phi-2 \
    --exp-name phi2-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[-1]-rw \
    --act-in int8-g[-1]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 per-row symmetric GPTQ quant.
python examples/phi.py \
    --model d:\\models\\phi-2 \
    --exp-name phi2-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[-1]-rw \
    --act-in int8-g[-1]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 per-row symmetric AWQ quant.
python examples/phi.py \
    --model d:\\models\\phi-2 \
    --exp-name phi2-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[-1]-rw \
    --act-in int8-g[-1]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0