#!/bin/bash

# bf16 model
python examples/llama.py \
    --model d:\\models\\llama-3.2-1b-it \
    --exp llama-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/llama.py \
    --model d:\\models\\llama-3.2-1b-it \
    --exp llama-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/llama.py \
    --model d:\\models\\llama-3.2-1b-it \
    --exp llama-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/llama.py \
    --model d:\\models\\llama-3.2-1b-it \
    --exp llama-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ+ quant.
python examples/llama.py \
    --model d:\\models\\llama-3.2-1b-it \
    --exp llama-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric SpinQuant quant.
python examples/llama.py \
    --model d:\\models\\llama-3.2-1b-it \
    --exp llama-eval \
    --quantize \
    --quant-method spinquant-had \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

####################################################################


# bf16 model
python examples/llama.py \
    --model d:\\models\\llama-3.2-3b-it \
    --exp llama-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/llama.py \
    --model d:\\models\\llama-3.2-3b-it \
    --exp llama-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/llama.py \
    --model d:\\models\\llama-3.2-3b-it \
    --exp llama-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/llama.py \
    --model d:\\models\\llama-3.2-3b-it \
    --exp llama-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ+ quant.
python examples/llama.py \
    --model d:\\models\\llama-3.2-3b-it \
    --exp llama-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric SpinQuant quant.
python examples/llama.py \
    --model d:\\models\\llama-3.2-3b-it \
    --exp llama-eval \
    --quantize \
    --quant-method spinquant-had \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0