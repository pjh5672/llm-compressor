#!/bin/bash

# bf16 model
python examples/bloom.py \
    --model d:\\models\\bloom-560m \
    --exp-name bloom-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/bloom.py \
    --model d:\\models\\bloom-560m \
    --exp-name bloom-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-560m \
    --exp-name bloom-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-560m \
    --exp-name bloom-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ+ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-560m \
    --exp-name bloom-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0


############################################

# bf16 model
python examples/bloom.py \
    --model d:\\models\\bloom-1.1b \
    --exp-name bloom-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/bloom.py \
    --model d:\\models\\bloom-1.1b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-1.1b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-1.1b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ+ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-1.1b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0


############################################

# bf16 model
python examples/bloom.py \
    --model d:\\models\\bloom-1.7b \
    --exp-name bloom-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/bloom.py \
    --model d:\\models\\bloom-1.7b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-1.7b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-1.7b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ+ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-1.7b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0


############################################

# bf16 model
python examples/bloom.py \
    --model d:\\models\\bloom-3b \
    --exp-name bloom-eval \
    --seq-len 512 \
    --task ppl \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/bloom.py \
    --model d:\\models\\bloom-3b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-3b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-3b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# W4A8 G128 symmetric AWQ+ quant.
python examples/bloom.py \
    --model d:\\models\\bloom-3b \
    --exp-name bloom-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0