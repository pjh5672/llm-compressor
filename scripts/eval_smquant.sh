#!/bin/bash

python examples/opt.py \
    --model d:\\models\\opt-125m \
    --exp opt-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

python examples/opt.py \
    --model d:\\models\\opt-350m \
    --exp opt-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

python examples/opt.py \
    --model d:\\models\\opt-1.3b \
    --exp opt-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

########################################################

python examples/bloom.py \
    --model d:\\models\\bloom-560m \
    --exp bloom-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

python examples/bloom.py \
    --model d:\\models\\bloom-1.1b \
    --exp bloom-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

python examples/bloom.py \
    --model d:\\models\\bloom-1.7b \
    --exp bloom-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

python examples/bloom.py \
    --model d:\\models\\bloom-3b \
    --exp bloom-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

########################################################

python examples/qwen2.py \
    --model d:\\models\\qwen2.5-0.5b-it \
    --exp qwen-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

python examples/qwen3.py \
    --model d:\\models\\qwen3-1.7b \
    --exp qwen-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

python examples/qwen3.py \
    --model d:\\models\\qwen3-4b \
    --exp qwen-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

########################################################

python examples/llama.py \
    --model d:\\models\\llama-3.2-1b-it \
    --exp llama-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0

python examples/llama.py \
    --model d:\\models\\llama-3.2-3b-it \
    --exp llama-smquant-eval \
    --quantize \
    --quant-method smoothquant \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --tasks ppl \
    --device 0
