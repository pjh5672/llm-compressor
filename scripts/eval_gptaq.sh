# #!/bin/bash

# python examples/opt.py \
#     --model d:\\models\\opt-125m \
#     --exp opt-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

# python examples/opt.py \
#     --model d:\\models\\opt-350m \
#     --exp opt-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

# python examples/opt.py \
#     --model d:\\models\\opt-1.3b \
#     --exp opt-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

# ########################################################

# python examples/bloom.py \
#     --model d:\\models\\bloom-560m \
#     --exp bloom-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

# python examples/bloom.py \
#     --model d:\\models\\bloom-1.1b \
#     --exp bloom-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

# python examples/bloom.py \
#     --model d:\\models\\bloom-1.7b \
#     --exp bloom-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

# python examples/bloom.py \
#     --model d:\\models\\bloom-3b \
#     --exp bloom-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

# ########################################################

# python examples/phi.py \
#     --model d:\\models\\phi-1.5 \
#     --exp phi-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

# python examples/phi.py \
#     --model d:\\models\\phi-2 \
#     --exp phi-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

# ########################################################

# python examples/gemma.py \
#     --model d:\\models\\gemma-2b-it \
#     --exp gemma-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

python examples/gemma2.py \
    --model d:\\models\\gemma-2-2b-it \
    --exp gemma-gptaq-eval \
    --quantize \
    --quant-method gptaq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

# python examples/gemma3.py \
#     --model d:\\models\\gemma-3-1b-it \
#     --exp gemma-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

python examples/gemma3.py \
    --model d:\\models\\gemma-3-4b-it \
    --exp gemma-gptaq-eval \
    --quantize \
    --quant-method gptaq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

########################################################

# python examples/qwen2.py \
#     --model d:\\models\\qwen2.5-0.5b-it \
#     --exp qwen-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

# python examples/qwen3.py \
#     --model d:\\models\\qwen3-1.7b \
#     --exp qwen-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

python examples/qwen3.py \
    --model d:\\models\\qwen3-4b \
    --exp qwen-gptaq-eval \
    --quantize \
    --quant-method gptaq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0

########################################################

# python examples/llama.py \
#     --model d:\\models\\llama-3.2-1b-it \
#     --exp llama-gptaq-eval \
#     --quantize \
#     --quant-method gptaq \
#     --weight int4-g[128]-rw \
#     --act-in int8-g[128]-rw \
#     --seq-len 512 \
#     --task ppl \
#     --device 0

python examples/llama.py \
    --model d:\\models\\llama-3.2-3b-it \
    --exp llama-gptaq-eval \
    --quantize \
    --quant-method gptaq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl \
    --device 0
