#!/bin/bash

# bf16 model
python examples/opt.py \
    --model d:\\models\\opt-125m \
    --exp opt-quant-eval \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/opt.py \
    --model d:\\models\\opt-125m \
    --exp opt-quant-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/opt.py \
    --model d:\\models\\opt-125m \
    --exp opt-quant-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/opt.py \
    --model d:\\models\\opt-125m \
    --exp opt-quant-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/opt.py \
    --model d:\\models\\opt-125m \
    --exp opt-quant-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0


########################################################


# bf16 model
python examples/opt.py \
    --model d:\\models\\opt-350m \
    --exp opt-quant-eval \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/opt.py \
    --model d:\\models\\opt-350m \
    --exp opt-quant-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/opt.py \
    --model d:\\models\\opt-350m \
    --exp opt-quant-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/opt.py \
    --model d:\\models\\opt-350m \
    --exp opt-quant-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/opt.py \
    --model d:\\models\\opt-350m \
    --exp opt-quant-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0


########################################################


# bf16 model
python examples/opt.py \
    --model d:\\models\\opt-1.3b \
    --exp opt-quant-eval \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0
    
# W4A8 G128 symmetric RTN quant.
python examples/opt.py \
    --model d:\\models\\opt-1.3b \
    --exp opt-quant-eval \
    --quantize \
    --quant-method rtn \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

# W4A8 G128 symmetric GPTQ quant.
python examples/opt.py \
    --model d:\\models\\opt-1.3b \
    --exp opt-quant-eval \
    --quantize \
    --quant-method gptq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/opt.py \
    --model d:\\models\\opt-1.3b \
    --exp opt-quant-eval \
    --quantize \
    --quant-method awq \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0

# W4A8 G128 symmetric AWQ quant.
python examples/opt.py \
    --model d:\\models\\opt-1.3b \
    --exp opt-quant-eval \
    --quantize \
    --quant-method awq_plus \
    --weight int4-g[128]-rw \
    --act-in int8-g[128]-rw \
    --seq-len 512 \
    --task ppl,mmlu \
    --device 0