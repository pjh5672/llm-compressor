python llm_compressor/models/llama.py --model test --exp llama-spin --quantize --quant-method rtn --weight int4-g[-1]-rw --act-in int8-g[-1]-rw
python llm_compressor/models/llama.py --model test --exp llama-spin --quantize --quant-method spinquant-had --weight int4-g[-1]-rw --act-in int8-g[-1]-rw

