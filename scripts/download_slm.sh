#!/bin/bash

# opt
huggingface-cli.exe download facebook/opt-125m --local-dir ./opt-125m
huggingface-cli.exe download facebook/opt-350m --local-dir ./opt-350m
huggingface-cli.exe download facebook/opt-1.3b --local-dir ./opt-1.3b
huggingface-cli.exe download facebook/opt-2.7b --local-dir ./opt-2.7b

# bloom
huggingface-cli.exe download bigscience/bloom-560m --local-dir ./bloom-560m
huggingface-cli.exe download bigscience/bloom-1b1 --local-dir ./bloom-1.1b
huggingface-cli.exe download bigscience/bloom-1b7 --local-dir ./bloom-1.7b
huggingface-cli.exe download bigscience/bloom-3b --local-dir ./bloom-3b

# gemma
huggingface-cli.exe download google/gemma-2b-it --local-dir ./gemma-2b-it
huggingface-cli.exe download google/gemma-2-2b-it --local-dir ./gemma-2-2b-it
huggingface-cli.exe download google/gemma-3-1b-it --local-dir ./gemma-3-1b-it

# phi
huggingface-cli.exe download microsoft/phi-1_5 --local-dir ./phi-1.5
huggingface-cli.exe download microsoft/phi-2 --local-dir ./phi-2

# llama3
huggingface-cli.exe download meta-llama/Llama-3.2-1B-Instruct --local-dir ./llama-3.2-1b-it
huggingface-cli.exe download meta-llama/Llama-3.2-3B-Instruct --local-dir ./llama-3.2-3b-it
