#!/usr/bin/bash

##############################################################################
cd /src/llmtuner/train/quantization/gptq-main/zeroShot
model=mistralai/Mixtral-8x7B-v0.1

python main.py $model c4 --wbits 4 --task piqa