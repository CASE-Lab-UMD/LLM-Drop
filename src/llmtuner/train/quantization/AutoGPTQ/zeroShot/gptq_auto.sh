#!/usr/bin/bash

##############################################################################
cd /src/llmtuner/train/quantization/gptq-main/zeroShot

model=mistralai/Mixtral-8x7B-v0.1
quantized_model_dir=./results_quantization
bits=4
seed=0
num_samples=128
calibration_template=default

abbreviation=${model##*/}-GPTQ-${bits}bits/checkpoint

python gptq_auto.py --pretrained_model_dir $model --quantized_model_dir $quantized_model_dir/$abbreviation/checkpoint \
                    --bits $bits --save_and_reload --desc_act \
                    --seed $seed --num_samples $num_samples \
                    --calibration-template $calibration_template
