#!/usr/bin/bash

model_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########"
quant_path="########PATH_TO_SAVE_THE_QUANTIZED_MODEL########"

bits=4
seed=0
num_samples=16
calibration_template=default

python AutoGPTQ/quantize.py \
  --pretrained_model_dir $model_path \
  --quantized_model_dir $quant_path \
  --bits $bits \
  --save_and_reload \
  --desc_act \
  --seed $seed \
  --num_samples $num_samples \
  --calibration-template $calibration_template \
  --trust_remote_code \
  --use_triton