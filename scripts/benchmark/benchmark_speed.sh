#!/usr/bin/bash

model_path="########PATH_TO_HUGGING_FACE_CHECKPOINT########"
save_file="########PATH_TO_SAVE_THE_RESULTS########/speed.csv"
model_type="normal" # normal or quantized

python src/benchmark_speed.py \
  --model_path $model_path \
  --model_type ${model_type} \
  --save_file ${save_file} \
  --pretrained