#!/usr/bin/bash
port="21304"
GPUs="0,1,2,3"

dataset="c4_val"
prune_data_type="pt"
n_calibration_samples=128
seq_len=2048

prune_method="layer_drop"
layer_drop_method="discrete"
target_layer="all"

drop_n=1
num_epochs=8

model_name=mistral-base
model_name_or_path=mistralai/Mistral-7B-v0.1

for ((epoch=1; epoch<=num_epochs; epoch++)) do
  layer_drop_method="discrete"
  folder_name="Iterative-epoch${epoch}-${model_name}-${prune_method}-${target_layer}-${layer_drop_method}-drop${drop_n}PerEpoch"
  similarity_cache_file="../results_prune/cache/Iterative-epoch${epoch}-${model_name}-drop_${target_layer}-${dataset}-${n_calibration_samples}samples.pt"
  echo ${folder_name}
  echo ${model_name_or_path}
  output_dir=./results_prune/Iterative/${folder_name}
  prune_model_save_path=${output_dir}/checkpoint

  CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port \
    src/compress.py \
    --stage prune \
    --model_name_or_path ${model_name_or_path} \
    --dataset ${dataset} \
    --dataset_dir ./src/llmtuner/data \
    --split "train" \
    --prune_data_type ${prune_data_type} \
    --cutoff_len ${seq_len} \
    --layer_drop_norm True \
    --target_layer ${target_layer} \
    --output_dir ${output_dir} \
    --logging_steps 10 \
    --bf16 \
    --n_calibration_samples ${n_calibration_samples} \
    --prune_method ${prune_method} \
    --layer_drop_method ${layer_drop_method} \
    --drop_n ${drop_n} \
    --similarity_cache_file ${similarity_cache_file} \
    --prune_model_save_path ${prune_model_save_path}

  # Save the converted the model without DeepSpeed
  layer_drop_method="post_dropping"
  # set only_update_config to True to save the disk memory
  only_update_config=False

  python src/compress.py \
    --stage prune \
    --model_name_or_path ${model_name_or_path} \
    --dataset ${dataset} \
    --dataset_dir ./src/llmtuner/data \
    --split "train" \
    --only_update_config $only_update_config \
    --layer_drop_norm True \
    --target_layer ${target_layer} \
    --prune_data_type ${prune_data_type} \
    --cutoff_len ${seq_len} \
    --output_dir ${output_dir} \
    --logging_steps 10 \
    --bf16 \
    --n_calibration_samples ${n_calibration_samples} \
    --prune_method ${prune_method} \
    --layer_drop_method ${layer_drop_method} \
    --drop_n ${drop_n} \
    --similarity_cache_file ${similarity_cache_file} \
    --prune_model_save_path ${prune_model_save_path}
  model_name_or_path=$prune_model_save_path
done