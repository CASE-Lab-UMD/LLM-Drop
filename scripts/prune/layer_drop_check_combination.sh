##############################################################################
port="21304"
GPUs="0,1,2,3,4,5,6,7"

dataset="c4_val"
prune_data_type="pt"
n_calibration_samples=256
seq_len=2048

prune_method="layer_drop"
layer_drop_method="discrete"
target_layer="all"

drop_n=64

# model_name=mistral-7b-base
# model_name_or_path=mistralai/Mistral-7B-v0.1

# model_name=openllamav1-7b-base
# model_name_or_path=openlm-research/open_llama_7b

# model_name=openllamav2-7b-base
# model_name_or_path=openlm-research/open_llama_7b_v2

# model_name=llama1-7b-base
# model_name_or_path=huggyllama/llama-7b

# model_name=llama2-7b-base
# model_name_or_path=meta-llama/Llama-2-7b-hf

# model_name=llama3-8b-base
# model_name_or_path=meta-llama/Meta-Llama-3-8B

# model_name=codellama-7b-base
# model_name_or_path=codellama/CodeLlama-7b-hf

model_name=llama2-13b-base
model_name_or_path=meta-llama/Llama-2-13b-hf

folder_name="${model_name}-${prune_method}_${target_layer}-${layer_drop_method}-drop${drop_n}"
similarity_cache_file="../results_prune/cache/${model_name}-${prune_method}_${target_layer}-${dataset}-${n_calibration_samples}samples.pt"

echo ${folder_name}

output_dir=./results_prune/${folder_name}
prune_model_save_path=${output_dir}/checkpoint

CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --dataset_dir ./src/llmtuner/data \
  --split "train" \
  --layer_drop_norm True \
  --target_layer ${target_layer} \
  --only_update_config True \
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


layer_drop_method="post_dropping"
# set only_update_config to True to save the disk memory
only_update_config=True 

python \
  src/train_bash.py \
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