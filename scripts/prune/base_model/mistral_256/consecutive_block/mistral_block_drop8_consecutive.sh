##############################################################################
dataset="c4_val"
prune_data_type="pt"

n_calibration_samples=256
# n_calibration_samples=10000
seq_len=2048

prune_method="block_drop"
block_drop_method="consecutive"
drop_n=8

model_name=mistral-base
model_name_or_path=mistralai/Mistral-7B-v0.1
folder_name="${model_name}-${prune_method}-${block_drop_method}-drop${drop_n}"
similarity_cache_file="../results_prune/cache/${model_name}-block-${dataset}-${n_calibration_samples}samples.pt"

echo ${folder_name}

output_dir=./results_prune/${folder_name}
prune_model_save_path=${output_dir}/checkpoint

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 21203 \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
  --dataset_dir ./src/llmtuner/data \
  --split "train" \
  --prune_data_type ${prune_data_type} \
  --cutoff_len ${seq_len} \
  --output_dir ${output_dir} \
  --logging_steps 10 \
  --bf16 \
  --n_calibration_samples ${n_calibration_samples} \
  --prune_method ${prune_method} \
  --block_drop_method ${block_drop_method} \
  --drop_n ${drop_n} \
  --similarity_cache_file ${similarity_cache_file} \
  --prune_model_save_path ${prune_model_save_path}

# # Save the converted the model without DeepSpeed
# block_drop_method="post_dropping"

# python \
#   src/train_bash.py \
#   --stage prune \
#   --model_name_or_path ${model_name_or_path} \
#   --dataset ${dataset} \
#   --dataset_dir ./src/llmtuner/data \
#   --split "train" \
#   --prune_data_type ${prune_data_type} \
#   --cutoff_len ${seq_len} \
#   --output_dir ${output_dir} \
#   --logging_steps 10 \
#   --bf16 \
#   --n_calibration_samples ${n_calibration_samples} \
#   --prune_method ${prune_method} \
#   --block_drop_method ${block_drop_method} \
#   --drop_n ${drop_n} \
#   --similarity_cache_file ${similarity_cache_file} \
#   --prune_model_save_path ${prune_model_save_path}