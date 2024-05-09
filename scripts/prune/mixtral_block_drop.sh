##############################################################################
dataset="c4_train"
prune_data_type="pt"

n_calibration_samples=1024
n_calibration_samples=10000
seq_len=2048

prune_method="block_drop"
block_drop_method="consecutive"
drop_n=4

model_name_or_path=meta-llama/Llama-2-7b-chat-hf
folder_name="Mixtral-${prune_method}-${block_drop_method}-drop${drop_n}"
similarity_cache_file="../results_prune/cache/block-${dataset}-${n_calibration_samples}samples.pt"

echo ${folder_name}

output_dir=./results_prune/${folder_name}
prune_model_save_path=${output_dir}/checkpoint

python \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
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

# Save the converted the model without DeepSpeed
block_drop_method="post_dropping"

python \
  src/train_bash.py \
  --stage prune \
  --model_name_or_path ${model_name_or_path} \
  --dataset ${dataset} \
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