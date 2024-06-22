
port="21804"
GPUs="4,5,6,7"
model_names=("mistral")
drop_modules=("attn")
drop_nums=("12" "16" "20" "24" "28")



for model_name in "${model_names[@]}"
do
  # wget -P ./"$model_name"_model https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00001-of-00003.safetensors
  # wget -P ./"$model_name"_model https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00002-of-00003.safetensors
  # wget -P ./"$model_name"_model https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00003-of-00003.safetensors

  for drop_module in "${drop_modules[@]}"
  do
    for drop_num in "${drop_nums[@]}"
    do

      cfg_path=./2_"$model_name"_drop"$drop_num"_"$drop_module"/config.json
      cp -f "$cfg_path" ./"$model_name"_model/config.json
      echo "eval the config of:"
      echo $cfg_path
      CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval --model hf \
        --model_args pretrained=./${model_name}_model,trust_remote_code=True,dtype="bfloat16",use_cache=False \
        --tasks arc_challenge \
        --num_fewshot 25 \
        --batch_size 1 \
        --output_path ./25shot_arc_challenge_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".json >> output_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".out

      CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval --model hf \
        --model_args pretrained=./${model_name}_model,trust_remote_code=True,dtype="bfloat16",use_cache=False \
        --tasks boolq \
        --num_fewshot 0 \
        --batch_size 1 \
        --output_path ./0shot_boolq_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".json >> output_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".out

      CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval --model hf \
        --model_args pretrained=./${model_name}_model,trust_remote_code=True,dtype="bfloat16",use_cache=False \
        --tasks hellaswag \
        --num_fewshot 10 \
        --batch_size 1 \
        --output_path ./10shot_hellaswag_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".json >> output_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".out

      CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval --model hf \
        --model_args pretrained=./${model_name}_model,trust_remote_code=True,dtype="bfloat16",use_cache=False \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size 1 \
        --output_path ./5shot_mmlu_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".json >> output_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".out

      CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval --model hf \
        --model_args pretrained=./${model_name}_model,trust_remote_code=True,dtype="bfloat16",use_cache=False \
        --tasks openbookqa \
        --num_fewshot 0 \
        --batch_size 1 \
        --output_path ./0shot_openbookqa_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".json >> output_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".out

      CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval --model hf \
        --model_args pretrained=./${model_name}_model,trust_remote_code=True,dtype="bfloat16",use_cache=False \
        --tasks piqa \
        --num_fewshot 0 \
        --batch_size 1 \
        --output_path ./0shot_piqa_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".json >> output_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".out

      CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval --model hf \
        --model_args pretrained=./${model_name}_model,trust_remote_code=True,dtype="bfloat16",use_cache=False \
        --tasks rte \
        --num_fewshot 0 \
        --batch_size 1 \
        --output_path ./0shot_rte_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".json >> output_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".out

      CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval --model hf \
        --model_args pretrained=./${model_name}_model,trust_remote_code=True,dtype="bfloat16",use_cache=False \
        --tasks winogrande \
        --num_fewshot 5 \
        --batch_size 1 \
        --output_path ./5shot_winogrande_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".json >> output_model_idx_"$model_name"_drop"$drop_num"_"$drop_module".out


    done
  done
done

