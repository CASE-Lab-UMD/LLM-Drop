#!/usr/bin/bash

port="21804"
GPUs="0,1,2,3,4,5,6,7"

# Taking mistralai/Mistral-7B-v0.1 as an example.
model_names=("mistral") # The model to be compressed.
drop_modules=("mlp" "attn" "block") # The modules to be dropped.
drop_nums=("4" "8") # The number of dropped modules.

tasks=("boolq" "rte" "openbookqa" "piqa" "mmlu" "winogrande" "gsm8k" "hellaswag" "arc_challenge")
num_fewshots=("0" "0" "0" "0" "5" "5" "5" "10" "25")

for model_name in "${model_names[@]}"
do
    # Download the model to a local directory. 
    git lfs install
    git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
    mv Mistral-7B-v0.1 ./"$model_name"_model

    for drop_module in "${drop_modules[@]}"
    do
        for drop_num in "${drop_nums[@]}"
        do
            cfg_path=./"$model_name"_drop"$drop_num"_"$drop_module"/config.json # PATH to the corresponding config.json file.
            cp -f "$cfg_path" ./"$model_name"_model/config.json # Replace the original config.json file.
            cp ./"$model_name"_drop"$drop_num"_"$drop_module"/*.py ./"$model_name"_model/ # Build the configuration and modeling files for remote code.
            echo "Eval the config of:"
            echo $cfg_path

            num_tasks=${#tasks[@]}
            for ((i=0; i<$num_tasks; i++)); do
                CUDA_VISIBLE_DEVICES=$GPUs accelerate launch --main_process_port $port  -m lm_eval \
                    --model hf \
                    --model_args pretrained=./${model_name}_model,trust_remote_code=True,dtype="bfloat16" \
                    --tasks ${tasks[$i]} \
                    --num_fewshot ${num_fewshots[$i]} \
                    --batch_size 1 \
                    --output_path ./${num_fewshots[$i]}shot_${tasks[$i]}_"$model_name"_drop"$drop_num"_"$drop_module".json >> output_"$model_name"_drop"$drop_num"_"$drop_module".out
            done
        done
    done
done