
model=meta-llama/Llama-2-7b-chat-hf
quantized_model_dir=./results_quantization
bits=4
seed=0
num_samples=128
# calibration_template=llama-2
calibration_template=mistral
calibration_template=default

abbreviation=${model##*/}-GPTQ-${bits}bits/checkpoint

python gptq_auto.py --pretrained_model_dir $model --quantized_model_dir $quantized_model_dir/$abbreviation/checkpoint \
                    --bits $bits --save_and_reload --desc_act \
                    --seed $seed --num_samples $num_samples \
                    --calibration-template $calibration_template
