## Environment
1. **lm-evaluation-harness**

   Install a new version of lm_eval

   ```shell
   git clone https://github.com/s1ghhh/lm-evaluation-harness.git
   cd lm-evaluation-harness
   git checkout offset_by_id
   
   git log
   #commit cc499b52c265853fe92305af1897eb137f3036e3 (HEAD -> offset_by_id, #origin/offset_by_id)
    #Author: s1ghhh <sghinscu@163.com>
    #Date:   Sun Apr 21 02:31:59 2024 +0000
   ```
   
   Ensure the version of the following Python libraries
   
   ```shell
   accelerate=0.27.2
   transformers=4.37.2
   lm_eval=0.4.0
   torch=2.1.2
   ```
   
   Then run the following scripts to check the result. If the result is `0.7372`, then the version is correct; otherwise, please refer to `./requirements_lm_eval.txt`

   ```shell
    CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval --model hf \
     --model_args pretrained=allenai/tulu-2-dpo-7b,trust_remote_code=True,dtype="bfloat16" \
     --tasks winogrande \
     --num_fewshot 5 \
     --batch_size 1 
   
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch -m lm_eval --model hf \
     --model_args pretrained=allenai/tulu-2-dpo-7b,trust_remote_code=True,dtype="bfloat16" \
     --tasks winogrande \
     --num_fewshot 5 \
     --batch_size 1 
   ```
   
   Then run the following scripts to check the result. If the result of `acc_norm` is `0.5836`, then the version is correct; otherwise, please refer to `./requirements_lm_eval.txt`

   ```shell
   CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=allenai/tulu-2-dpo-7b,trust_remote_code=True,dtype="bfloat16" \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --batch_size 1
   
   CUDA_VISIBLE_DEVICES=0,1 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=allenai/tulu-2-dpo-7b,trust_remote_code=True,dtype="bfloat16" \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --batch_size 1
   ```



2. **Eval the model**

   Download model weights to the corresponding location

   For `meta-llama/Llama-2-13b-chat-hf`, download to `LLM-Block-Drop/main_exp/llama2-13b/llama2-13b_model`
   
   For `mistralai/Mistral-7B-Instruct-v0.2`, download to `LLM-Block-Drop/main_exp/mistral/mistral_model`

   ```shell
     # wget -P ./mistral_model https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00001-of-00003.safetensors
     # wget -P ./mistral_model https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00002-of-00003.safetensors
     # wget -P ./mistral_model https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/model-00003-of-00003.safetensors

     # wget -P ./llama2-13b_model https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/resolve/main/model-00001-of-00003.safetensors
     # wget -P ./llama2-13b_model https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/resolve/main/model-00002-of-00003.safetensors
     # wget -P ./llama2-13b_model https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/resolve/main/model-00003-of-00003.safetensors
   ```

   Then

   ```shell
      cd LLM-Block-Drop/main_exp/llama2-13b
      bash eval_llama2-13b.sh
   
      cd LLM-Block-Drop/main_exp/mistral
      bash eval_mistral.sh
   ```
   # I remember that `nohup` couldn't directly suspend tasks and we need to use `screen` on the server of Clemson! 
