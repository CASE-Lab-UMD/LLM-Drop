import json
import sys
import os

import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM

from src.llmtuner.train.quantization.AutoAWQ.awq import AutoAWQForCausalLM
from src.llmtuner.train.quantization.AutoAWQ.awq.models.deepseek_moe.configuration_deepseek import DeepseekConfig
from src.llmtuner.train.quantization.AutoAWQ.awq.models.deepseek_moe.modeling_deepseek import DeepseekModel, DeepseekForCausalLM

AutoConfig.register("deepseek", DeepseekConfig)
AutoModel.register(DeepseekConfig, DeepseekModel)
AutoModelForCausalLM.register(DeepseekConfig, DeepseekForCausalLM)

model_path = sys.argv[1]
quant_path = sys.argv[2]
bits = sys.argv[3]

if "deepseek" in model_path.lower():
    q_group_size = 64
else:
    q_group_size = 128

modules_to_not_convert = ["self_attn"]
quant_config = {
                "zero_point": True, 
                "q_group_size": q_group_size, 
                "w_bit": int(bits), 
                "version": "GEMM", 
                # "modules_to_not_convert": modules_to_not_convert,
                }

print(f"quant_config: {quant_config}")
# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=False, 
        trust_remote_code=True,
        )
except:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        use_fast=True, 
        trust_remote_code=True,
        )

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
f = open(os.path.join(quant_path, "quantize_config.json"), 'w')
config_to_save = json.dumps(quant_config, indent=2, sort_keys=True)
f.write(config_to_save)
f.close()
print(f'Model is quantized and saved at "{quant_path}"')