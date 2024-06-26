import json
import sys
import os

from transformers import AutoTokenizer
from llmtuner.compression.quantization.AutoAWQ.awq import AutoAWQForCausalLM


model_path = sys.argv[1]
quant_path = sys.argv[2]
bits = sys.argv[3]

quant_config = {
                "zero_point": True, 
                "q_group_size": q_group_size, 
                "w_bit": int(bits), 
                "version": "GEMM", 
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