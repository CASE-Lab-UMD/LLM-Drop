#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

model_path="########PATH_TO_HUGGING_FACE_CHECKPOINT#########"
quant_path="########PATH_TO_SAVE_THE_QUANTIZED_MODEL########"
bits=4

python "${ROOT_DIR}/src/llmtuner/compression/quantization/AutoAWQ/quantize.py" \
  $model_path \
  $quant_path \
  $bits
