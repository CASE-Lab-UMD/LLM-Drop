__version__ = "0.2.4"

import sys

transformers_path = "/mnt/petrelfs/dongdaize.d/workspace/compression/src"
sys.path = [transformers_path] + sys.path

from awq.models.auto import AutoAWQForCausalLM
