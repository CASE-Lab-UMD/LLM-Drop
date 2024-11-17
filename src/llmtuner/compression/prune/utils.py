import torch
from torch import nn as nn, cuda
import os

def print_gpu_memory(accelerator):
    if accelerator.is_local_main_process:  # 🔍
        for i in range(cuda.device_count()):
            used_memory = cuda.memory_allocated(0) // 1024 ** 2
            print(f"GPU {i} Used Memory: {used_memory}MB")


def print_gpu_memory_device():
    device = cuda.current_device()
    used_memory = cuda.memory_allocated(device) // 1024 ** 2
    print(f"GPU {device} Used Memory: {used_memory}MB")


def find_modules(module, layers=[], name='') -> dict:
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_modules(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def find_linears(module) -> dict:
    # 🔍 find only the expert weights
    res = find_modules(module, [nn.Linear])
    return res


@torch.no_grad()
def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_modules(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


@torch.no_grad()
def check_sparsity_from_state_dict(state_dict):
    """
    🔍 This function has been rewritten to calculate sparsity from "state_dict".
    """
    # Get corresponding names for each layer
    layer_params = {}
    for name in sorted(list(state_dict.keys())):
        if "layers" in name:
            layer_id = int(name.split(".")[2])
            if layer_id not in layer_params:
                layer_params[layer_id] = [name]
            else:
                layer_params[layer_id].append(name)
    layer_num = max(list(layer_params.keys())) + 1

    # Calculate sparsity
    count = 0
    total_params = 0
    for i in range(layer_num):
        sub_count = 0
        sub_params = 0
        for name in layer_params[i]:
            count += (state_dict[name] == 0).sum().item()
            total_params += state_dict[name].numel()

            sub_count += (state_dict[name] == 0).sum().item()
            sub_params += state_dict[name].numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    return float(count) / total_params


@torch.no_grad()
def prepare_calibration_input(model, dataloader, num_samples=16):
    layers = model.model.layers

    cache = {'inputs': [], 'attention_mask': [], "position_ids": [], "position_ids": [], "cache_position": []}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.self_attn = None

        def forward(self, input, **kwargs):
            # print(input.shape)
            cache['inputs'].append(input)
            cache['attention_mask'].append(kwargs['attention_mask'])
            cache['position_ids'].append(kwargs['position_ids'])
            cache['cache_position'].append(kwargs['cache_position'] if 'cache_position' in kwargs else None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for index, batch in enumerate(dataloader):
        if index >= num_samples:  # 🔍 limit the number of samples in each device, batch_size must be 1
            break
        try:
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    outputs = [None] * len(cache['inputs'])
    return cache['inputs'], outputs, cache['attention_mask'], cache['position_ids'], cache['cache_position']



auto_map = {
    "llama": {
                "AutoConfig": "configuration_dropped_llama.LlamaConfig",
                "AutoModelForCausalLM": "modeling_dropped_llama.LlamaForCausalLM"
            }, 
    "mistral": {
                "AutoConfig": "configuration_dropped_mistral.MistralConfig",
                "AutoModelForCausalLM": "modeling_dropped_mistral.MistralForCausalLM"
            },
    "deepseek":
                {
                "AutoConfig": "configuration_deepseek.DeepseekConfig",
                "AutoModelForCausalLM": "modeling_dropped_deepseek.DeepseekForCausalLM"
                },
    "gemma2":
                {
                "AutoConfig": "configuration_dropped_gemma2.Gemma2Config",
                "AutoModelForCausalLM": "modeling_dropped_gemma2.Gemma2ForCausalLM"
                },
    "baichuan":
                {
                "AutoConfig": "configuration_dropped_baichuan.BaichuanConfig",
                "AutoModelForCausalLM": "modeling_dropped_baichuan.BaichuanForCausalLM"
                }
}

CUSTOM_FILE ={
    "llama": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_llama.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_llama.py")
    },
    "mistral": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_mistral.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_mistral.py")
    },
    "deepseek": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_deepseek.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_deepseek.py")
    }, 
    "gemma2": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_gemma2.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_gemma2.py")
    }, 
    "baichuan": {
        "config": os.path.join(os.path.dirname(__file__), "models/configuration_dropped_baichuan.py"),
        "model": os.path.join(os.path.dirname(__file__), "models/modeling_dropped_baichuan.py")
    }
}
