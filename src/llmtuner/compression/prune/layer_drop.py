import logging
import math
import os
import sys
import shutil
from copy import deepcopy

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from .io import create_dir
from .utils import print_gpu_memory, prepare_calibration_input, auto_map, CUSTOM_FILE
from .wrapper import HiddenStatesRecordWrapper

logger = logging.getLogger(__name__)

#  🔍 compute similarity
@no_grad()
def get_layer_similarities(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, drop_norm: bool, target_layer: str, cache_file=None):
    device = accelerator.device

    if cache_file is not None and os.path.exists(cache_file):
        # use cached file
        accelerator.print(f"Loading cached model from {cache_file}")
        similarities = torch.load(cache_file, map_location=device)

    else:
        # calculate similarities
        accelerator.print(f"No cached model found. Running model on {num_samples} samples for each device.")
        unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
        unwrapped_model.config.use_cache = False
        layers = unwrapped_model.model.layers

        accelerator.print("Getting features...")
        inputs, outputs, attention_mask, position_ids, cache_position = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍

        # 🔍 Get layer ids
        num_layers = unwrapped_model.config.num_hidden_layers
        layer_indices = list(range(num_layers))

        # 🔍 Initialize the similarities.
        # Row: each layer
        # Column: similarity to the next n layer
        # Example: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # shape(6)
        similarities = torch.full((num_layers,), -math.inf, device=device)
        if hasattr(unwrapped_model.config, f'drop_{target_layer}_list'):
            skipped_layers = [idx for idx, v in enumerate(getattr(unwrapped_model.config, f'drop_{target_layer}_list', [])) if v]
        else:
            skipped_layers = []

        accelerator.print('Starting ...')
        for i in tqdm(range(num_layers), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            if i in skipped_layers:
                similarities[i] = -math.inf
                accelerator.print('Skip the dropped layer: ', i)
                continue
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]

            if i in layer_indices:
                if target_layer == 'mlp':
                    module_pre_norm = layer.post_attention_layernorm
                    module = layer.mlp
                elif target_layer == 'attn':
                    module_pre_norm = layer.input_layernorm
                    module = layer.self_attn
                elif target_layer == 'all':
                    raise ValueError("Unsupported target_layer!")
                if drop_norm:
                    wrapped_module_pre_norm = HiddenStatesRecordWrapper(module_pre_norm, record_input=True, record_output=False)  # 🔍 Wrap layer
                else:
                    wrapped_module_pre_norm = HiddenStatesRecordWrapper(module_pre_norm, record_input=False, record_output=True)  # 🔍 Wrap layer
                wrapped_module = HiddenStatesRecordWrapper(module, record_input=False, record_output=True)  # 🔍 Wrap layer

                # Forward hook for recording hidden states
                def record_module_pre_norm_states_hook(_, input, output):
                    wrapped_module_pre_norm.record(input[0].data, output[0].data)

                if target_layer == 'mlp':
                    def record_module_states_hook(_, input, output):
                        wrapped_module.record(input[0].data, output[0].data)
                elif target_layer == 'attn':
                    def record_module_states_hook(_, input, output):
                        wrapped_module.record(None, output[0].data)
                else:
                    raise ValueError("Unsupported target_layer!")
                # Get hidden states
                handles = []
                handles.append(module_pre_norm.register_forward_hook(record_module_pre_norm_states_hook))
                handles.append(module.register_forward_hook(record_module_states_hook))
                for j in range(num_samples):
                    if getattr(unwrapped_model.config, "model_type", None) == "llama":
                        outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j], cache_position=cache_position[j])[0]
                    else:
                        outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
                for handle in handles:
                    handle.remove()
                
                dtype = torch.float32

                if drop_norm:
                    input_hidden_states = torch.cat(wrapped_module_pre_norm.input_hidden_states, dim=0).to(dtype).to(device)
                    output_hidden_states = input_hidden_states + torch.cat(wrapped_module.output_hidden_states, dim=0).to(dtype).to(device)
                else:
                    input_hidden_states = torch.cat(wrapped_module_pre_norm.output_hidden_states, dim=0).to(dtype).to(device)
                    output_hidden_states = torch.cat(wrapped_module.output_hidden_states, dim=0).to(dtype).to(device)

                # 🔍 Calculate similarity (output+input due to residual connection)
                cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)  # (total_token_num)
                cos_sim = cos_sim.mean()
                cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices
                accelerator.print(f'layer {i} similarity: {cos_sim.item()}')
                similarities[i] = cos_sim
                
            else:
                for j in range(num_samples):
                    if getattr(unwrapped_model.config, "model_type", None) == "llama":
                        outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j], cache_position=cache_position[j])[0]
                    else:
                        outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]

            # Update inputs & outputs
            inputs, outputs = outputs, inputs

        # Save to the cache file
        if cache_file is not None:
            if accelerator.is_main_process:
                create_dir(os.path.dirname(cache_file))
                torch.save(similarities.clone().cpu(), cache_file)
                print(f"Saving cached similarities to {cache_file}")
            accelerator.wait_for_everyone()

    accelerator.print("similarities\n", similarities)

    return similarities

#  🔍 find indices of dropped layers
def discrete_layer_dropping(args, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune mlp layers in a discrete order.
    E.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [0, 2, 6, 8, 9]
    """
    drop_n = args.drop_n

    if args.target_layer == 'all':
        similarities_attn = get_layer_similarities(model, dataloader, accelerator, num_samples, args.layer_drop_norm, target_layer='attn', cache_file=args.similarity_cache_file.replace("all", "all_attn"))
        similarities_mlp = get_layer_similarities(model, dataloader, accelerator, num_samples, args.layer_drop_norm, target_layer='mlp', cache_file=args.similarity_cache_file.replace("all", "all_mlp"))
        similarities = torch.cat((similarities_attn, similarities_mlp), dim=0)
    else:
        similarities = get_layer_similarities(model, dataloader, accelerator, num_samples, args.layer_drop_norm, target_layer=args.target_layer, cache_file=args.similarity_cache_file)

    sorted_similarities, sorted_layer_id = torch.sort(similarities, dim=0, descending=True)

    dropped_layer_list = sorted_layer_id[:drop_n].tolist()
    accelerator.print(f"Dropped layer: {dropped_layer_list}, similarities: {sorted_similarities[:drop_n].tolist()}")
    return dropped_layer_list


def post_layers_drop(prune_model_save_path, target_layer, model, tokenizer, reserved_layer_list, accelerator: Accelerator, only_update_config=False):
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first

    if accelerator.is_main_process:
        out_cfg = deepcopy(unwrapped_model.config)
        model_type = getattr(unwrapped_model.config, "model_type", None)

        if model_type in auto_map:
            out_cfg.auto_map = auto_map[model_type]
        else:
            raise ValueError("Unsupported model type!")
        dropped_attn_list = []
        dropped_mlp_list = []
        if target_layer == 'all':
            dropped_layer_list = list(set(list(range(out_cfg.num_hidden_layers * 2))) - set(reserved_layer_list))
            for idx in dropped_layer_list:
                if idx >= out_cfg.num_hidden_layers:
                    dropped_mlp_list.append(idx - out_cfg.num_hidden_layers)
                else:
                    dropped_attn_list.append(idx)
        elif target_layer == 'attn':
            dropped_attn_list = list(set(list(range(out_cfg.num_hidden_layers))) - set(reserved_layer_list))
        elif target_layer == 'mlp':
            dropped_mlp_list = list(set(list(range(out_cfg.num_hidden_layers))) - set(reserved_layer_list))
        else:
            raise ValueError("Unsupported target_layer!")

        out_cfg.drop_mlp_list = [idx for idx, v in enumerate(getattr(unwrapped_model.config, f'drop_mlp_list', [])) if v] + dropped_mlp_list
        out_cfg.drop_attn_list = [idx for idx, v in enumerate(getattr(unwrapped_model.config, f'drop_attn_list', [])) if v] + dropped_attn_list

        accelerator.print(f"Dropped attention list: {dropped_attn_list}")
        accelerator.print(f"Dropped MLP list: {dropped_mlp_list}")

        accelerator.print("Saving...")
        shutil.copy(CUSTOM_FILE[out_cfg.model_type]["config"], prune_model_save_path)
        shutil.copy(CUSTOM_FILE[out_cfg.model_type]["model"], prune_model_save_path)
        if not only_update_config:
            model.save_pretrained(prune_model_save_path)
            tokenizer.save_pretrained(prune_model_save_path)
        out_cfg.save_pretrained(prune_model_save_path)