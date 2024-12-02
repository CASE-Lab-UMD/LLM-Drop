import logging
import math
import os
import sys
from copy import deepcopy
import shutil
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from .io import create_dir
from .utils import prepare_calibration_input, print_gpu_memory, auto_map, CUSTOM_FILE
from .wrapper import HiddenStatesRecordWrapper

logger = logging.getLogger(__name__)


def get_block_similarities(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
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
        num_layers = unwrapped_model.config.num_hidden_layers
        # 🔍 Initialize the similarities.
        # Row: each layer
        # Column: similarity to the next n layer
        # Example: [ [0.5],  [0.5],  [0.5],  [0.5],  [0.5],  [0.5]]  # shape(6, 1)
        similarities = torch.full((len(layers), 1), -math.inf, device=device)

        accelerator.print('Starting ...')
        dtype = torch.float32

        for i in tqdm(range(num_layers), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]

            # Wrap layer
            wrapped_layer = HiddenStatesRecordWrapper(layer, record_input=True, record_output=True)  # 🔍 Wrap layer

            # Forward hook for recording hidden states
            def record_states_hook(_, input, output):
                wrapped_layer.record(input[0].data, output[0].data)

            # Get states
            handle = layer.register_forward_hook(record_states_hook)
            for j in range(num_samples):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            handle.remove()

            # Update inputs & outputs
            inputs, outputs = outputs, inputs
            print_gpu_memory(accelerator)

            input_hidden_states = torch.cat(wrapped_layer.input_hidden_states, dim=0).to(dtype).to(device)
            output_hidden_states = torch.cat(wrapped_layer.output_hidden_states, dim=0).to(dtype).to(device)
            cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)  # (total_token_num)
            cos_sim = cos_sim.mean()
            cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices
            similarities[i, 0] = cos_sim
            layer.to("cpu")  # 🔍

        # Save to the cache file
        if cache_file is not None:
            if accelerator.is_main_process:
                create_dir(os.path.dirname(cache_file))
                torch.save(similarities.clone().cpu(), cache_file)
                print(f"Saving cached similarities to {cache_file}")
            accelerator.wait_for_everyone()

    accelerator.print("similarities\n", similarities)

    return similarities

@no_grad()
def get_block_similarities_consecutive(model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
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
        # 🔍 Initialize the similarities.
        # Row: each layer
        # Column: similarity to the next n layer
        # Example: [[ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5],
        #           [ 0.5,  0.5,  0.5,  0.5,  0.5, -inf],
        #           [ 0.5,  0.5,  0.5,  0.5, -inf, -inf],
        #           [ 0.5,  0.5,  0.5, -inf, -inf, -inf],
        #           [ 0.5,  0.5, -inf, -inf, -inf, -inf],
        #           [ 0.5, -inf, -inf, -inf, -inf, -inf]]  # shape(6, 6)
        similarities = torch.full((len(layers), len(layers)), -math.inf, device=device)
        accelerator.print('Starting ...')
        wrapped_layers = []

        for i in tqdm(range(num_layers), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]

            # Wrap layer
            wrapped_layer = HiddenStatesRecordWrapper(layer, record_input=True, record_output=(i == len(layers) - 1))  # 🔍 Wrap layer
            wrapped_layers.append(wrapped_layer)

            # Forward hook for recording hidden states
            def record_states_hook(_, input, output):
                wrapped_layer.record(input[0].data, output[0].data)

            # Get states
            handle = layer.register_forward_hook(record_states_hook)
            for j in range(num_samples):
                if getattr(unwrapped_model.config, "model_type", None) == "llama":
                    outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j], cache_position=cache_position[j])[0]
                else:
                    outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            handle.remove()

            # Update inputs & outputs
            inputs, outputs = outputs, inputs
            print_gpu_memory(accelerator)

        dtype = torch.float32
        all_hidden_states = []
        for i in tqdm(range(len(layers)), desc="Concatenating hidden states...", disable=not accelerator.is_main_process):
            all_hidden_states.append(torch.cat(wrapped_layers[i].input_hidden_states, dim=0).to(dtype))  # (total_token_num, hidden_size)
        all_hidden_states.append(torch.cat(wrapped_layers[-1].output_hidden_states, dim=0).to(dtype))
        accelerator.print(f'Total {len(all_hidden_states)} hidden states concatenated.')

        for i in tqdm(range(len(all_hidden_states)), desc="Calculating similarities...", disable=not accelerator.is_main_process):
            for j in range(i + 1, len(all_hidden_states)):
                packed_hidden_states_layer_i = all_hidden_states[i].to(device)
                packed_hidden_states_layer_j = all_hidden_states[j].to(device)
                index_gap = j - i

                cos_sim = F.cosine_similarity(packed_hidden_states_layer_i, packed_hidden_states_layer_j, dim=-1)  # (total_token_num)
                cos_sim = cos_sim.mean()
                cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices

                similarities[i, index_gap - 1] = cos_sim

        # Save to the cache file
        if cache_file is not None:
            if accelerator.is_main_process:
                create_dir(os.path.dirname(cache_file))
                torch.save(similarities.clone().cpu(), cache_file)
                print(f"Saving cached similarities to {cache_file}")
            accelerator.wait_for_everyone()

    accelerator.print("similarities\n", similarities)

    return similarities


def max_with_tolerance(similarities: torch.tensor, tolerance: float):
    max_value, _ = torch.max(similarities, dim=0)
    close_indices = torch.where(torch.abs(similarities - max_value) < tolerance)[0]
    begin_layer_id = close_indices[0]

    return max_value, begin_layer_id


def get_top_k(similarities, k, tolerance):
    dropped_layer_list = []
    dropped_sim_list = []
    for _ in range(k):
        max_value, max_index = max_with_tolerance(similarities, tolerance=tolerance)
        dropped_layer_list.append(max_index.item())
        dropped_sim_list.append(max_value.item())
        similarities[max_index] = 0
    return dropped_sim_list, dropped_layer_list

def consecutive_block_dropping(args, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune blocks in a consecutive order.
    E.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [0, 1, 7, 8, 9]
    """
    drop_n = args.drop_n

    similarities = get_block_similarities_consecutive(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)
    similarities_drop_n = similarities[:, drop_n].view(-1)
    max_similarity, begin_layer_id = torch.max(similarities_drop_n, dim=0)
    accelerator.print(f"similarities_drop_n: {similarities_drop_n}")
    accelerator.print(f"max_similarity: {max_similarity}, begin_layer_id: {begin_layer_id}")

    end_layer_id = begin_layer_id + drop_n
    dropped_layer_list = [i for i in range(begin_layer_id, end_layer_id)]

    accelerator.print(f"Dropped layer: {dropped_layer_list}, max_similarity: {max_similarity}")
    return dropped_layer_list


def discrete_block_dropping(args, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune blocks in a discrete order.
    E.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [0, 2, 6, 8, 9]
    """
    drop_n = args.drop_n

    similarities = get_block_similarities(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)

    similarities_drop_1 = similarities[:, 0].view(-1)
    sorted_similarities, sorted_layer_id = torch.sort(similarities_drop_1, dim=0, descending=True)
    accelerator.print(f"similarities_drop_1: {similarities_drop_1}")

    dropped_layer_list = sorted_layer_id[:drop_n].tolist()
    accelerator.print(f"Dropped layer: {dropped_layer_list}, similarities: {sorted_similarities[:drop_n].tolist()}")
    return dropped_layer_list



def post_block_drop(prune_model_save_path, model, tokenizer, reserved_layer_list, accelerator: Accelerator, only_update_config=False):
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first

    if accelerator.is_main_process:
        out_cfg = deepcopy(unwrapped_model.config)
        model_type = getattr(unwrapped_model.config, "model_type", None)

        if model_type in auto_map:
            out_cfg.auto_map = auto_map[model_type]
        else:
            raise ValueError("Unsupported model type!")

        dropped_attn_list = dropped_mlp_list = list(set(list(range(out_cfg.num_hidden_layers))) - set(reserved_layer_list))
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