import logging
import math
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from argparse import Namespace
from copy import deepcopy
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from global_utils.io import create_dir
from llmtuner.train.prune.utils import prepare_calibration_input, print_gpu_memory
from llmtuner.train.prune.wrapper import InputStatesRecordWrapper
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM

logger = logging.getLogger(__name__)


@no_grad()
def get_block_similarities(model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
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
        inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍

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

        for i in tqdm(range(len(layers)), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            layer = layers[i]
            wrapped_layer = InputStatesRecordWrapper(layer, record_output=i == len(layers) - 1)  # 🔍 Wrap layer
            wrapped_layers.append(wrapped_layer)

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

        all_hidden_states = []
        for i in tqdm(range(len(layers)), desc="Concatenating hidden states...", disable=not accelerator.is_main_process):
            all_hidden_states.append(torch.cat(wrapped_layers[i].hidden_states, dim=0).to(device))  # (total_token_num, hidden_size)
        all_hidden_states.append(torch.cat(wrapped_layers[-1].output_hidden_states, dim=0).to(device))
        accelerator.print(f'Total {len(all_hidden_states)} hidden states concatenated.')

        for i in tqdm(range(len(all_hidden_states)), desc="Calculating similarities...", disable=not accelerator.is_main_process):
            for j in range(i + 1, len(all_hidden_states)):
                packed_hidden_states_layer_i = all_hidden_states[i]
                packed_hidden_states_layer_j = all_hidden_states[j]
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


def consecutive_block_dropping(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune blocks in a consecutive order.
    E.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [0, 1, 7, 8, 9]
    """
    drop_n = args.drop_n

    similarities = get_block_similarities(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)
    similarities_drop_n = similarities[:, drop_n].view(-1)
    max_similarity, begin_layer_id = torch.max(similarities_drop_n, dim=0)
    accelerator.print(f"similarities_drop_n: {similarities_drop_n}")
    # max_similarity, begin_layer_id = torch.max(similarities, dim=0)  # 🔍 shape(layer_num)
    accelerator.print(f"max_similarity: {max_similarity}, begin_layer_id: {begin_layer_id}")

    # max_similarity = max_similarity[drop_n - 1].item()
    # begin_layer_id = begin_layer_id[drop_n - 1].item()

    end_layer_id = begin_layer_id + drop_n
    dropped_layer_list = [i for i in range(begin_layer_id, end_layer_id)]

    accelerator.print(f"Dropped layer: {dropped_layer_list}, max_similarity: {max_similarity}")
    return dropped_layer_list


def discrete_block_dropping(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune blocks in a discrete order.
    E.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [0, 2, 6, 8, 9]
    """
    drop_n = args.drop_n

    similarities = get_block_similarities(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)
    similarities_drop_1 = similarities[:, 0].view(-1)
    accelerator.print(f"similarities_drop_1: {similarities_drop_1}")
    sorted_similarities, sorted_layer_id = torch.sort(similarities_drop_1, dim=0, descending=True)

    dropped_layer_list = sorted_layer_id[:drop_n].tolist()
    accelerator.print(f"Dropped layer: {dropped_layer_list}, similarities: {sorted_similarities[:drop_n].tolist()}")
    return dropped_layer_list


def post_block_drop(prune_model_save_path, model, tokenizer, layer_id_mapping, accelerator):
    # get state dict
    state_dict = model.state_dict()
    accelerator.print(f"layer_id_mapping: {layer_id_mapping}")

    # 🔍 update state dict for saving
    if accelerator.is_main_process:
        save_state_dict = {}
        for state_name in sorted(list(state_dict.keys())):
            for old_layer_id, new_layer_id in layer_id_mapping.items():
                if f"layers.{old_layer_id}." in state_name:  # convert old ids to new ones
                    save_state_dict[state_name.replace(f"layers.{old_layer_id}", f"layers.{new_layer_id}")] = state_dict[state_name]
                    accelerator.print(state_name, "-->", state_name.replace(f"layers.{old_layer_id}", f"layers.{new_layer_id}"))
                    break
                elif f"layers." not in state_name:  # copy other states
                    save_state_dict[state_name] = state_dict[state_name]
                    accelerator.print(state_name, "-->", state_name)
                    break

        accelerator.print("Keys in save_state_dict:")
        for key in save_state_dict.keys():
            accelerator.print(key)

        # 🔍 initialize a new model and save
        accelerator.print("Initializing the new model...")
        new_config = deepcopy(model.config)
        new_config.num_hidden_layers = len(layer_id_mapping)
        accelerator.print(new_config)
        new_model = type(model)(new_config)
        new_model.load_state_dict(save_state_dict, strict=True)
        new_model.bfloat16()
        accelerator.print("new_model", new_model)
        accelerator.print("Saving...")
        new_model.save_pretrained(prune_model_save_path)
        tokenizer.save_pretrained(prune_model_save_path)

    accelerator.wait_for_everyone()
    accelerator.print(f"Model saved to {prune_model_save_path}")
