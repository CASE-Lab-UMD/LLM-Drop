import logging
import math
import os
from argparse import Namespace

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from global_utils.io import create_dir
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from .utils import print_gpu_memory, prepare_calibration_input
from .wrapper import InputStatesRecordWrapper

logger = logging.getLogger(__name__)


@no_grad()
def get_layer_similarities(model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int, cache_file=None):
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
        # Example: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # shape(6)
        similarities = torch.full((len(layers),), -math.inf, device=device)

        accelerator.print('Starting ...')
        wrapped_mlps = []

        for i in tqdm(range(len(layers)), desc="Recording hidden states...", disable=not accelerator.is_main_process):
            layer = layers[i]
            mlp = layer.block_sparse_moe  # 🔍
            wrapped_mlp = InputStatesRecordWrapper(mlp, record_output=True)  # 🔍 Wrap layer
            wrapped_mlps.append(wrapped_mlp)

            # Forward hook for recording hidden states
            def record_states_hook(_, input, output):
                wrapped_mlp.record(input[0].data, output[0].data)

            # Get states
            handle = mlp.register_forward_hook(record_states_hook)
            for j in range(num_samples):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            handle.remove()

            # Update inputs & outputs
            inputs, outputs = outputs, inputs
            print_gpu_memory(accelerator)

        all_input_hidden_states = []
        all_output_hidden_states = []
        for i in tqdm(range(len(layers)), desc="Concatenating hidden states...", disable=not accelerator.is_main_process):
            all_input_hidden_states.append(torch.cat(wrapped_mlps[i].hidden_states, dim=0).to(device))  # (total_token_num, hidden_size)
            all_output_hidden_states.append(torch.cat(wrapped_mlps[i].output_hidden_states, dim=0).to(device))  # (total_token_num, hidden_size)
        accelerator.print(f'Total {len(all_input_hidden_states)} hidden states concatenated.')

        for i in tqdm(range(len(all_input_hidden_states)), desc="Calculating similarities...", disable=not accelerator.is_main_process):
            input_hidden_states = all_input_hidden_states[i]
            output_hidden_states = all_output_hidden_states[i]

            cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)  # (total_token_num)
            cos_sim = cos_sim.mean()
            cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices

            similarities[i] = cos_sim

        # Save to the cache file
        if cache_file is not None:
            if accelerator.is_main_process:
                create_dir(os.path.dirname(cache_file))
                torch.save(similarities.clone().cpu(), cache_file)
                print(f"Saving cached similarities to {cache_file}")
            accelerator.wait_for_everyone()

    accelerator.print("similarities\n", similarities)

    return similarities



def discrete_layer_dropping(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    🔍 Prune mlp layers in a discrete order.
    E.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> [0, 2, 6, 8, 9]
    """
    drop_n = args.drop_n

    similarities = get_layer_similarities(model, dataloader, accelerator, num_samples, cache_file=args.similarity_cache_file)
    accelerator.print(f"similarities: {similarities}")
    sorted_similarities, sorted_layer_id = torch.sort(similarities, dim=0, descending=True)

    dropped_layer_list = sorted_layer_id[:drop_n].tolist()
    accelerator.print(f"Dropped layer: {dropped_layer_list}, similarities: {sorted_similarities[:drop_n].tolist()}")
    return dropped_layer_list


def post_layers_drop(model, reserved_layer_list, accelerator: Accelerator):
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    layers = unwrapped_model.model.layers

    num_local_experts = []

    if accelerator.is_main_process:
        for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping MLPs...'):
            if layer_id in reserved_layer_list:
                num_local_experts.append(unwrapped_model.config.num_local_experts)
            else:
                layer.block_sparse_moe = None
                num_local_experts.append(0)

        unwrapped_model.config.num_local_experts = num_local_experts

    accelerator.wait_for_everyone()
    accelerator.print(f"model: {model}")

# def post_layer_drop(prune_model_save_path, model, tokenizer, layer_id_mapping, accelerator):
#     # get state dict
#     state_dict = model.state_dict()
#     accelerator.print(f"layer_id_mapping: {layer_id_mapping}")
#
#     # 🔍 update state dict for saving
#     if accelerator.is_main_process:
#         save_state_dict = {}
#         for state_name in sorted(list(state_dict.keys())):
#             for old_layer_id, new_layer_id in layer_id_mapping.items():
#                 if f"layers.{old_layer_id}." in state_name:  # convert old ids to new ones
#                     save_state_dict[state_name.replace(f"layers.{old_layer_id}", f"layers.{new_layer_id}")] = state_dict[state_name]
#                     accelerator.print(state_name, "-->", state_name.replace(f"layers.{old_layer_id}", f"layers.{new_layer_id}"))
#                     break
#                 elif f"layers." not in state_name:  # copy other states
#                     save_state_dict[state_name] = state_dict[state_name]
#                     accelerator.print(state_name, "-->", state_name)
#                     break
#
#         accelerator.print("Keys in save_state_dict:")
#         for key in save_state_dict.keys():
#             accelerator.print(key)
#
#         # 🔍 initialize a new model and save
#         accelerator.print("Initializing the new model...")
#         new_config = deepcopy(model.config)
#         new_config.num_hidden_layers = len(layer_id_mapping)
#         accelerator.print(new_config)
#         new_model = type(model)(new_config)
#         new_model.load_state_dict(save_state_dict, strict=True)
#         new_model.bfloat16()
#         accelerator.print("new_model", new_model)
#         accelerator.print("Saving...")
#         new_model.save_pretrained(prune_model_save_path)
#         tokenizer.save_pretrained(prune_model_save_path)
#
#     accelerator.wait_for_everyone()
#     accelerator.print(f"Model saved to {prune_model_save_path}")


# @torch.no_grad()
# def layer_pruning(args: Namespace, model: MixtralForCausalLM, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
#     cache_file = args.similarity_cache_file
#
#
#
#     device = accelerator.device
#     unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
#     use_cache = unwrapped_model.config.use_cache
#     unwrapped_model.config.use_cache = False
#     num_local_experts = unwrapped_model.config.num_local_experts
#     layers = unwrapped_model.model.layers
#
#
#     accelerator.print("Getting features...")
#     inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍
#     new_subset = {}
#     wrapped_layers = {}
#
#     # Wrap layers
#     accelerator.print('Starting ...')
#     for i in tqdm(range(len(layers)), desc="Dropping layers...", disable=not accelerator.is_main_process):
#         sys.stderr.flush()
#         torch.cuda.empty_cache()
#         print_gpu_memory(accelerator)
#         layer = layers[i]
#         subset = find_modules(layer, [MixtralSparseMoeBlock])  # 🔍
#
#         # Wrap MixtralSparseMoeBlock
#         name = "block_sparse_moe"
#         module_state_dict_name = f"model.layers.{i}.{name}"  # name = "block_sparse_moe"
#         wrapped_layers[module_state_dict_name] = MixtralLayerDropWrapper(subset[name])  # 🔍
#         new_subset[module_state_dict_name] = subset[name]
#
#         # Forward hook for recording metrics
#         def add_batch(name):
#             def hook(_, input, output):
#                 wrapped_layers[name].add_batch(input[0].data, output[0].data)  # output[1] is router_logits (before softmax)
#
#             return hook
#
#         # Get importance
#         handles = []
#         for name in wrapped_layers:
#             handles.append(new_subset[name].register_forward_hook(add_batch(name)))
#         for j in range(num_samples):
#             outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
#         for h in handles:
#             h.remove()
#
#         inputs, outputs = outputs, inputs
#
#     print_gpu_memory(accelerator)
#     torch.cuda.empty_cache()
#
#     # 🔍 Expert Drop
#     global_scores = []
#     update_num_local_experts_list = []
#     for module_state_dict_name in wrapped_layers:
#         accelerator.print(f"Dropping for {module_state_dict_name}")
#
#         # 🔍 sort total scores
#         # [IMPORTANT] all reduce across devices
#         hidden_states = torch.cat(wrapped_layers[module_state_dict_name].hidden_states, dim=0).to(device)
#         output_hidden_states = torch.cat(wrapped_layers[module_state_dict_name].output_hidden_states, dim=0).to(device)
#         cos_sim = F.cosine_similarity(hidden_states, output_hidden_states, dim=-1)  # (total_token_num)
#         cos_sim = cos_sim.mean()
#         # cos_sim = accelerator.reduce(cos_sim, reduction="mean")  # 🔍 All reduce across devices
#         # scores = wrapped_layers[module_state_dict_name].scores
#         accelerator.print(f"cos_sim: {cos_sim}")
#         # accelerator.print(f"scores: {scores.size()}")
#         # scores = accelerator.reduce(scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.
#         # global_scores = torch.cat((global_scores, cos_sim), dim=0) if global_scores is not None else cos_sim  # 🔍 gather the scores.
#         global_scores.append(cos_sim.data)
#
#     accelerator.print(f"global_scores: {global_scores}")
#     if cache_file is not None:
#         if accelerator.is_main_process:
#             create_dir(os.path.dirname(cache_file))
#             torch.save(global_scores.clone().cpu(), cache_file)
#             print(f"Saving cached similarities to {cache_file}")
#         accelerator.wait_for_everyone()
#
#     # sorted_scores = sorted(global_scores)
#     _, layers_to_drop = torch.topk(torch.tensor(global_scores), int(args.sparsity_ratio * len(layers)), largest=False)
#     # accelerator.print(f"global_scores: {global_scores.size()}")
#
#     layers_to_drop = sorted(layers_to_drop.tolist())
#     for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):
#         experts = num_local_experts if layer_id not in layers_to_drop else 0
#         update_num_local_experts_list.append(experts)
#
#     unwrapped_model.config.layer_experts_idx = update_num_local_experts_list
#     accelerator.print(f"layers_to_drop: {layers_to_drop}")
#     accelerator.print("Layer dropping done!")
#     unwrapped_model.config.use_cache = use_cache
#     torch.cuda.empty_cache()
