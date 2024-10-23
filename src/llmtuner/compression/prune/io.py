import os
import json
import torch
from accelerate import Accelerator

from .utils import check_sparsity_from_state_dict

def load_json(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def save_json(data, file_path, indent=4, **kwargs):
    create_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf8") as f:
        f.write(f"{json.dumps(data, ensure_ascii=False, indent=indent, **kwargs)}\n")

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_update_state_dict(save_path, accelerator, update_state_dict):
    accelerator.print("Saving state dicts...")
    if accelerator.is_main_process:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(update_state_dict, os.path.join(save_path, "update_state_dict.pt"))
    accelerator.wait_for_everyone()


def save_sparse_model(prune_model_save_path, model, tokenizer, accelerator: Accelerator, update_state_dict, check_sparsity=True):
    # 🔍 check sparsity
    if check_sparsity and accelerator.is_main_process:
        accelerator.print("*" * 30)
        accelerator.print("Calculating sparsity for pruned params in the state dict...")
        sparsity_ratio = check_sparsity_from_state_dict(update_state_dict)
        accelerator.print(f"sparsity sanity check {sparsity_ratio:.4f}")
        accelerator.print("*" * 30)
    accelerator.wait_for_everyone()

    # 🔍 save
    accelerator.print("Saving models... (may take minutes)")
    if accelerator.is_main_process:
        if not os.path.exists(prune_model_save_path):
            os.makedirs(prune_model_save_path)
    accelerator.wait_for_everyone()

    # get state dict for saving
    save_state_dict = accelerator.get_state_dict(model)

    if save_state_dict is not None:
        accelerator.print(f"State dict stored in CPU on process {accelerator.process_index}")

        # update state dict
        # accelerator.print("save_state_dict", list(save_state_dict.keys()))
        # accelerator.print("update_state_dict", list(update_state_dict.keys()))

        for name, param in save_state_dict.items():
            if name in update_state_dict:
                accelerator.print(f"Updating {name} (device = {save_state_dict[name].device})")
                save_state_dict[name] = update_state_dict[name]

        # check sparsity
        if check_sparsity:
            accelerator.print("*" * 30)
            accelerator.print("Calculating sparsity for all params in the model after update...")
            sparsity_ratio = check_sparsity_from_state_dict(save_state_dict)
            accelerator.print(f"sparsity sanity check {sparsity_ratio:.4f}")
            accelerator.print("*" * 30)

        # save updated state dict
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            prune_model_save_path,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=save_state_dict,
        )
        tokenizer.save_pretrained(prune_model_save_path)

    accelerator.wait_for_everyone()
    accelerator.print(f"Model saved to {prune_model_save_path}")

def save_layer_dropped_config(target_layer, prune_model_save_path, model, tokenizer, accelerator: Accelerator, dropped_layer_list):
    # 🔍 save
    if accelerator.is_main_process:
        if not os.path.exists(prune_model_save_path):
            os.makedirs(prune_model_save_path)

        # 🔍 get reserved MLP layer ids
        unwrapped_model = accelerator.unwrap_model(model)
        if target_layer == 'all':
            reserved_layer_list = sorted(list(set(range(unwrapped_model.config.num_hidden_layers * 2)) - set(dropped_layer_list)))
        else:
            reserved_layer_list = sorted(list(set(range(unwrapped_model.config.num_hidden_layers)) - set(dropped_layer_list)))
        accelerator.print(f"Reserved layers: {reserved_layer_list}")

        # 🔍 save the config
        save_file = os.path.join(prune_model_save_path, "reserved_layers.json")
        save_json(reserved_layer_list, save_file)

    accelerator.wait_for_everyone()


def save_block_dropped_config(prune_model_save_path, model, tokenizer, accelerator: Accelerator, dropped_layer_list):
    # 🔍 save
    if accelerator.is_main_process:
        if not os.path.exists(prune_model_save_path):
            os.makedirs(prune_model_save_path)

        # 🔍 get new layer id mapping
        unwrapped_model = accelerator.unwrap_model(model)
        reserved_layer_list = sorted(list(set(range(unwrapped_model.config.num_hidden_layers)) - set(dropped_layer_list)))
        accelerator.print(f"Reserved layers: {reserved_layer_list}")
        
        save_file = os.path.join(prune_model_save_path, "reserved_layers.json")
        save_json(reserved_layer_list, save_file)

        layer_id_mapping = {}
        for new_id, reserved_old_id in enumerate(reserved_layer_list):
            layer_id_mapping[reserved_old_id] = new_id

        # 🔍 save the config
        save_mapping_file = os.path.join(prune_model_save_path, "layer_mapping.json")
        save_json(layer_id_mapping, save_mapping_file)

    accelerator.wait_for_everyone()