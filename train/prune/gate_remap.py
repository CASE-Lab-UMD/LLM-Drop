import sys
import torch
from accelerate import Accelerator
from tqdm import tqdm

from llmtuner.train.prune.utils import prepare_calibration_input, print_gpu_memory, find_moe_gates
from llmtuner.train.prune.wrapper import GateRemapWrapper


@torch.no_grad()
def gate_remap(model, model_pruned, dataloader, accelerator: Accelerator, num_samples):
    """
    Remap the gate network to refine expert selection.
    $W = (X^T @ X)^{-1} @ X^T @ Y = X^{-1} @ Y$ (Moore-Penrose pseudo-inverse)
    """
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    unwrapped_model_pruned = accelerator.unwrap_model(model_pruned)  # 🔍 unwrap model first
    unwrapped_model_pruned.config.use_cache = False
    layers_pruned = unwrapped_model_pruned.model.layers

    # 🔍 store the parameters in CPU
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍
    inputs_pruned, outputs_pruned, attention_mask_pruned, position_ids_pruned = prepare_calibration_input(unwrapped_model_pruned, dataloader, num_samples)  # 🔍

    accelerator.print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Remapping gates...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        layer_pruned = layers_pruned[i]

        subset = find_moe_gates(layer)  # 🔍 Find gates for remapping
        subset_pruned = find_moe_gates(layer_pruned)  # 🔍 Find gates for remapping

        accelerator.print("subset", subset)
        accelerator.print("subset_pruned", subset_pruned)

        # Wrap layers
        subsets = {}
        subsets_pruned = {}
        for name in subset:
            subsets[name] = GateRemapWrapper(subset[name], record_input=True, record_output=True)
            subsets_pruned[name] = GateRemapWrapper(subset_pruned[name], record_input=True, record_output=True)

        def add_batch(name, is_pruned):
            def hidden_states_hook(_, input, output):
                subsets[name].add_batch(input[0].data, output.data)

            def hidden_states_pruned_hook(_, input, output):
                subsets_pruned[name].add_batch(input[0].data, output.data)

            if not is_pruned:
                return hidden_states_hook
            else:
                return hidden_states_pruned_hook

        # Add Hooks
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name, is_pruned=False)))
            handles.append(subset_pruned[name].register_forward_hook(add_batch(name, is_pruned=True)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            outputs_pruned[j] = layer_pruned(inputs_pruned[j], attention_mask=attention_mask_pruned[j], position_ids=position_ids_pruned[j])[0]
        for h in handles:
            h.remove()

        # 🔍 remap the gates
        if i > 0:  # we skip layer 0 whose inputs are intact
            for name in subset:
                module_state_dict_name = f"model.layers.{i}.{name.replace('._fsdp_wrapped_module', '')}"  # 🔍
                accelerator.print(f"Aggregating {module_state_dict_name}")

                # aggregate sample hidden states
                original_dense_gate_outputs = torch.cat(subsets[name].outputs, dim=0)  # (total_token_num, num_experts)
                sparse_gate_inputs = torch.cat(subsets_pruned[name].inputs, dim=0)  # (total_token_num, hidden_dim)
                sparse_gate_outputs = torch.cat(subsets_pruned[name].outputs, dim=0)  # (total_token_num, num_experts)

                total_token_num_each_device = original_dense_gate_outputs.shape[0]

                if total_token_num_each_device * accelerator.num_processes <= 128 * 2048:  # can handle it with 80G GPU memory
                    # use gathered hidden_states ro calculate the inverse
                    original_dense_gate_outputs = accelerator.gather(original_dense_gate_outputs)  # 🔍 all gather across devices
                    accelerator.print(f"original_gate_outputs: {original_dense_gate_outputs.shape}")
                    sparse_gate_outputs = accelerator.gather(sparse_gate_outputs)  # 🔍 all gather across devices
                    accelerator.print(f"Former Relative L2 Loss: {torch.pow(sparse_gate_outputs - original_dense_gate_outputs, exponent=2).mean()}")

                    sparse_gate_outputs.cpu()
                    torch.cuda.empty_cache()

                    sparse_gate_inputs = accelerator.gather(sparse_gate_inputs)  # 🔍 all gather across devices
                    remapped_weights = torch.linalg.lstsq(sparse_gate_inputs, original_dense_gate_outputs).solution
                    # remapped_weights = torch.pinverse(sparse_gate_inputs) @ original_dense_gate_outputs
                    accelerator.print(f"Latter Relative L2 Loss: {torch.pow(sparse_gate_inputs @ remapped_weights - original_dense_gate_outputs, exponent=2).mean()}")
                else:
                    # calculate the inverse on each device, and then average the remapped_weights over devices
                    accelerator.print(f"original_gate_outputs: {original_dense_gate_outputs.shape}")
                    accelerator.print(f"Former Relative L2 Loss: {torch.pow(sparse_gate_outputs - original_dense_gate_outputs, exponent=2).mean()}")

                    sparse_gate_outputs.cpu()
                    torch.cuda.empty_cache()

                    remapped_weights = torch.linalg.lstsq(sparse_gate_inputs, original_dense_gate_outputs).solution
                    accelerator.print("remapped_weights", remapped_weights)
                    # remapped_weights = torch.pinverse(sparse_gate_inputs) @ original_dense_gate_outputs
                    remapped_weights = accelerator.reduce(remapped_weights, reduction="mean")  # 🔍 all reduce across devices
                    # HERE IS WHERE THE ERROR OCCURS
                    accelerator.print(f"Latter Relative L2 Loss: {torch.pow(sparse_gate_inputs @ remapped_weights - original_dense_gate_outputs, exponent=2).mean()}")

                # 🔍 update the state dict
                # 🔍 the weights would not change if directly applying them
                update_state_dict[module_state_dict_name + ".weight"] = remapped_weights.t().bfloat16().cpu()

        # Update inputs & outputs
        inputs, outputs = outputs, inputs
        inputs_pruned, outputs_pruned = outputs_pruned, inputs_pruned

    accelerator.print("Remapping done!")
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict
