import logging
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralSparseMoeBlock, MixtralPreTrainedModel
from .utils import print_gpu_memory, prepare_calibration_input, find_modules
from .wrapper import MixtralExpertDropWrapper, DeepseekExpertDropWrapper
from ...model.deepseek.modeling_deepseek import DeepseekPreTrainedModel, MoEGate

logger = logging.getLogger(__name__)


def fill_missing_values_for_non_moe_layers(values: list, layer_indices: list, num_layers: int):
    filled_values = []

    for i in range(num_layers):
        if i not in layer_indices:
            filled_values.append(None)
        else:
            filled_values.append(values[layer_indices.index(i)])

    return filled_values


# 🔍 The final attempt that strictly follows the pruning pipeline
# Finally, the whole shit has been done. THANK GOD!!!!!!!!!!!!!!!!!!!!!
@torch.no_grad()
def layerwise_pruning(args: Namespace, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    """
    :param num_samples: samples on each device, calculated as "num_samples = n_calibration_samples // num_processes"
    """
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 Get MoE layer ids
    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        num_experts = unwrapped_model.config.num_local_experts
        num_layers = unwrapped_model.config.num_hidden_layers
        layer_indices = list(range(num_layers))
    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
        num_experts = unwrapped_model.config.n_routed_experts
        num_layers = unwrapped_model.config.num_hidden_layers
        layer_indices = [layer_idx for layer_idx in range(num_layers) if (unwrapped_model.config.n_routed_experts is not None and layer_idx >= unwrapped_model.config.first_k_dense_replace and layer_idx % unwrapped_model.config.moe_layer_freq == 0)]
    accelerator.print("layer_indices", layer_indices)

    # 🔍 store the pruned parameters in CPU
    # update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍

    accelerator.print('Starting ...')
    update_num_experts_list = []
    update_experts_idx = []
    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        layer_deltas = []

    for i in tqdm(layer_indices, desc="Dropping layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]

        # Find modules
        if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
            subset = find_modules(layer, [MixtralSparseMoeBlock])
        elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
            subset = find_modules(layer, [MoEGate])
        # captured_weights_subset = find_moe_expert_linears_and_gate(layer)  # 👆 Find weights to capture (here the gate & w1 & w2 & w3)
        # accelerator.print(subset)
        # accelerator.print(captured_weights_subset)

        # Wrap layers
        wrapped_layers = {}
        for name in subset:
            if isinstance(unwrapped_model, MixtralPreTrainedModel):
                wrapped_layers[name] = MixtralExpertDropWrapper(subset[name])  # 🔍
            elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
                wrapped_layers[name] = DeepseekExpertDropWrapper(subset[name])  # 🔍

        # 👆 Wrap weights to record during forward
        # (WHY: DeepSpeed will temporarily collect intact weights to GPU during forward, so we can capture them using forward hooks)
        # captured_weights_wrapped_layers = {}
        # for name in captured_weights_subset:
        #     captured_weights_wrapped_layers[name] = WeightRecordWrapper(captured_weights_subset[name], layer_name=name)

        # Forward hook for recording metrics
        def add_batch(name):
            def mixtral_hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output[1].data)  # output[1] is router_logits (before softmax)

            def deepseek_hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output[0].data, output[1].data)  # output[0] is topk ids, output[1] is topk scores (after softmax)

            if isinstance(unwrapped_model, MixtralPreTrainedModel):
                return mixtral_hook  # 🔍
            elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
                return deepseek_hook  # 🔍

        # def record_weight(name):  # 👆
        #     def hook(_, input, output):
        #         captured_weights_wrapped_layers[name].record(input, output)
        #
        #     return hook

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        # for name in captured_weights_wrapped_layers:  # 👆
        #     handles.append(captured_weights_subset[name].register_forward_hook(record_weight(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        # 🔍 Expert Drop
        for name in subset:
            module_state_dict_name = f"model.layers.{i}.{name}"
            accelerator.print(f"Dropping for {module_state_dict_name}")

            # 🔍 sort total scores
            # [IMPORTANT] all reduce across devices
            scores = wrapped_layers[name].scores
            scores = accelerator.reduce(scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.
            accelerator.print(f"layer {i} scores: {scores}")

            _, experts_to_drop = torch.topk(scores, num_experts - args.r, largest=False)
            experts_to_drop = experts_to_drop.tolist()
            accelerator.print(f"layer {i} experts_to_drop: {experts_to_drop}")
            experts_to_preserve = sorted(list(set(range(num_experts)) - set(experts_to_drop)))
            update_num_experts_list.append(len(experts_to_preserve))
            update_experts_idx.append(experts_to_preserve)

            if isinstance(unwrapped_model, MixtralPreTrainedModel):
                # delta = measure_delta_top2(scores)
                ana_list = wrapped_layers[name].ana_list
                delta = np.mean(ana_list)
                accelerator.print(f"layer {i} delta: {delta}")
                layer_deltas.append(delta)

        # 🔍 update the state dict
        # 👆 get weights from the "captured_weights_wrapped_layers"
        # update_state_dict[f"{module_state_dict_name}.gate.weight"] = captured_weights_wrapped_layers[f"{name}.gate"].weight[list(experts_to_preserve)]
        # for new_expert_id, old_expert_id in enumerate(experts_to_preserve):
        #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w1.weight"] = captured_weights_wrapped_layers[f"{name}.experts.{old_expert_id}.w1"].weight
        #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w2.weight"] = captured_weights_wrapped_layers[f"{name}.experts.{old_expert_id}.w2"].weight
        #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w3.weight"] = captured_weights_wrapped_layers[f"{name}.experts.{old_expert_id}.w3"].weight

        # update_state_dict[f"{module_state_dict_name}.gate.weight"] = wrapped_layers[name].gate[list(experts_to_preserve)].clone().bfloat16().cpu()
        # for new_expert_id, old_expert_id in enumerate(experts_to_preserve):
        #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w1.weight"] = wrapped_layers[name].w1.clone().bfloat16().cpu()
        #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w2.weight"] = wrapped_layers[name].w2.clone().bfloat16().cpu()
        #     update_state_dict[f"{module_state_dict_name}.experts.{new_expert_id}.w3.weight"] = wrapped_layers[name].w3.clone().bfloat16().cpu()

        # Update inputs & outputs
        inputs, outputs = outputs, inputs

    # 🔍 Fill in the missing values for non-MoE layers
    update_num_experts_list = fill_missing_values_for_non_moe_layers(update_num_experts_list, layer_indices, num_layers)
    update_experts_idx = fill_missing_values_for_non_moe_layers(update_experts_idx, layer_indices, num_layers)
    accelerator.print("update_num_experts_list", update_num_experts_list)
    accelerator.print("update_experts_idx", update_experts_idx)

    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        layer_deltas = fill_missing_values_for_non_moe_layers(layer_deltas, layer_indices, num_layers)
        accelerator.print("layer_deltas", layer_deltas)

    # 🔍 Update the config
    accelerator.print("Expert dropping done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        accelerator.print("Updating model config...")
        setattr(unwrapped_model.config, "num_local_experts", update_num_experts_list)
        setattr(unwrapped_model.config, "layer_experts_idx", update_experts_idx)
        setattr(unwrapped_model.config, "layer_deltas", layer_deltas)
        setattr(unwrapped_model.config, "mode", "dynamic")  # 🔍 ensure dynamic skipping.
    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
        setattr(unwrapped_model.config, "n_routed_experts", update_num_experts_list)
        setattr(unwrapped_model.config, "layer_experts_idx", update_experts_idx)

    # 🔍 return the state dict
    # return update_state_dict


@torch.no_grad()
def global_pruning(args: Namespace, model, dataloader: DataLoader, accelerator: Accelerator, num_samples: int):
    # device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 Get MoE layer ids
    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        num_experts = unwrapped_model.config.num_local_experts
        num_layers = unwrapped_model.config.num_hidden_layers
        layer_indices = list(range(num_layers))
    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
        num_experts = unwrapped_model.config.n_routed_experts
        num_layers = unwrapped_model.config.num_hidden_layers
        layer_indices = [layer_idx for layer_idx in range(num_layers) if (unwrapped_model.config.n_routed_experts is not None and layer_idx >= unwrapped_model.config.first_k_dense_replace and layer_idx % unwrapped_model.config.moe_layer_freq == 0)]
    accelerator.print("layer_indices", layer_indices)

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)

    accelerator.print('Starting ...')
    global_scores = []
    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        layer_deltas = []

    for i in tqdm(layer_indices, desc="Gathering scores...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]

        # Find modules
        if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
            subset = find_modules(layer, [MixtralSparseMoeBlock])
        elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
            subset = find_modules(layer, [MoEGate])

            # Wrap layers
        wrapped_layers = {}
        for name in subset:
            if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
                wrapped_layers[name] = MixtralExpertDropWrapper(subset[name])
            elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
                wrapped_layers[name] = DeepseekExpertDropWrapper(subset[name])

        # Forward hook for recording metrics
        def add_batch(name):
            def mixtral_hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output[1].data)  # output[1] is router_logits (before softmax)

            def deepseek_hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output[0].data, output[1].data)  # output[0] is topk ids, output[1] is topk scores (after softmax)

            if isinstance(unwrapped_model, MixtralPreTrainedModel):
                return mixtral_hook  # 🔍
            elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
                return deepseek_hook  # 🔍

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        # 🔍 Expert Drop
        for name in subset:
            module_state_dict_name = f"model.layers.{i}.{name}"
            accelerator.print(f"Dropping for {module_state_dict_name}")

            # 🔍 sort total scores
            # [IMPORTANT] all reduce across devices
            scores = wrapped_layers[name].scores
            scores = accelerator.reduce(scores, reduction="sum")  # Here we use "sum" as the number of tokens processed by each device may be different.
            global_scores.append(scores)
            accelerator.print(f"layer {i} scores: {scores}")

            if isinstance(unwrapped_model, MixtralPreTrainedModel):
                # delta = measure_delta_top2(scores)
                ana_list = wrapped_layers[name].ana_list
                delta = np.mean(ana_list)
                accelerator.print(f"layer {i} delta: {delta}")
                layer_deltas.append(delta)

        inputs, outputs = outputs, inputs

    print_gpu_memory(accelerator)
    torch.cuda.empty_cache()

    # 🔍 Cat scores
    global_scores = torch.cat(global_scores, dim=0)  # 🔍 gather the scores.
    accelerator.print(f"global_scores: {global_scores}")

    _, experts_to_drop = torch.topk(global_scores, (num_experts - args.r) * len(layer_indices), largest=False)
    experts_to_drop = sorted(experts_to_drop.tolist())
    accelerator.print(f"experts_to_drop: {experts_to_drop}")

    # 🔍 Expert Drop
    update_num_experts_list = []
    update_experts_idx = []

    for position_id, layer_id in tqdm(enumerate(layer_indices), desc='Dropping Experts...'):
        # position_id: position of the element in the list
        experts_to_preserve = sorted(list(set(range(num_experts * position_id, num_experts * (position_id + 1))) - set(experts_to_drop)))
        experts_to_preserve = [i - position_id * num_experts for i in experts_to_preserve]
        accelerator.print(f"layer {layer_id} experts_to_preserve: {experts_to_preserve}")

        update_num_experts_list.append(len(experts_to_preserve))
        update_experts_idx.append(experts_to_preserve)

    # 🔍 Fill in the missing values for non-MoE layers
    update_num_experts_list = fill_missing_values_for_non_moe_layers(update_num_experts_list, layer_indices, num_layers)
    update_experts_idx = fill_missing_values_for_non_moe_layers(update_experts_idx, layer_indices, num_layers)
    accelerator.print("update_num_experts_list", update_num_experts_list)
    accelerator.print("update_experts_idx", update_experts_idx)

    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        layer_deltas = fill_missing_values_for_non_moe_layers(layer_deltas, layer_indices, num_layers)
        accelerator.print("layer_deltas", layer_deltas)

    # 🔍 Update the config
    accelerator.print("Expert dropping done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    if isinstance(unwrapped_model, MixtralPreTrainedModel):
        accelerator.print("Updating model config...")
        setattr(unwrapped_model.config, "num_local_experts", update_num_experts_list)
        setattr(unwrapped_model.config, "layer_experts_idx", update_experts_idx)
        setattr(unwrapped_model.config, "layer_deltas", layer_deltas)
        setattr(unwrapped_model.config, "mode", "dynamic")  # 🔍 ensure dynamic skipping.
    elif isinstance(unwrapped_model, DeepseekPreTrainedModel):
        setattr(unwrapped_model.config, "n_routed_experts", update_num_experts_list)
        setattr(unwrapped_model.config, "layer_experts_idx", update_experts_idx)


@torch.no_grad()
def progressive_pruning(model, calib_loader: DataLoader, args: Namespace):
    raise NotImplementedError

    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = MixtralExpertDropWrapper(
            layer.block_sparse_moe, r=args.r)
        layer.block_sparse_moe.cache_Z = True

    for i, batch in enumerate(tqdm(calib_loader, desc='Computing Z activations on sample set...')):
        model_inputs = model.prepare_inputs_for_generation(**batch)
        outputs = model(**model_inputs)
        assert outputs is not None

    del model_inputs
    del outputs
    torch.cuda.empty_cache()

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe.cache_Z = False

    # Drop
    global_loss_history = dict()

    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Dropping layers...'):
        b = layer.block_sparse_moe

        b.cache_X = True
        with torch.inference_mode():
            for i, batch in enumerate(calib_loader):
                print(f"batch.keys(): {batch.keys()}")
                model_inputs = model.prepare_inputs_for_generation(**batch)
                outputs = model(**model_inputs)
                assert outputs is not None

        del model_inputs
        del outputs
        torch.cuda.empty_cache()
        b.cache_X = False

        loss_history = b.enumerate()
        global_loss_history[l] = loss_history

        b.prune()
        layer.block_sparse_moe = b.model

    # Prune & save
    model.num_experts = args.r
    # model.config.num_local_experts = args.r


@torch.no_grad()
def dynamic_skipping(model, calib_loader: DataLoader, args: Namespace):
    raise NotImplementedError

    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = MixtralExpertDropWrapper(
            layer.block_sparse_moe)
        layer.block_sparse_moe.cache_logits = True
        layer.block_sparse_moe.cache_X = True
        layer.block_sparse_moe.cache_Z = True

    for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...')):
        model_inputs = model.prepare_inputs_for_generation(**batch)
        outputs = model(**model_inputs)
        assert outputs is not None

    res_median = {}
    res_mean = {}

    for layer_idx in range(len(model.model.layers)):
        b = model.model.layers[layer_idx].block_sparse_moe
        b.cache_space.prepare_for_loader()
        dataloader = torch.utils.data.DataLoader(
            b.cache_space,
            batch_size=args.batch_size,
            shuffle=True,
        )
        logger.info(len(dataloader))

        ana_list = []
        for i, (router_logits, X, Z) in enumerate(dataloader):
            routing_weights = F.softmax(
                router_logits, dim=-1, dtype=torch.float).view(-1, b.model.num_experts)
            for j in range(len(routing_weights)):
                sorted_weights, sort_indices = torch.sort(
                    routing_weights[j], descending=True)
                ana_list.append(float(sorted_weights[1] / sorted_weights[0]))

        median = np.median(ana_list)
        mean = np.mean(ana_list)
        logger.info(f'layer {layer_idx} | mean: {mean}, median: {median}')
        res_median[str(layer_idx)] = median
        res_mean[str(layer_idx)] = mean

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = layer.block_sparse_moe.model

    model.config.betas = res_median
    return model, (res_median, res_mean)


@torch.no_grad()
def post_experts_drop(model, layer_experts_idx, accelerator: Accelerator):
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    layers = unwrapped_model.model.layers

    if accelerator.is_main_process:
        for layer_id, layer in tqdm(list(enumerate(layers)), desc='Dropping Experts...'):
            experts_to_preserve = layer_experts_idx[layer_id]
            accelerator.print(f"layer {layer_id} experts_to_preserve: {experts_to_preserve}")

            if experts_to_preserve is not None:
                r = len(experts_to_preserve)

                if isinstance(unwrapped_model, MixtralPreTrainedModel):  # 🔍
                    # rewrite gate.
                    new_gate_weight = layer.block_sparse_moe.gate.weight.data[experts_to_preserve]
                    layer.block_sparse_moe.gate = nn.Linear(in_features=layer.block_sparse_moe.gate.in_features, out_features=r, bias=False, device=layer.block_sparse_moe.gate.weight.device, dtype=torch.bfloat16)
                    layer.block_sparse_moe.gate.weight.data = new_gate_weight
                    # drop experts.
                    layer.block_sparse_moe.experts = nn.ModuleList([layer.block_sparse_moe.experts[i] for i in experts_to_preserve])
                    layer.num_experts = r

                elif isinstance(unwrapped_model, DeepseekPreTrainedModel):  # 🔍
                    # rewrite gate.
                    new_gate_weight = layer.mlp.gate.weight.data[experts_to_preserve]
                    layer.mlp.gate.n_routed_experts = r
                    layer.mlp.gate.top_k = min(layer.mlp.gate.top_k, r)
                    layer.mlp.gate.weight.data = new_gate_weight
                    # drop experts.
                    layer.mlp.experts = nn.ModuleList([layer.mlp.experts[i] for i in experts_to_preserve])
                    layer.mlp.num_experts_per_tok = r

    accelerator.wait_for_everyone()
    accelerator.print(f"model: {model}")

# def measure_delta_top2(scores):
#     sorted_scores, _ = torch.sort(scores, descending=True)
#     delta = sorted_scores[1] / sorted_scores[0]
#     print(f"sorted_scores:{sorted_scores}, sorted_scores[0]:{sorted_scores[0]}, sorted_scores[1]:{sorted_scores[1]}, delta:{delta}")
#     return float(delta.data)
