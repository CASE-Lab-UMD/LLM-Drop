import sys
import torch
from accelerate import Accelerator
from tqdm import tqdm

from .utils import find_moe_expert_linears, prepare_calibration_input, print_gpu_memory
from .wrapper import WandaWrapper, SparseGPTWrapper


# print("transformers", transformers)


@torch.no_grad()
def prune_magnitude(args, model, accelerator, prune_n=0, prune_m=0):
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Pruning layers..."):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_moe_expert_linears(layer)  # 🔍 Find layers to prune

        # Prune
        layer.to(device)  # 🔍

        for name in subset:
            module_state_dict_name = f"model.layers.{i}.{name}"
            print(f"Pruning module {module_state_dict_name}")
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            # print(f"W_metric: {W_metric}")
            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten())[0][int(W.numel() * args.sparsity_ratio)]
                W_mask = (W_metric <= thresh)

            # 🔍 update the state dict
            # 🔍 the weights would not change if directly updating them using "W[W_mask] = 0"
            update_state_dict[module_state_dict_name + ".weight"] = (W * W_mask).bfloat16().cpu()

        layer.to("cpu")  # 🔍

    print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict


@torch.no_grad()
def prune_wanda_moe(args, model, dataloader, accelerator: Accelerator, num_samples, prune_n=0, prune_m=0):
    """
    :param num_samples: samples on each device, calculated as "num_samples = n_calibration_samples // num_processes"
    """
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍
    inputs_for_scores = [input.clone() for input in inputs]
    outputs_for_scores = [None for _ in range(len(outputs))]

    accelerator.print('Starting ...')
    ######################################
    pre_defined_sparsity = True
    # pre_defined_sparsity = False  # 是否只借助score，为每个专家分配固定的稀疏率。如果为否，则按照之前的实现，对合并后的"All_experts_metric"进行sort
    ######################################

    expert_scores, expert_numels = {}, {}
    # 🔍 TODO compute sparse ratios for experts. 
    for i in tqdm(range(len(layers)), desc="Pruning layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_moe_expert_linears(layer)  # 🔍 Find layers to prune
        experts_subset = [name for name in subset if "experts" in name]
        separate_weights = ["w1", "w2", "w3"]
        w1 = [name for name in experts_subset if "w1" in name] if "w1" in separate_weights else []
        w2 = [name for name in experts_subset if "w2" in name] if "w2" in separate_weights else []
        w3 = [name for name in experts_subset if "w3" in name] if "w3" in separate_weights else []

        # Wrap layers
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WandaWrapper(subset[name], layer_name=name, multiply_score=not pre_defined_sparsity, p=1)  # 🔍

        # Forward hook for recording row importance
        def add_scores(name):
            def scores_hook(_, input, output):
                wrapped_layers[name].add_scores(input[1].data)  # 🔍 only feed routing scores.

            return scores_hook

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_scores(name)))
        for j in range(num_samples):
            outputs_for_scores[j] = layer(inputs_for_scores[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        # Get scores
        for weights in [w1, w2, w3]:
            if len(weights) == 0:
                continue
            for ei in range(len(weights)):
                name = weights[ei]
                module_state_dict_name = f"model.layers.{i}.{name}"
                this_expert_score = wrapped_layers[name].score_memery
                this_expert_score = accelerator.reduce(this_expert_score, reduction="sum")  # 🔍 all reduce across devices
                expert_scores[module_state_dict_name] = (this_expert_score)
                expert_numels[module_state_dict_name] = (wrapped_layers[name].numel())

        # Update inputs & outputs
        inputs_for_scores, outputs_for_scores = outputs_for_scores, inputs_for_scores

    inputs_for_scores, outputs_for_scores = None, None  # clear cache

    # 🔍 TODO normalize the scores
    all_scores = torch.cat(list(expert_scores.values()), dim=0)
    all_numels = torch.tensor(list(expert_numels.values()), device=all_scores.device)
    accelerator.print(f"all_scores: {all_scores}")
    accelerator.print(f"all_numels: {all_numels}")

    # inverse_all_scores = all_scores.sum() / all_scores  # 🔍 bigger scores -> smaller sparsity
    # accelerator.print(f"inverse_all_scores: {inverse_all_scores}")
    # expert_sparsity = (inverse_expert_scores / inverse_expert_scores.sum() * len(layers) * unwrapped_model.config.num_local_experts * args.sparsity_ratio).tolist()

    normalized_scores = all_scores / all_scores.sum()
    N_select = all_numels.sum() * (1 - args.sparsity_ratio)
    expert_sparsity = 1 - (normalized_scores * N_select) / all_numels
    expert_sparsity = torch.clamp(expert_sparsity, max=1.0, min=0.)

    for i in range(len(expert_scores)):
        key = list(expert_scores.keys())[i]
        expert_scores[key] = expert_sparsity[i]
    accelerator.print(f"expert_scores: {expert_scores}")
    accelerator.print(f"expert_sparsity: {expert_sparsity}")

    # 🔍 TODO Prune
    for i in tqdm(range(len(layers)), desc="Pruning layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_moe_expert_linears(layer)  # 🔍 Find layers to prune

        #####################################
        separate = True
        # separate = False

        # separate_weights = ["w2"]  # gate
        # separate_weights = ["w2", "w3"]  # gate up
        separate_weights = ["w1", "w2", "w3"]  # down gate up
        #####################################

        if separate:
            experts_subset = [name for name in subset if "experts" in name]
            w1 = [name for name in experts_subset if "w1" in name] if "w1" in separate_weights else []
            w2 = [name for name in experts_subset if "w2" in name] if "w2" in separate_weights else []
            w3 = [name for name in experts_subset if "w3" in name] if "w3" in separate_weights else []
            vanilla_subset = [name for name in subset if name not in w1 + w2 + w3]
        else:
            vanilla_subset = subset
            w1 = []
            w2 = []
            w3 = []
        # accelerator.print(f"w1, w2, w3: {w1, w2, w3}")
        # accelerator.print(f"vanilla_subset: {vanilla_subset}")

        # #####################################
        # pre_defined_sparsity = True
        # # pre_defined_sparsity = False  # 是否只借助score，为每个专家分配固定的稀疏率。如果为否，则按照之前的实现，对合并后的"All_experts_metric"进行sort
        # #####################################

        # Wrap layers
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WandaWrapper(subset[name], layer_name=name, multiply_score=not pre_defined_sparsity, p=1)  # 🔍

        # Forward hook for recording row importance
        def add_batch(name):
            def hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output.data)

            def moe_hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output.data, input[1].data)  # 🔍 input[1] is routing scores.

            if 'experts' in name:
                return moe_hook
            else:
                return hook

            # accelerator.print(f'subset: {subset}')

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        # 🔍 Prune non-moe weights
        # for name in vanilla_subset:
        #     module_state_dict_name = f"model.layers.{i}.{name}"
        #     accelerator.print(f"Pruning module {module_state_dict_name}")
        #     W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
        #     W_metric = accelerator.reduce(W_metric, reduction="sum")  # 🔍 all reduce across devices
        #     W_mask = torch.zeros_like(W_metric)  # initialize a mask to be all 0
        #     # accelerator.print(f"W_metric: {W_metric}")

        #     if prune_n != 0:
        #         # structured n:m sparsity
        #         for ii in range(W_metric.shape[1]):
        #             if ii % prune_m == 0:
        #                 tmp = W_metric[:, ii:(ii + prune_m)].float()
        #                 W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
        #     else:
        #         sort_res = torch.sort(W_metric, dim=-1, stable=True)

        #         if args.use_variant:
        #             # wanda variant
        #             def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
        #                 thres_cumsum = sum_before * alpha
        #                 sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
        #                 thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
        #                 W_mask = (W_metric <= thres)
        #                 cur_sparsity = (W_mask == True).sum() / W_mask.numel()
        #                 return W_mask, cur_sparsity

        #             tmp_metric = torch.cumsum(sort_res[0], dim=1)
        #             sum_before = W_metric.sum(dim=1)

        #             alpha = 0.4
        #             alpha_hist = [0., 0.8]
        #             W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
        #             while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
        #                 if cur_sparsity > args.sparsity_ratio:
        #                     alpha_new = (alpha + alpha_hist[0]) / 2.0
        #                     alpha_hist[1] = alpha
        #                 else:
        #                     alpha_new = (alpha + alpha_hist[1]) / 2.0
        #                     alpha_hist[0] = alpha

        #                 alpha = alpha_new
        #                 W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
        #             accelerator.print(f"Alpha found {alpha} sparsity {cur_sparsity:.6f}")
        #         else:
        #             # unstructured pruning
        #             indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
        #             W_mask.scatter_(1, indices, True)

        #     # 🔍 update the state dict
        #     # 🔍 the weights would not change if directly updating them using "subset[name].weight.data[W_mask] = 0"
        #     update_state_dict[module_state_dict_name + ".weight"] = (subset[name].weight * (torch.ones_like(W_mask) - W_mask)).bfloat16().cpu()

        #####################################
        # column_wise = True # 是否按照wanda图1右侧所示，按列做稀疏，原文称这样做"在LLM上效果更好，但在CV上更差"
        column_wise = False  # 实测出来对MoE，直接做Weight-Wise稀疏的loss更低一点，也就是设置成False
        #####################################

        # 🔍 prune moe keys.
        for weights in [w1, w2, w3]:
            if len(weights) == 0:
                continue
            # accelerator.print(f"weights: {weights}")

            # if pre_defined_sparsity:
            # # 🔍 distribute sparsity
            # expert_scores = []
            # for ei in range(len(weights)):
            #     name = weights[ei]
            #     this_expert_score = wrapped_layers[name].score_memery
            #     this_expert_score = accelerator.reduce(this_expert_score, reduction="sum")  # 🔍 all reduce across devices
            #     expert_scores.append(this_expert_score)
            # expert_scores = torch.cat(expert_scores, dim=0)  # shape(expert_num)
            # inverse_expert_scores = expert_scores.sum() / expert_scores  # 🔍 bigger scores -> smaller sparsity
            # expert_sparsity = (inverse_expert_scores / inverse_expert_scores.sum() * unwrapped_model.config.num_local_experts * args.sparsity_ratio).tolist()
            # accelerator.print(expert_sparsity)

            for ei in range(len(weights)):
                name = weights[ei]
                module_state_dict_name = f"model.layers.{i}.{name}"

                if expert_scores[module_state_dict_name] <= 0.:
                    continue

                accelerator.print(f"Pruning module {module_state_dict_name}")
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_metric = accelerator.reduce(W_metric, reduction="sum")  # 🔍 all reduce across devices
                W_mask = torch.zeros_like(W_metric)  # initialize a mask to be all 0

                if prune_n != 0:
                    raise NotImplementedError
                else:
                    if args.use_variant:
                        raise NotImplementedError
                    else:
                        # unstructured pruning
                        sort_res = torch.sort(W_metric, dim=-1, stable=True)
                        # indices = sort_res[1][:, :int(W_metric.shape[1] * expert_sparsity[ei])]  # 🔍 use specified sparsity
                        indices = sort_res[1][:, :int(W_metric.shape[1] * expert_scores[module_state_dict_name])]
                        W_mask.scatter_(1, indices, True)

                # 🔍 update the state dict
                # 🔍 the weights would not change if directly updating them using "subset[name].weight.data[W_mask] = 0"
                accelerator.print(f"{module_state_dict_name}: {W_mask.float().mean()}")
                update_state_dict[module_state_dict_name + ".weight"] = (subset[name].weight * (torch.ones_like(W_mask) - W_mask)).bfloat16().cpu()

            # else:
            #     All_experts_metric = torch.zeros((len(weights),) + subset[weights[0]].weight.data.size(), device=device)
            #     All_mask = torch.zeros((len(weights),) + subset[weights[0]].weight.data.size(), device=device)

            #     for ei in range(len(weights)):
            #         name = weights[ei]
            #         module_state_dict_name = f"model.layers.{i}.{name}"
            #         accelerator.print(f"Pruning module {module_state_dict_name}")

            #         W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            #         # W_metric = torch.abs(subset[name].weight.data)
            #         # W_metric = torch.ones_like(subset[name].weight.data, device=subset[name].weight.data.device) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            #         W_metric = accelerator.reduce(W_metric, reduction="sum")  # 🔍 all reduce across devices
            #         All_experts_metric[ei, :, :] = W_metric

            #     accelerator.print("All_experts_metric", All_experts_metric.shape)
            #     experts, din, dout = All_experts_metric.size()
            #     All_experts_metric = All_experts_metric.transpose(0, 1).reshape(din, experts * dout)
            #     All_mask = All_mask.transpose(0, 1).reshape(din, experts * dout)

            #     if prune_n != 0:
            #         # structured n:m sparsity
            #         if column_wise:
            #             # TODO: some semi-structure, some no prune, some all prune
            #             raise NotImplementedError
            #     else:
            #         if args.use_variant:
            #             # wanda variant
            #             raise NotImplementedError
            #         else:
            #             # unstructured pruning
            #             if column_wise:
            #                 sort_res = torch.sort(All_experts_metric, dim=-1, stable=True)
            #                 indices = sort_res[1][:, :int(All_experts_metric.shape[-1] * args.sparsity_ratio)]
            #                 All_mask.scatter_(-1, indices, True)
            #                 All_mask = All_mask.reshape(din, experts, dout).transpose(0, 1)
            #             else:
            #                 sort_res = torch.sort(All_experts_metric.flatten(), stable=True)
            #                 indices = sort_res[1][:int(All_experts_metric.numel() * args.sparsity_ratio)]
            #                 All_mask = All_mask.flatten()
            #                 All_mask.scatter_(-1, indices, True)
            #                 All_mask = All_mask.reshape(din, experts, dout).transpose(0, 1)

            #     for ei in range(len(weights)):
            #         name = weights[ei]
            #         module_state_dict_name = f"model.layers.{i}.{name}"

            #         accelerator.print(f"{module_state_dict_name}: {All_mask[ei].float().mean()}")
            #         update_state_dict[module_state_dict_name + ".weight"] = (subset[name].weight
            #                                                                  * (torch.ones_like(All_mask[ei]) - All_mask[ei])).bfloat16().cpu()

        # Update inputs & outputs
        inputs, outputs = outputs, inputs

    accelerator.print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict


@torch.no_grad()
def prune_wanda(args, model, dataloader, accelerator: Accelerator, num_samples, prune_n=0, prune_m=0):
    """
    :param num_samples: samples on each device, calculated as "num_samples = n_calibration_samples // num_processes"
    """
    sparsity_type = getattr(args, "sparsity_type", "unstructured")
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍

    accelerator.print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Pruning layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset_experts = find_moe_expert_linears(layer)  # 🔍 Find layers to prune

        # Wrap layers
        wrapped_layers = {}
        for name in subset_experts:
            wrapped_layers[name] = WandaWrapper(subset_experts[name], layer_name=name, multiply_score=False, p=1)  # 🔍

        # Forward hook for recording row importance
        def add_batch_linear(name):
            def hook(_, input, output):
                wrapped_layers[name].add_batch_no_score(input[0].data, output.data)

            return hook

        def add_batch_experts(name):
            def hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output.data, input[1].data if input[1] is not None else None)  # 🔍 input[1] is routing scores.

            return hook

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset_experts[name].register_forward_hook(add_batch_experts(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        # 🔍 Prune
        for name in subset_experts:  # 🔍semi-structured
            module_state_dict_name = f"model.layers.{i}.{name}"
            accelerator.print(f"Pruning module {module_state_dict_name}")
            W = wrapped_layers[name].weight.data.to(device)  # 👆 use the captured weights
            W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_metric = accelerator.reduce(W_metric, reduction="sum")  # 🔍 all reduce across devices
            W_mask = torch.zeros_like(W_metric)  # initialize a mask to be all 0

            if prune_n != 0:
                # 🔍semi-structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            # elif sparsity_type == "structured": # 🔍structured pruning, but may be applied to another dimention. 
            #     # for ii in range(W_metric.shape[1]):
            #         # if ii % prune_m == 0:
            #     if "w2" in name: 
            #         # prune_m = W_metric.size()[-1]
            #         # prune_n = prune_m * args.sparsity_ratio
            #         # tmp = W_metric[:, :prune_m].float()
            #         # W_mask.scatter_(1, torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

            #         prune_m = W_metric.size()[0]
            #         prune_n = prune_m * args.sparsity_ratio
            #         tmp = W_metric[:prune_m, :].float()
            #         W_mask.scatter_(0, torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

            else:
                # 🔍unstructured
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant
                    def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
                        thres_cumsum = sum_before * alpha
                        sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
                        thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
                        W_mask = (W_metric <= thres)
                        cur_sparsity = (W_mask == True).sum() / W_mask.numel()
                        return W_mask, cur_sparsity

                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    accelerator.print(f"Alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            # 🔍 update the state dict
            # 🔍 the weights would not change if directly updating them using "W.data[W_mask] = 0"
            update_state_dict[module_state_dict_name + ".weight"] = (W * (torch.ones_like(W_mask) - W_mask)).bfloat16().cpu()

        # Update inputs & outputs
        inputs, outputs = outputs, inputs

    accelerator.print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict


@torch.no_grad()
def prune_sparsegpt(args, model, dataloader, accelerator: Accelerator, num_samples, prune_n=0, prune_m=0, blocksize=128, percdamp=0.01):
    """
        SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
        :param num_samples: samples on each device, calculated as "num_samples = n_calibration_samples // num_processes"
    """
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍

    accelerator.print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Pruning layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_moe_expert_linears(layer)  # 🔍 Find layers to prune

        # Wrap layers
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = SparseGPTWrapper(subset[name])

        def add_batch(name):
            def hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output.data)

            return hook

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
        for h in handles:
            h.remove()

        # Prune
        for name in wrapped_layers:
            module_state_dict_name = f"model.layers.{i}.{name}"
            accelerator.print(f"Pruning module {module_state_dict_name}")

            W = wrapped_layers[name].weight.data.to(device).float()  # 👆 use the captured weights
            H = wrapped_layers[name].H
            H = accelerator.reduce(H, reduction="mean")  # 🔍 all reduce across devices
            # gpts[name].H = None
            # torch.cuda.empty_cache()

            dead = (torch.diag(H) == 0)
            H[dead, dead] = 1
            W[:, dead] = 0

            Losses = torch.zeros(wrapped_layers[name].rows, device=wrapped_layers[name].device)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(wrapped_layers[name].columns, device=wrapped_layers[name].device)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            mask = None

            # formally begin
            for i1 in range(0, wrapped_layers[name].columns, blocksize):
                i2 = min(i1 + blocksize, wrapped_layers[name].columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if prune_n == 0:
                    if mask is not None:
                        mask1 = mask[:, i1:i2]
                    else:
                        tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * args.sparsity_ratio)]
                        mask1 = (tmp <= thresh)
                else:
                    mask1 = (torch.zeros_like(W1) == 1)

                for j in range(count):
                    w = W1[:, j]
                    d = Hinv1[j, j]

                    if prune_n != 0 and j % prune_m == 0:
                        tmp = W1[:, j:(j + prune_m)] ** 2 / (torch.diag(Hinv1)[j:(j + prune_m)].reshape((1, -1))) ** 2
                        mask1.scatter_(1, j + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                    q = w.clone()
                    q[mask1[:, j]] = 0

                    Q1[:, j] = q
                    Losses1[:, j] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, j:] -= err1.unsqueeze(1).matmul(Hinv1[j, j:].unsqueeze(0))
                    Err1[:, j] = err1

                W[:, i1:i2] = Q1
                Losses += torch.sum(Losses1, 1) / 2
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # 🔍 update the state dict
            # 🔍 the weights would not change if directly applying them
            update_state_dict[module_state_dict_name + ".weight"] = W.bfloat16().cpu()
            # subset[name].weight.data = W.reshape(subset[name].weight.shape).to(subset[name].weight.data.dtype)

        # Update inputs & outputs
        inputs, outputs = outputs, inputs

    accelerator.print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict


@torch.no_grad()
def prune_template(args, model, dataloader, accelerator: Accelerator, num_samples, prune_n=0, prune_m=0):
    """Template for pruning methods"""
    raise NotImplementedError("Please copy this function and implement the full method.")

    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}

    accelerator.print("Getting features...")
    inputs, outputs, attention_mask, position_ids = prepare_calibration_input(unwrapped_model, dataloader, num_samples)  # 🔍

    accelerator.print('Starting ...')
    for i in tqdm(range(len(layers)), desc="Pruning layers...", disable=not accelerator.is_main_process):
        sys.stderr.flush()
        torch.cuda.empty_cache()
        print_gpu_memory(accelerator)
        layer = layers[i]
        subset = find_moe_expert_linears(layer)  # 🔍 Find layers to prune

        # Wrap layers
        wrapped_layers = {}
        for name in subset:
            # TODO
            pass

        # Forward hook for recording row importance
        def add_batch(name):
            def hook(_, input, output):
                wrapped_layers[name].add_batch(input[0].data, output.data)

            return hook

        # Get importance
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(num_samples):
            outputs[j] = layer(inputs[j], attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        # Prune
        for name in subset:
            module_state_dict_name = f"model.layers.{i}.{name}"  # 🔍
            accelerator.print(f"Pruning module {module_state_dict_name}")

            # TODO

            xxxxxx = None

            # 🔍 update the state dict
            # 🔍 the weights would not change if directly applying them
            update_state_dict[module_state_dict_name + ".weight"] = xxxxxx.bfloat16().cpu()  # TODO

    accelerator.print("Pruning done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    # 🔍 return the state dict
    return update_state_dict
