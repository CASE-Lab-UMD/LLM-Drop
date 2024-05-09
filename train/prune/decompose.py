import math
import random
import sys
import torch
from accelerate import Accelerator
from torch import nn as nn
from tqdm import tqdm
from typing import Optional

from llmtuner.train.prune.permute import get_permuted_state_dict_mixtral
from llmtuner.train.prune.utils import print_gpu_memory, find_moe_expert_linears


def svd(weight: torch.Tensor):
    matrix_dimension = len(weight.size())
    assert matrix_dimension == 2, "Only Support 2D matrix"

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    dtype = weight.dtype
    weight = weight.float()
    u_, s_, v_ = torch.linalg.svd(weight, full_matrices=False)
    u_ = u_.to(dtype)
    s_ = s_.to(dtype)
    v_ = v_.to(dtype)
    weight = weight.to(dtype)
    return u_, s_, v_


def low_rank_decomposition(
        weight: torch.Tensor,
        rank_ratio: Optional[float] = 0.1,
        parameter_ratio: Optional[float] = 0.15,
        remove_criteria: Optional[str] = 'max_eigenvalue',
        return_dict: Optional[bool] = False,
        u_=None,
        s_=None,
        v_=None,
        reduced_rank=None,
        accelerator=None,
):
    """
    Parameters
    ----------
    weight: torch.Tensor
        The matrix to decompose, of shape (H, W)
    rank_ratio: float, optional, default 0.1
        The ratio of the reduced rank to the original rank:
            rank_of_decomposed_matrix / rank_of_input_weight
    parameter_ratio: float, optional, default 0.15
        The ratio of the number of parameters of the decomposed matrix to the original matrix:
            parameter_num_of_decomposed_matrix / (H * W).
        If specify, override rank_ratio
    remove_criteria: str, optional, default 'max_eigenvalue'
        The criteria to remove the small eigenvalues, of ['max_eigenvalue', 'random', 'min_eigenvalue']
    return_dict: bool, optional, default False
        Return a dict if True, else return a tuple (L, R)
    debug: bool, optional, default False
        Print debug information if True
    """
    height, width = weight.size()

    # Use SVD to decompose a matrix
    if u_ is None or s_ is None or v_ is None:
        u_, s_, v_ = svd(weight)
    rank = torch.count_nonzero(s_)
    if reduced_rank is None:
        if parameter_ratio is not None:
            reduced_rank = math.ceil(parameter_ratio * (height * width) / (height + width))
        else:
            reduced_rank = math.ceil(rank * rank_ratio)

    if accelerator is not None:
        accelerator.print(f"nonzero rank: {rank}")

    if remove_criteria == 'max_eigenvalue':
        l_ = u_ @ (torch.sqrt(torch.diag(s_)[:, 0:reduced_rank]))
        r_ = torch.sqrt(torch.diag(s_)[0:reduced_rank, :]) @ v_
    elif remove_criteria == 'random':
        selected_index = random.choices(range(len(s_)), k=reduced_rank)
        l_ = u_ @ (torch.sqrt(torch.diag(s_)[:, selected_index]))
        r_ = torch.sqrt(torch.diag(s_)[selected_index, :]) @ v_
    elif remove_criteria == 'min_eigenvalue':
        len_s = len(s_)
        l_ = u_ @ (torch.sqrt(torch.diag(s_)[:, len_s - reduced_rank:]))
        r_ = torch.sqrt(torch.diag(s_)[len_s - reduced_rank:, :]) @ v_
    else:
        raise NameError("remove criteria not support")

    if return_dict:
        # return {"L": l_, "R": r_, "U": u_, "S": s_, "Vh": v_, 'reduced_rank': reduced_rank}
        return {"L": l_, "R": r_, 'reduced_rank': reduced_rank}
    else:
        return l_, r_


def _substitute_single_linear_weight(
        module: nn.Module,
        accelerator: Accelerator,
        parameter_ratio: float,
        has_sparse: bool,
        use_svd: bool,
        update_state_dict: dict,
        module_state_dict_name: str,
        device,
        sparse_weight: torch.Tensor = None,
        **kwargs
) -> nn.Module:
    """
    Substitute a single Linear weight with to LinearLoSparse

    Examples
    --------
    >>> linear = nn.Linear(16, 32)
    >>> linear = _substitute_single_linear_weight(linear, parameter_ratio=0.15, has_sparse=True, use_svd=True)
    Reduced Rank: 2 | Num Parameters: 96
    >>> linear
    LinearLoSparse(
      (right): Linear(in_features=16, out_features=2, bias=False)
      (left): Linear(in_features=2, out_features=32, bias=False)
      (sparse): Linear(in_features=16, out_features=32, bias=False)
    )
    """
    has_bias = module.bias is not None

    if use_svd:
        # Decompose a matrix by SVD
        if sparse_weight is None:
            weight_tensor = module.weight.data.to(device)  # 🔍 here to device
            output = low_rank_decomposition(weight_tensor, parameter_ratio=parameter_ratio, return_dict=True,
                                            accelerator=accelerator, **kwargs)
            l_, r_, reduced_rank = output['L'], output['R'], output['reduced_rank']
            s_ = weight_tensor - torch.mm(l_, r_)
        else:
            weight_tensor = module.weight.data.to(device) - sparse_weight  # 🔍 here to device
            output = low_rank_decomposition(weight_tensor, parameter_ratio=parameter_ratio, return_dict=True,
                                            accelerator=accelerator, **kwargs)
            l_, r_, reduced_rank = output['L'], output['R'], output['reduced_rank']
            s_ = sparse_weight
    else:
        height, width = module.weight.shape
        reduced_rank = math.ceil(parameter_ratio * (height * width) / (height + width))
        l_ = torch.zeros(height, reduced_rank, requires_grad=False)
        r_ = torch.zeros(reduced_rank, width, requires_grad=False)
        s_ = torch.zeros(height, width, requires_grad=False)

    # Create a nn.Module and assign decomposed weights to the parameters
    # in_features, out_features = module.in_features, module.out_features
    # module = LoSparseLinear(in_features, out_features, reduced_rank, has_bias=has_bias, has_sparse=has_sparse)
    # l_, r_, s_ = l_.to("cpu"), r_.to("cpu"), s_.to("cpu")
    # module.initialize_weight(l_, r_, s_)

    update_state_dict[module_state_dict_name + ".left.weight"] = l_.bfloat16().cpu()  # 🔍 to cpu
    update_state_dict[module_state_dict_name + ".right.weight"] = r_.bfloat16().cpu()
    if has_sparse:
        update_state_dict[module_state_dict_name + ".sparse.weight"] = s_.bfloat16().cpu()

    torch.cuda.empty_cache()
    accelerator.free_memory()


def _substitute_layer_linear_weight(
        module_list: list[nn.Module],
        accelerator: Accelerator,
        parameter_ratio: float,
        has_sparse: bool,
        use_svd: bool,
        update_state_dict: dict,
        module_state_dict_name_list: list[str],
        device,
        sparse_weight: torch.Tensor = None,
        **kwargs
) -> nn.Module:
    has_bias = module_list[0].bias is not None

    u_list = []
    s_list = []
    v_list = []
    max_reduced_rank = 0

    # Calculate all singular values
    for i, module in enumerate(module_list):
        module_state_dict_name = module_state_dict_name_list[i]
        accelerator.print(f"Performing SVD for layer {module_state_dict_name}...")

        if use_svd:
            # Decompose a matrix by SVD
            if sparse_weight is None:
                weight_tensor = module.weight.data.to(device)
            else:
                weight_tensor = module.weight.data.to(device) - sparse_weight
            u_, s_, v_ = svd(weight_tensor)
            height, width = weight_tensor.size()
            this_max_reduced_rank = math.ceil(parameter_ratio * (height * width) / (height + width))
            u_list.append(u_)
            s_list.append(s_)
            v_list.append(v_)
            max_reduced_rank += this_max_reduced_rank
        else:
            raise NotImplementedError

        torch.cuda.empty_cache()
        accelerator.free_memory()

    # Distribute reduced ranks for each weight
    accelerator.print(f"Calculating reduced ranks...")
    all_s = torch.cat(s_list)
    sorted_s, indices_s = torch.sort(all_s, descending=True)
    topk_indices_s = indices_s[:max_reduced_rank]

    weight_lens = torch.tensor([s_.shape[0] for s_ in s_list], device=topk_indices_s.device)
    weight_ids = torch.arange(len(module_list), device=topk_indices_s.device).repeat_interleave(weight_lens)

    reduced_ranks = torch.bincount(weight_ids[topk_indices_s]).tolist()
    accelerator.print("Reduced ranks: ", reduced_ranks)

    # Decompose
    for i, module in enumerate(module_list):
        module_state_dict_name = module_state_dict_name_list[i]
        accelerator.print(f"Decomposing layer {module_state_dict_name}...")

        if use_svd:
            if sparse_weight is None:
                weight_tensor = module.weight.data.to(device)
                output = low_rank_decomposition(weight_tensor, parameter_ratio=parameter_ratio, return_dict=True,
                                                u_=u_list[i], s_=s_list[i], v_=v_list[i], reduced_rank=reduced_ranks[i],
                                                accelerator=accelerator, **kwargs)
                l_, r_, reduced_rank = output['L'], output['R'], output['reduced_rank']
                s_ = weight_tensor - torch.mm(l_, r_)
            else:
                weight_tensor = module.weight.data.to(device) - sparse_weight
                output = low_rank_decomposition(weight_tensor, parameter_ratio=parameter_ratio, return_dict=True,
                                                u_=u_list[i], s_=s_list[i], v_=v_list[i], reduced_rank=reduced_ranks[i],
                                                accelerator=accelerator, **kwargs)
                l_, r_, reduced_rank = output['L'], output['R'], output['reduced_rank']
                s_ = sparse_weight
        else:
            raise NotImplementedError

        update_state_dict[module_state_dict_name + ".left.weight"] = l_.bfloat16().cpu()  # 🔍 to cpu
        update_state_dict[module_state_dict_name + ".right.weight"] = r_.bfloat16().cpu()
        if has_sparse:
            update_state_dict[module_state_dict_name + ".sparse.weight"] = s_.bfloat16().cpu()

        torch.cuda.empty_cache()
        accelerator.free_memory()

    return reduced_ranks


@torch.no_grad()
def decompose_moe(args, model, accelerator: Accelerator):
    device = accelerator.device
    unwrapped_model = accelerator.unwrap_model(model)  # 🔍 unwrap model first
    use_cache = unwrapped_model.config.use_cache
    unwrapped_model.config.use_cache = False
    layers = unwrapped_model.model.layers

    # Get configs
    parameter_ratio = 1.0 - args.sparsity_ratio
    level = args.level
    has_sparse = args.has_sparse
    do_permute = args.do_permute
    use_svd = args.use_svd

    # 🔍 store the pruned parameters in CPU
    update_state_dict = {}
    accelerator.print('Starting ...')

    if level == "expert":
        for i in tqdm(range(len(layers)), desc="Decomposing layers...", disable=not accelerator.is_main_process):
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]
            subset = find_moe_expert_linears(layer)  # 🔍 Find layers to prune
            accelerator.print(subset)

            layer.to(device)

            if has_sparse:
                if do_permute:  # 🔍
                    get_permuted_state_dict_mixtral(
                        state_dict=subset,
                        target_expert_id=3,
                        source_expert_id_list=[0, 1, 2, 4, 5, 6, 7],
                    )
                sparse_weight = {
                    "w1": torch.stack([subset[f"block_sparse_moe.experts.{i}.w1"].weight.data for i in range(unwrapped_model.config.num_local_experts)], dim=0).mean(0),
                    "w2": torch.stack([subset[f"block_sparse_moe.experts.{i}.w2"].weight.data for i in range(unwrapped_model.config.num_local_experts)], dim=0).mean(0),
                    "w3": torch.stack([subset[f"block_sparse_moe.experts.{i}.w3"].weight.data for i in range(unwrapped_model.config.num_local_experts)], dim=0).mean(0),
                }
            else:
                sparse_weight = {
                    "w1": None,
                    "w2": None,
                    "w3": None,
                }

            accelerator.print("sparse_weight", sparse_weight)

            # Wrap layers
            for name in subset:
                # name: block_sparse_moe.experts.{}.w1
                module_state_dict_name = f"model.layers.{i}.{name}"
                accelerator.print(f"Decomposing layer {i} {name}...")
                weight_type = (
                    "w1" if "w1" in module_state_dict_name else
                    "w2" if "w2" in module_state_dict_name else
                    "w3"
                )
                _substitute_single_linear_weight(
                    module=subset[name],
                    accelerator=accelerator,
                    parameter_ratio=parameter_ratio,
                    has_sparse=has_sparse,
                    use_svd=use_svd,
                    update_state_dict=update_state_dict,
                    module_state_dict_name=module_state_dict_name,
                    device=device,
                    sparse_weight=sparse_weight[weight_type],
                    # **kwargs
                )

            layer.to("cpu")

        # 🔍 set parameter ratio for saving.
        setattr(unwrapped_model.config, "reduced_rank", update_state_dict["model.layers.0.block_sparse_moe.experts.0.w1.left.weight"].shape[1])

    elif level == "layer":
        reduced_rank = []

        for i in tqdm(range(len(layers)), desc="Decomposing layers...", disable=not accelerator.is_main_process):
            sys.stderr.flush()
            torch.cuda.empty_cache()
            print_gpu_memory(accelerator)
            layer = layers[i]
            subset = find_moe_expert_linears(layer)  # 🔍 Find layers to prune
            accelerator.print(subset)

            layer.to(device)

            if has_sparse:
                if do_permute:
                    get_permuted_state_dict_mixtral(
                        state_dict=subset,
                        target_expert_id=3,
                        source_expert_id_list=[0, 1, 2, 4, 5, 6, 7],
                    )
                sparse_weight = {
                    "w1": torch.stack([subset[f"block_sparse_moe.experts.{i}.w1"].weight.data for i in range(unwrapped_model.config.num_local_experts)], dim=0).mean(0),
                    "w2": torch.stack([subset[f"block_sparse_moe.experts.{i}.w2"].weight.data for i in range(unwrapped_model.config.num_local_experts)], dim=0).mean(0),
                    "w3": torch.stack([subset[f"block_sparse_moe.experts.{i}.w3"].weight.data for i in range(unwrapped_model.config.num_local_experts)], dim=0).mean(0),
                }
            else:
                sparse_weight = {
                    "w1": None,
                    "w2": None,
                    "w3": None,
                }

            accelerator.print("sparse_weight", sparse_weight)

            #######################################################
            # 🔍 Wrap layers (HERE DIFFERENT FROM EXPERT-LEVEL)
            w1_list = []
            w2_list = []
            w3_list = []
            w1_module_state_dict_name_list = []
            w2_module_state_dict_name_list = []
            w3_module_state_dict_name_list = []

            for name in subset:
                # name: block_sparse_moe.experts.{}.w1
                module_state_dict_name = f"model.layers.{i}.{name}"
                if "w1" in module_state_dict_name:
                    w1_list.append(subset[name])
                    w1_module_state_dict_name_list.append(module_state_dict_name)
                elif "w2" in module_state_dict_name:
                    w2_list.append(subset[name])
                    w2_module_state_dict_name_list.append(module_state_dict_name)
                else:
                    w3_list.append(subset[name])
                    w3_module_state_dict_name_list.append(module_state_dict_name)

            reduced_ranks_w1 = _substitute_layer_linear_weight(
                module_list=w1_list,
                accelerator=accelerator,
                parameter_ratio=parameter_ratio,
                has_sparse=has_sparse,
                use_svd=use_svd,
                update_state_dict=update_state_dict,
                module_state_dict_name_list=w1_module_state_dict_name_list,
                device=device,
                sparse_weight=sparse_weight["w1"],
                # **kwargs
            )
            reduced_ranks_w2 = _substitute_layer_linear_weight(
                module_list=w2_list,
                accelerator=accelerator,
                parameter_ratio=parameter_ratio,
                has_sparse=has_sparse,
                use_svd=use_svd,
                update_state_dict=update_state_dict,
                module_state_dict_name_list=w2_module_state_dict_name_list,
                device=device,
                sparse_weight=sparse_weight["w2"],
                # **kwargs
            )
            reduced_ranks_w3 = _substitute_layer_linear_weight(
                module_list=w3_list,
                accelerator=accelerator,
                parameter_ratio=parameter_ratio,
                has_sparse=has_sparse,
                use_svd=use_svd,
                update_state_dict=update_state_dict,
                module_state_dict_name_list=w3_module_state_dict_name_list,
                device=device,
                sparse_weight=sparse_weight["w3"],
                # **kwargs
            )

            # 🔍 set reduced rank for saving.
            reduced_rank.append(
                {
                    "w1": reduced_ranks_w1,
                    "w2": reduced_ranks_w2,
                    "w3": reduced_ranks_w3,
                }
            )
            setattr(unwrapped_model.config, "reduced_rank", reduced_rank)
            #######################################################

            layer.to("cpu")

    elif level == "model":
        raise NotImplementedError
    else:
        raise NotImplementedError

    accelerator.print(f"update_state_dict: {update_state_dict.keys()}")
    accelerator.print("Decomposition done!")
    unwrapped_model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    return update_state_dict
