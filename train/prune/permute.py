import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from typing import Dict, List


@torch.no_grad()
def get_permutation_indices(
        target_input_weights: torch.Tensor,
        target_output_weights: torch.Tensor,
        source_input_weights: torch.Tensor,
        source_output_weights: torch.Tensor,
):
    # input_weights: (intermediate_size, hidden_size)
    # output_weights: (hidden_size, intermediate_size)
    lsa_cost_matrix = torch.mm(target_input_weights, source_input_weights.t())
    lsa_cost_matrix += torch.mm(target_output_weights.t(), source_output_weights)
    _, perm = linear_sum_assignment(lsa_cost_matrix.float().cpu().numpy(), maximize=True)
    return torch.from_numpy(perm).to(lsa_cost_matrix.device)


@torch.no_grad()
def get_permuted_state_dict_mixtral(
        state_dict: Dict[str, nn.Linear],
        target_expert_id: int,
        source_expert_id_list: List[int],
):
    # permute source weights
    for source_expert_id in source_expert_id_list:
        # get state names
        target_gate_state_name = "block_sparse_moe.experts.{}.w1".format(target_expert_id)
        target_down_state_name = "block_sparse_moe.experts.{}.w2".format(target_expert_id)
        source_gate_state_name = "block_sparse_moe.experts.{}.w1".format(source_expert_id)
        source_down_state_name = "block_sparse_moe.experts.{}.w2".format(source_expert_id)

        # permute neurons
        print(f"Permuting weights for source expert {source_expert_id} (target {target_expert_id})")
        permutation_indices = get_permutation_indices(
            state_dict[target_gate_state_name].weight.data,
            state_dict[target_down_state_name].weight.data,
            state_dict[source_gate_state_name].weight.data,
            state_dict[source_down_state_name].weight.data
        )
        print("Permutation", permutation_indices)

        # add the source permutation order to dict
        # permutation_indices_dict["block_sparse_moe.experts.{}.permutation".format(source_expert_id)] = permutation_indices

        # auxiliary state names
        source_up_state_name = "block_sparse_moe.experts.{}.w3".format(source_expert_id)

        # add to state dict
        state_dict[source_up_state_name].weight.data = state_dict[source_up_state_name].weight.data[permutation_indices, :]
        state_dict[source_gate_state_name].weight.data = state_dict[source_gate_state_name].weight.data[permutation_indices, :]
        state_dict[source_down_state_name].weight.data = state_dict[source_down_state_name].weight.data[:, permutation_indices]

    # add the target permutation order to dict
    # permutation_indices_dict["block_sparse_moe.experts.{}.permutation".format(target_expert_id)] = torch.arange(permutation_indices.numel(), device=permutation_indices.device)
    # return permutation_indices_dict


@torch.no_grad()
def get_permuted_state_dict_llama_moe(
        state_dict: Dict[str, torch.Tensor],
        layer_id: int,
        target_expert_id: int,
        source_expert_id_list: List[int],
):
    # TODO: adjust according to the mixtral version
    permuted_state_dict = {}

    # add target weights
    for template in [
        "model.layers.{}.mlp.calculator.experts.weight_up.{}",
        "model.layers.{}.mlp.calculator.experts.weight_gate.{}",
        "model.layers.{}.mlp.calculator.experts.weight_down.{}"
    ]:
        state_name = template.format(layer_id, target_expert_id)
        permuted_state_dict[state_name] = state_dict[state_name]

    # add source weights
    for source_expert_id in source_expert_id_list:
        # get state names
        target_gate_state_name = "model.layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_id, target_expert_id)
        target_down_state_name = "model.layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_id, target_expert_id)
        source_gate_state_name = "model.layers.{}.mlp.calculator.experts.weight_gate.{}".format(layer_id, source_expert_id)
        source_down_state_name = "model.layers.{}.mlp.calculator.experts.weight_down.{}".format(layer_id, source_expert_id)

        # permute neurons
        permutation_indices = get_permutation_indices(
            state_dict[target_gate_state_name],
            state_dict[target_down_state_name],
            state_dict[source_gate_state_name],
            state_dict[source_down_state_name]
        )

        # auxiliary state names
        source_up_state_name = "model.layers.{}.mlp.calculator.experts.weight_up.{}".format(layer_id, source_expert_id)

        # add to state dict
        permuted_state_dict[source_up_state_name] = state_dict[source_up_state_name][permutation_indices, :]
        permuted_state_dict[source_gate_state_name] = state_dict[source_gate_state_name][permutation_indices, :]
        permuted_state_dict[source_down_state_name] = state_dict[source_down_state_name][:, permutation_indices]

    return permuted_state_dict
