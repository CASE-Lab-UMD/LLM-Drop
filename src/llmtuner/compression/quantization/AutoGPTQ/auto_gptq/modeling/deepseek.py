from ._base import BaseGPTQForCausalLM


n_shared_experts = 2
n_routed_experts = 64


class DeepseekGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "DeepseekDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        [f"mlp.experts.{i}.gate_proj" for i in range(n_routed_experts)] + [f"mlp.experts.{i}.up_proj" for i in range(n_routed_experts)], 
        [f"mlp.experts.{i}.down_proj" for i in range(n_routed_experts)],
        [f"mlp.shared_experts.gate_proj"] + ["mlp.shared_experts.up_proj"], 
        [f"mlp.shared_experts.down_proj"],
    ]


__all__ = ["MixtralGPTQForCausalLM"]
