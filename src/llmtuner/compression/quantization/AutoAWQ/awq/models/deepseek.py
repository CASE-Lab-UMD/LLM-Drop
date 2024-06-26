from typing import List, Tuple

import torch
import tqdm
from awq.modules.fused.block import MixtralBlock
from awq.modules.fused.model import MixtralModel
# from awq.modules.fused.moe import FusedSparseMoeBlock
from awq.modules.fused.moe import FusedDeepseekMoEBlock as FusedSparseMoeBlock

from awq.modules.fused.norm import FasterTransformerRMSNorm
from awq.modules.linear import WQLinear_GEMM
from awq.utils.fused_utils import fuse_qkv, fuse_linears

from .base import BaseAWQForCausalLM
from .deepseek_moe.modeling_deepseek import (
    DeepseekDecoderLayer as OldDeepseekDecoderLayer,
    DeepseekForCausalLM as OldDeepseekForCausalLM,
    DeepseekMoE,
)


class DeepseekAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "DeepseekDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    # modules_to_not_convert = ["gate", "self_attn"]  # 🔍 may exclude the first layer too.

    @staticmethod
    def fuse_layers(model: OldDeepseekForCausalLM):
        fuser = DeepseekFuser(model)
        fuser.fuse_transformer()

    @staticmethod
    def get_model_layers(model: OldDeepseekForCausalLM):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldDeepseekForCausalLM, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

    @staticmethod
    def get_layers_for_scaling(
            module: OldDeepseekDecoderLayer, input_feat, module_kwargs
    ):
        layers = []
        print(f"input_feat: {input_feat.keys()}")
        # attention input
        if "self_attn.q_proj" in input_feat:
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        # The line `# print(f"input_feat: {input_feat.keys()}")` is a commented-out line of code in
                        # Python. It is using string formatting to print out the keys of the `input_feat` dictionary.
                        # However, since it is commented out with a `#` at the beginning, it will not be executed when
                        # the code runs.
                        module.self_attn.q_proj,
                        module.self_attn.k_proj,
                        module.self_attn.v_proj,
                    ],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=module.self_attn,
                    kwargs=module_kwargs,
                )
            )

        # attention out
        if "self_attn.o_proj" in input_feat:
            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.self_attn.v_proj,
                        layers=[module.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )

        if isinstance(module.mlp, DeepseekMoE):  # MoE
            # linear in
            shared_experts_in = [module.mlp.shared_experts.gate_proj, module.mlp.shared_experts.up_proj] \
                if module.mlp.config.n_shared_experts is not None else []
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[
                               w
                               for expert in module.mlp.experts
                               for w in [expert.gate_proj, expert.up_proj]
                           ] + shared_experts_in,
                    inp=input_feat["mlp"],
                    module2inspect=module.mlp,
                )
            )

            # linear out
            for i, expert in enumerate(module.mlp.experts):
                layers.append(
                    dict(
                        prev_op=expert.up_proj,
                        layers=[expert.down_proj],
                        inp=input_feat[f"mlp.experts.{i}.down_proj"],
                    )
                )
            if module.mlp.config.n_shared_experts is not None:
                layers.append(
                    dict(
                        prev_op=module.mlp.shared_experts.up_proj,
                        layers=[module.mlp.shared_experts.down_proj],
                        inp=input_feat[f"mlp.shared_experts.down_proj"],
                    )
                )

        else:  # MLP
            # linear in
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp"],
                    module2inspect=module.mlp,
                )
            )

            # linear out
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )
        # print(layers)
        return layers


class DeepseekFuser:
    # TODO: here not modified yet
    def __init__(self, model: OldDeepseekForCausalLM):
        self.model = model

        self.mixtral_blocks: List[Tuple[str, OldDeepseekDecoderLayer]] = [
            (name, module)
            for name, module in self.model.named_modules()
            if "DeepseekDecoderLayer".lower() in module.__class__.__name__.lower()
        ]

    def fuse_transformer(self):
        blocks = []

        module: OldDeepseekDecoderLayer
        for module in tqdm.tqdm(self.model.model.layers, desc="Fusing layers..."):
            device = next(iter(module.state_dict().values())).device

            qkv = fuse_qkv(
                module,
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            )
            norm_1 = FasterTransformerRMSNorm(
                module.input_layernorm.weight, module.input_layernorm.variance_epsilon
            )
            norm_2 = None
            if module.post_attention_layernorm is not None:
                norm_2 = FasterTransformerRMSNorm(
                    module.post_attention_layernorm.weight,
                    module.post_attention_layernorm.variance_epsilon,
                )

            sparse_moe = module.mlp
            if sparse_moe is not None and isinstance(sparse_moe, DeepseekMoE) and isinstance(sparse_moe.experts[0].gate_proj, WQLinear_GEMM):
                fused_w1w3s = [
                    fuse_linears(
                        [
                            sparse_moe.experts[i].gate_proj,
                            sparse_moe.experts[i].up_proj,
                        ],
                        device,
                    )
                    for i in range(len(sparse_moe.experts))
                ]

                stacked_w1w3s = fuse_linears(
                    fused_w1w3s, device, dim=0, operation=torch.stack
                )

                stacked_w2s = fuse_linears(
                    [expert.down_proj for expert in sparse_moe.experts],
                    device,
                    dim=0,
                    operation=torch.stack,
                )

                shared_experts = sparse_moe.shared_experts if hasattr(sparse_moe, "shared_experts") else None
                sparse_moe = FusedSparseMoeBlock(
                    top_k=sparse_moe.gate.top_k,
                    gate=sparse_moe.gate,
                    ws=stacked_w1w3s,
                    w2s=stacked_w2s,
                    shared_experts=shared_experts, 
                )

            blocks.append(
                MixtralBlock(
                    hidden_size=self.model.config.hidden_size,
                    n_heads=self.model.config.num_attention_heads,
                    n_kv_heads=self.model.config.num_key_value_heads,
                    qkv_layer=qkv,
                    o_proj=module.self_attn.o_proj,
                    moe=sparse_moe,
                    norm_1=norm_1,
                    norm_2=norm_2,
                    dev=device,
                    max_seq_len=self.model.config.max_seq_len,
                    rope_theta=self.model.config.rope_theta,
                )
            )

        model_norm = FasterTransformerRMSNorm(
            self.model.model.norm.weight,
            self.model.model.norm.variance_epsilon,
        )

        self.model.model = MixtralModel(
            self.model.config.vocab_size,
            blocks,
            self.model.model.embed_tokens,
            model_norm,
        )
        setattr(self.model.model, "blocks", self.model.model.blocks)
