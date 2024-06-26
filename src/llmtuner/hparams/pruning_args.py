from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Literal

EXPERT_DROP_METHODS = ('global_pruning', 'layerwise_pruning', 'progressive_pruning', 'dynamic_skipping', 'post_dropping')
LAYER_DROP_METHODS = ('consecutive', 'discrete', 'post_dropping')
BLOCK_DROP_METHODS = ('consecutive', 'discrete', 'post_dropping')


@dataclass
class PruningArguments:
    r"""
    Arguments pertaining to specify the decoding parameters.
    """
    prune_seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed for sampling the calibration data."},
    )
    prune_method: Optional[str] = field(
        default="wanda",
        metadata={"choices": ["wanda", "sparsegpt", "gradient-first", "gradient-zeroth", "magnitude", "remap_gate", "decompose_moe", "expert_drop", "block_drop", "layer_drop"]},
    )
    prune_model_save_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save the pruned model."},
    )
    n_calibration_samples: Optional[int] = field(
        default=128,
        metadata={"help": "Number of calibration samples."},
    )
    prune_data_type: Literal["pt", "sft", "rm", "ppo"] = field(
        default="sft",
        metadata={"choices": ["pt", "sft", "rm", "ppo"],
                  "help": "Path to save the pruned model."},
    )

    # 🔍 For pruning
    sparsity_ratio: Optional[float] = field(  # this term denotes the "parameter_ratio" for decomposition
        default=0.5,
        metadata={"help": "Sparsity Level."},
    )
    sparsity_type: Optional[Literal["structured", "unstructured", "4:8", "2:4"]] = field(
        default="unstructured",
        metadata={"choices": ["structured", "unstructured", "4:8", "2:4"]},
    )
    use_variant: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use the variant for Wanda."},
    )

    # 🔍 For decomposition
    level: Optional[str] = field(
        default="expert",
        metadata={"choices": ["expert", "layer", "model"]},
    )
    has_sparse: Optional[bool] = field(
        default=True,
    )
    do_permute: Optional[bool] = field(
        default=True,
    )
    use_svd: Optional[bool] = field(
        default=True,
    )
    top_scores: Optional[bool] = field(
        default=True,
    )

    # 🔍 For expert drop
    expert_drop_method: Optional[str] = field(
        default="layerwise_pruning",
        metadata={"help": ' '.join(['Supported dropping methods:'] + list(EXPERT_DROP_METHODS)),
                  "choices": EXPERT_DROP_METHODS},
    )
    r: Optional[int] = field(
        default=4,
        metadata={"help": 'Number of experts to preserve'}
    )

    # 🔍 For layer drop & block drop
    layer_drop_method: Optional[str] = field(
        default="discrete",
        metadata={"help": ' '.join(['Supported dropping methods:'] + list(LAYER_DROP_METHODS)),
                  "choices": LAYER_DROP_METHODS},
    )
    block_drop_method: Optional[str] = field(
        default="discrete",
        metadata={"help": ' '.join(['Supported dropping methods:'] + list(BLOCK_DROP_METHODS)),
                  "choices": BLOCK_DROP_METHODS},
    )
    drop_n: Optional[int] = field(
        default=4,
        metadata={"help": 'Number of blocks to drop'}
    )
    layer_drop_norm: Optional[bool] = field(
        default=True,
        metadata={"help": 'determine whether to consider norm when calculating similarity. If True, use the hidden states before norm to calculate similarity.'}
    )
    target_layer: Optional[str] = field(
        default=None,
        metadata={"help": 'determine which type of layer is dropped when layer_drop. ',
            "choices": ["mlp", "attn", "all"]},
    )
    only_update_config: Optional[bool] = field(
        default=False,
        metadata={"help": 'Only output the config file without saving model weights. '}
    )
    similarity_cache_file: Optional[str] = field(
        default=None,
        metadata={"help": 'Cached file storing the similarity scores across layers to reduce the computation consumption. '
                          'If the file does not exist, it will be created.'},
    )

    # 🔍 For gate-remapping
    pruned_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pruned model. (Only for Gate-Remapping)"},
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args
