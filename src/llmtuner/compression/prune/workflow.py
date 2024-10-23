
import os
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional

from accelerate import Accelerator
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader

from llmtuner.data import get_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.model import load_model_and_tokenizer

from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, DataCollatorWithPadding
from .io import load_json, save_sparse_model, save_block_dropped_config, save_layer_dropped_config
from .block_drop import consecutive_block_dropping, discrete_block_dropping, post_block_drop
from .layer_drop import discrete_layer_dropping, post_layers_drop

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from llmtuner.hparams import DataArguments, FinetuningArguments, ModelArguments, PruningArguments

LAYER_DROP_METHODS_FUNC = {
    'discrete': discrete_layer_dropping,
}

BLOCK_DROP_METHODS_FUNC = {
    'consecutive': consecutive_block_dropping,
    'discrete': discrete_block_dropping,
}


# 🔍 Modified from src.llmtuner.compression.pt.workflow.run_pt
def run_prune(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        pruning_args: "PruningArguments",  # 🔍 for pruning
        callbacks: Optional[List["TrainerCallback"]] = None,
):
    """Workflow for pruning and decomposing."""
    # 🔍 accelerator
    accelerator = Accelerator()
    accelerator.print(f"{AcceleratorState()}")
    accelerator.print("Pruning Args:", pruning_args)
    accelerator.print("Model Args:", model_args)

    # 🔍 model & tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train)
    
    if pruning_args.prune_method == "layer_drop" and pruning_args.layer_drop_method == "post_dropping":
        assert (os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) and (os.environ.get("ACCELERATE_USE_FSDP", "false"))
        reserved_layer_list = load_json(os.path.join(pruning_args.prune_model_save_path, "reserved_layers.json"))
        post_layers_drop(pruning_args.prune_model_save_path, pruning_args.target_layer, model, tokenizer, reserved_layer_list, accelerator, pruning_args.only_update_config)
        exit()
    if pruning_args.prune_method == "block_drop" and pruning_args.block_drop_method == "post_dropping":
        assert (os.environ.get("ACCELERATE_USE_DEEPSPEED", "false")) and (os.environ.get("ACCELERATE_USE_FSDP", "false"))
        reserved_layer_list = load_json(os.path.join(pruning_args.prune_model_save_path, "reserved_layers.json"))
        post_block_drop(pruning_args.prune_model_save_path, model, tokenizer, reserved_layer_list, accelerator, pruning_args.only_update_config)
        exit()

    # 🔍 dataset & data collator & dataloader
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage=pruning_args.prune_data_type)

    if pruning_args.prune_data_type == "pt":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # concat all data to seq_length for each batch
    elif pruning_args.prune_data_type == "sft":
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        )
    else:
        raise NotImplementedError
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=data_collator, num_workers=8)  # batch size must be 1
    accelerator.print("Total Sample Num:", len(dataset))
    accelerator.print("Total Used Sample Num:", pruning_args.n_calibration_samples)
    accelerator.print("Max sequence Length:", data_args.cutoff_len)
    accelerator.print(f"Example Data (len = {len(dataset[0]['input_ids'])}):", dataset[0])

    if pruning_args.n_calibration_samples > len(dataset):
        raise ValueError("Number of calibration samples is greater than the number of samples in the dataset!")

    # 🔍 Prepare model & dataloader
    print("Preparing model...")
    model, dataloader = accelerator.prepare(model, dataloader)

    # 🔍 Distribute samples to each device for acceleration
    assert (pruning_args.n_calibration_samples % accelerator.num_processes == 0)  # have to be divided evenly
    num_samples_each_device = pruning_args.n_calibration_samples // accelerator.num_processes
    accelerator.print("Number of samples per device:", len(dataloader))
    accelerator.print("Number of used samples per device:", num_samples_each_device)

    #######################################################################################################
    if pruning_args.prune_method == "layer_drop":
        dropped_layer_list = LAYER_DROP_METHODS_FUNC[pruning_args.layer_drop_method](pruning_args, model, dataloader, accelerator, num_samples_each_device)
    elif pruning_args.prune_method == "block_drop":
        dropped_layer_list = BLOCK_DROP_METHODS_FUNC[pruning_args.block_drop_method](pruning_args, model, dataloader, accelerator, num_samples_each_device)
    else:
        raise NotImplementedError
    #######################################################################################################
    accelerator.print(f"model: {model}")
    if pruning_args.prune_model_save_path is not None:
        if pruning_args.prune_method == "layer_drop":
            save_layer_dropped_config(pruning_args.target_layer, pruning_args.prune_model_save_path, model, tokenizer, accelerator, dropped_layer_list=dropped_layer_list)
        elif pruning_args.prune_method == "block_drop":
            save_block_dropped_config(pruning_args.prune_model_save_path, model, tokenizer, accelerator, dropped_layer_list=dropped_layer_list)
        else:
            # 🔍 Save sparse model to disk
            save_sparse_model(pruning_args.prune_model_save_path, model, tokenizer, accelerator, update_state_dict, check_sparsity=True)

    accelerator.print("All done!")

