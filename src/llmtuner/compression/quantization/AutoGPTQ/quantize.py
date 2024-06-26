import json
import time
import torch
import random
from transformers import AutoTokenizer, TextGenerationPipeline, GenerationConfig
from argparse import ArgumentParser

from llmtuner.compression.quantization.AutoGPTQ.auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from fastchat.conversation import get_conv_template


llama_2_template = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{input} [/INST]
"""


def load_data(data_path, tokenizer, n_samples, template='default'):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                if template == 'default':
                    prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                    text = prompt + opt
                else:
                    conv = get_conv_template(template)
                    conv.append_message(conv.roles[0], f'{istr} {inp}')
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    conv = get_conv_template(template)
                    conv.append_message(conv.roles[0], f'{istr} {inp}')
                    conv.append_message(conv.roles[1], opt)
                    text = conv.get_prompt()
            else:
                if template == 'default':
                    prompt = f"Instruction:\n{istr}\nOutput:\n"
                    text = prompt + opt
                else:
                    conv = get_conv_template(template)
                    conv.append_message(conv.roles[0], istr)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    conv = get_conv_template(template)
                    conv.append_message(conv.roles[0], istr)
                    conv.append_message(conv.roles[1], opt)
                    text = conv.get_prompt()
            print('*' * 20)
            print(prompt)
            print('-' * 20)
            print(text)
            print('*' * 20)

            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"]
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_dir", type=str)
    parser.add_argument("--quantized_model_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 6, 8])
    parser.add_argument("--group_size", type=int, default=128, help="group size, -1 means no grouping or full rank")
    parser.add_argument("--desc_act", action="store_true", help="whether to quantize with desc_act")
    parser.add_argument("--num_samples", type=int, default=128, help="how many samples will be used to quantize model")
    parser.add_argument("--save_and_reload", action="store_true", help="whether save quantized model to disk and reload back")
    parser.add_argument("--fast_tokenizer", action="store_true", help="whether use fast tokenizer")
    parser.add_argument("--use_triton", action="store_true", help="whether use triton to speedup at inference")
    parser.add_argument("--per_gpu_max_memory", type=int, default=None, help="max memory used to load model per gpu")
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="max memory used to offload model to cpu")
    parser.add_argument("--quant_batch_size", type=int, default=1, help="examples batch size for quantization")
    parser.add_argument("--trust_remote_code", action="store_true", help="whether to trust remote code when loading model")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--calibration-template', default='default', choices=['default', 'llama-2', 'mistral', 'vicuna_v1.1', 'redpajama-incite-instruct'])
    args = parser.parse_args()

    if args.seed is not None:
        print(f'Random Seed: {args.seed}')
        random.seed(args.seed)
    else:
        print('No seed is set')

    max_memory = dict()
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update(
                {i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())}
            )
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_dir,
        use_fast=args.fast_tokenizer,
        trust_remote_code=args.trust_remote_code
    )
    model = AutoGPTQForCausalLM.from_pretrained(
        args.pretrained_model_dir,
        quantize_config=BaseQuantizeConfig(bits=args.bits, group_size=args.group_size, desc_act=args.desc_act),
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code
    )

    examples = load_data("./dataset/alpaca_data_cleaned.json", tokenizer, args.num_samples, template=args.calibration_template)
    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
        for example in examples
    ]

    start = time.time()
    model.quantize(
        examples_for_quant,
        batch_size=args.quant_batch_size,
        use_triton=args.use_triton,
        autotune_warmup_after_quantized=args.use_triton
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    if not args.quantized_model_dir:
        args.quantized_model_dir = args.pretrained_model_dir

    if args.save_and_reload:
        tokenizer.save_pretrained(args.quantized_model_dir)
        model.save_quantized(args.quantized_model_dir)
        gen_config = GenerationConfig.from_pretrained(args.pretrained_model_dir)
        gen_config.save_pretrained(args.quantized_model_dir)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoGPTQForCausalLM.from_quantized(
            args.quantized_model_dir,
            device="cuda:0",
            use_triton=args.use_triton,
            max_memory=max_memory,
            inject_fused_mlp=True,
            inject_fused_attention=False,
            trust_remote_code=args.trust_remote_code
        )

    print(f"model: {model}")
    pipeline_init_kwargs = {"model": model, "tokenizer": tokenizer}
    pipeline = TextGenerationPipeline(**pipeline_init_kwargs)
    for example in random.sample(examples, k=min(4, len(examples))):
        print(f"prompt: {example['prompt']}")
        print("-" * 42)
        print(f"golden: {example['output']}")
        print("-" * 42)
        start = time.time()
        generated_text = pipeline(
            example['prompt'],
            return_full_text=False,
            num_beams=1,
            max_length=len(example["input_ids"]) + 128  # use this instead of max_new_token to disable UserWarning when integrate with logging
        )[0]['generated_text']
        end = time.time()
        print(f"quant: {generated_text}")
        num_new_tokens = len(tokenizer(generated_text)["input_ids"])
        print(f"generate {num_new_tokens} tokens using {end-start: .4f}s, {num_new_tokens / (end - start)} tokens/s.")
        print("=" * 42)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()