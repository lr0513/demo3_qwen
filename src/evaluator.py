import math
from pathlib import Path

import torch
from tqdm import tqdm

from src.data import build_prompt
from src.metrics import EntityMetrics
from src.model import load_inference_model, load_tokenizer


def _set_use_cache(model, value):
    if hasattr(model, "config"):
        model.config.use_cache = value
    try:
        model.base_model.model.config.use_cache = value
    except AttributeError:
        pass


def _clear_sampling_flags(model):
    if not hasattr(model, "generation_config"):
        return
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


@torch.inference_mode()
def evaluate_dataset(
    model,
    tokenizer,
    dataset,
    instruction,
    batch_size=4,
    max_length=256,
    max_new_tokens=256,
    limit=None,
    description="Evaluating",
):
    if limit is not None:
        if hasattr(dataset, "samples"):
            dataset = dataset.samples[:limit]
        else:
            dataset = dataset[:limit]

    device = next(model.parameters()).device
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    was_training = model.training
    model.eval()
    _set_use_cache(model, True)
    _clear_sampling_flags(model)
    metric = EntityMetrics()

    total_batches = math.ceil(len(dataset) / batch_size)
    for start in tqdm(
        range(0, len(dataset), batch_size),
        total=total_batches,
        desc=description,
    ):
        batch_items = [
            dataset[index]
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        prompts = [
            build_prompt(tokenizer, item["sentence"], instruction)
            for item in batch_items
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in inputs.items()
        }

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        input_length = inputs["input_ids"].shape[1]

        for index, item in enumerate(batch_items):
            generated_ids = output_ids[index, input_length:]
            model_output = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            metric.add_sample(
                sentence=item["sentence"],
                gold_entities=item["entities"],
                model_output=model_output,
            )

    tokenizer.padding_side = old_padding_side
    _set_use_cache(model, False)
    if was_training:
        model.train()
    return metric.compute()


def evaluate_adapter(
    base_model_path,
    adapter_path,
    data_path,
    data_dir,
    instruction,
    method,
    config,
    split="test",
    output_path=None,
    limit=None,
):
    from src.data import BC2GMDataset

    tokenizer = load_tokenizer(base_model_path)
    model = load_inference_model(
        base_model_path,
        adapter_path,
        method,
        config,
    )
    dataset = BC2GMDataset(
        Path(data_dir) / f"{split}.json",
        instruction,
    )
    result = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        instruction=instruction,
        batch_size=config.get("eval_batch_size", 4),
        max_length=config.get("max_length", 256),
        max_new_tokens=config.get("max_new_tokens", 256),
        limit=limit,
        description=f"Evaluating {split}",
    )
    if output_path is not None:
        from src.utils import save_json

        save_json(output_path, result)
    return result
