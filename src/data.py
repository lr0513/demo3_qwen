import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_json_samples(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError(f"Dataset must contain a JSON list: {path}")
    return samples


def target_from_entities(entities):
    payload = {
        "entities": [
            {
                "name": entity["name"],
                "type": entity["type"],
            }
            for entity in entities
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_messages(sentence, instruction):
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": sentence},
    ]


def build_prompt(tokenizer, sentence, instruction):
    return tokenizer.apply_chat_template( # 自动拼接<|im_start|>system...<|im_end|>等特殊标记
        build_messages(sentence, instruction),
        tokenize=False, # 返回文本字符串，不转token id
        add_generation_prompt=True, # 末尾加上assistant起始标记<|im_start|>assistant
    )


def build_training_text(tokenizer, sentence, instruction, target):
    messages = build_messages(sentence, instruction)
    messages.append({"role": "assistant", "content": target}) # 追加assistant的标准答案
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


class BC2GMDataset(Dataset):
    def __init__(self, path, instruction, max_samples=None):
        self.samples = load_json_samples(path)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        self.instruction = instruction

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        entities = sample.get("entities", [])
        return {
            "sentence": sample["sentence"],
            "entities": entities,
            "target": target_from_entities(entities),
        }


class CausalLMCollator:
    def __init__(self, tokenizer, instruction, max_length, use_sft_mask=True):
        self.tokenizer = tokenizer
        self.instruction = instruction # 系统提示词
        self.max_length = max_length
        self.use_sft_mask = use_sft_mask
        self.tokenizer.padding_side = "right"

    def __call__(self, batch):
        input_ids_list = []
        labels_list = []

        for item in batch:
            prompt_text = build_prompt(
                self.tokenizer,
                item["sentence"],
                self.instruction,
            )
            full_text = build_training_text(
                self.tokenizer,
                item["sentence"],
                self.instruction,
                item["target"],
            )
            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            full_ids = self.tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise RuntimeError(
                    "The tokenized training text does not start with the prompt."
                )
            if len(prompt_ids) >= self.max_length:
                raise RuntimeError(
                    f"Prompt length {len(prompt_ids)} exceeds "
                    f"max_length={self.max_length}."
                )

            target_ids = full_ids[len(prompt_ids):]
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is not None and (
                not target_ids or target_ids[-1] != eos_token_id
            ):
                target_ids.append(eos_token_id) # 强制让所有样本的答案结尾带上模型结束标记

            merged_ids = prompt_ids + target_ids
            if self.use_sft_mask:
                labels = [-100] * len(prompt_ids) + target_ids
            else:
                labels = list(merged_ids)

            input_ids_list.append(merged_ids[:self.max_length])
            labels_list.append(labels[:self.max_length])

        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        batch_length = max(len(ids) for ids in input_ids_list)

        input_ids = []
        attention_mask = []
        labels = []
        for ids, label_ids in zip(input_ids_list, labels_list):
            padding = batch_length - len(ids)
            input_ids.append(ids + [pad_token_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append(label_ids + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
