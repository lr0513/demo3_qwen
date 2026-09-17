---
library_name: peft
license: other
base_model: /root/autodl-tmp/demo3/llama/models/Qwen2.5-0.5B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: identity_lora_smoke
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# identity_lora_smoke

This model is a fine-tuned version of [/root/autodl-tmp/demo3/llama/models/Qwen2.5-0.5B-Instruct](https://huggingface.co//root/autodl-tmp/demo3/llama/models/Qwen2.5-0.5B-Instruct) on the identity_demo dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 1
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.15.2
- Transformers 4.52.4
- Pytorch 2.5.1+cu121
- Datasets 3.6.0
- Tokenizers 0.21.1