import argparse

import torch

from src.config import ExperimentConfig
from src.data import build_prompt
from src.model import load_inference_model, load_tokenizer
from src.postprocess import postprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--text", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig.from_file(args.config)
    model_path = args.model_path or config.require("model_path")
    output_dir = config.resolve_path(config.require("output_dir"))
    adapter = args.adapter or str(output_dir / "best")
    tokenizer = load_tokenizer(model_path)
    model = load_inference_model(
        model_path,
        adapter,
        config.require("method"),
        config,
    )
    if hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    device = next(model.parameters()).device
    text = args.text or input("Input sentence: ").strip()
    prompt = build_prompt(tokenizer, text, config.require("instruction"))
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.get("max_length", 256),
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.get("max_new_tokens", 256),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    entities, _ = postprocess(text, raw)
    print("model_output:", raw)
    print("entities:", entities)


if __name__ == "__main__":
    main()
