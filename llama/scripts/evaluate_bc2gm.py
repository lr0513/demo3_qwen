import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from postprocess import postprocess
BASE_MODEL = "/root/autodl-tmp/demo3/llama/models/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "/root/autodl-tmp/demo3/llama/outputs/bc2gm_lora_name_only_v3"
TEST_PATH = "/root/autodl-tmp/demo3/llama/data/raw/test.json"
OUTPUT_PATH = "/root/autodl-tmp/demo3/llama/outputs/bc2gm_eval_name_only_v3_500.json"

INSTRUCTION = (
    "Extract all GENE entities from the sentence. "
    "Return a JSON array. Each item must contain "
    "name and type. Return [] if none."
)


def build_prompt(tokenizer, sentence):
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": sentence},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_entities(model, tokenizer, sentence, device, max_new_tokens=256):
    prompt = build_prompt(tokenizer, sentence)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_text


def gold_to_set(entities):
    return {
        (entity["type"], entity["pos"][0], entity["pos"][1])
        for entity in entities
    }


def pred_to_set(entities):
    return {
        (entity["type"], entity["start"], entity["end"])
        for entity in entities
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--adapter", type=str, default=ADAPTER_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()

    samples = json.loads(
        Path(TEST_PATH).read_text(encoding="utf-8")
    )
    samples = samples[:args.limit]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    records = []

    for index, sample in enumerate(samples, start=1):
        sentence = sample["sentence"]
        gold = gold_to_set(sample.get("entities", []))

        model_output = generate_entities(
            model,
            tokenizer,
            sentence,
            device,
        )
        predicted_entities = postprocess(sentence, model_output)
        predicted = pred_to_set(predicted_entities)

        tp = len(gold & predicted)
        fp = len(predicted - gold)
        fn = len(gold - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        records.append(
            {
                "sentence": sentence,
                "gold": sorted(gold),
                "model_output": model_output,
                "predicted": sorted(predicted),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

        print(
            f"[{index}/{len(samples)}] "
            f"tp={tp} fp={fp} fn={fn}"
        )

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )
    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "samples": records,
    }

    Path(OUTPUT_PATH).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("========== BC2GM Evaluation ==========")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1:        {f1:.4f}")
    print(f"tp={total_tp}, fp={total_fp}, fn={total_fn}")
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()