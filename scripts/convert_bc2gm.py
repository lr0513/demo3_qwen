import json
from pathlib import Path


ROOT = Path(r"/root/autodl-tmp/demo3/llama")
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data"

INSTRUCTION = (
    "Extract all GENE entities from the sentence. "
    "Return a JSON array. Each item must contain "
    "name and type. Return [] if none."
)


def convert(split):
    raw_path = RAW_DIR / f"{split}.json"

    out_path = OUT_DIR / f"bc2gm_{split}_name_only.json"

    samples = json.loads(raw_path.read_text(encoding="utf-8"))
    converted = []

    for sample in samples:
        entities = [
            {
                "name": entity["name"],
                "type": entity["type"],
            }
            for entity in sample.get("entities", [])
        ]

        converted.append(
            {
                "instruction": INSTRUCTION,
                "input": sample["sentence"],
                "output": json.dumps(entities, ensure_ascii=False),
            }
        )

    out_path.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"{raw_path} -> {out_path}: {len(converted)}")


if __name__ == "__main__":
    convert("train")
    convert("dev")
    convert("test")