import json
from pathlib import Path

path = Path(
    r"/root/autodl-tmp/demo3/llama/outputs/bc2gm_eval_name_only_v3_200.json"
)
data = json.loads(path.read_text(encoding="utf-8"))

for sample in data["samples"]:
    if sample["fp"] == 0 and sample["fn"] == 0:
        continue

    print("=" * 80)
    print("SENTENCE:", sample["sentence"])
    print("GOLD:", sample["gold"])
    print("PREDICTED:", sample["predicted"])
    print("MODEL OUTPUT:", sample["model_output"])
    print("TP:", sample["tp"], "FP:", sample["fp"], "FN:", sample["fn"])