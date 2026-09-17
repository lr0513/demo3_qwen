import json
import re


def parse_entities(text):
    """
    把模型生成的文本解析成实体列表。

    支持：
    1. 纯 JSON
    2. ```json ... ``` 包裹的 JSON
    """
    text = text.strip()

    # 去掉 Markdown 代码块标记
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(data, dict):
        data = data.get("entities", [])

    if not isinstance(data, list):
        return []

    entities = []
    for item in data:
        if not isinstance(item, dict):
            continue

        name = item.get("name")
        entity_type = item.get("type", "GENE")

        if isinstance(name, str) and name.strip():
            entities.append(
                {
                    "name": name.strip(),
                    "type": entity_type,
                }
            )

    return entities


def locate_entities(sentence, entities):
    """
    根据实体名称，在原始句子中计算 start/end。

    模型不负责生成准确位置，Python 负责计算。
    """
    results = []
    cursor = 0

    for entity in entities:
        name = entity["name"]
        entity_type = entity["type"]

        # 从 cursor 后面开始查找，处理同一个句子中重复出现的实体
        start = sentence.find(name, cursor)

        # 如果后面找不到，再从句子开头找一次
        if start == -1:
            start = sentence.find(name)

        if start == -1:
            # 原句中找不到这个实体，跳过，避免错误位置进入 F1
            continue

        end = start + len(name)
        cursor = end

        results.append(
            {
                "name": name,
                "type": entity_type,
                "start": start,
                "end": end,
            }
        )

    return results


def postprocess(sentence, model_output):
    """
    完整后处理流程：
    模型文本 -> 实体列表 -> 补全位置
    """
    entities = parse_entities(model_output)
    return locate_entities(sentence, entities)