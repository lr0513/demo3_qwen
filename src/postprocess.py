import json
import re


def _extract_json_text(text):
    '''去掉markdown代码块标记'''
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_prediction(text):
    '''提取并解析模型输出，拿到实体列表[{"name":"xxx","type":"GENE"}]'''
    text = _extract_json_text(text)
    candidates = [text]

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end > object_start:
        candidates.append(text[object_start:object_end + 1])

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end > array_start:
        candidates.append(text[array_start:array_end + 1])

    parsed = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError):
            continue

    if parsed is None:
        return [], False

    if isinstance(parsed, dict):
        parsed = parsed.get("entities", [])
    if not isinstance(parsed, list):
        return [], False

    entities = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        entity_type = item.get("type", "GENE")
        if isinstance(name, str) and name.strip():
            entities.append(
                {
                    "name": name.strip(),
                    "type": str(entity_type),
                }
            )
    return entities, True


def locate_entities(sentence, entities):
    '''在原始输入句子中，查找实体字符串，算出字符start、end位置'''
    located = []
    used_starts = set()

    for entity in entities:
        name = entity["name"]
        search_start = 0
        start = -1

        while True:
            start = sentence.find(name, search_start)
            if start == -1 or start not in used_starts:
                break
            search_start = start + 1

        if start == -1:
            continue

        used_starts.add(start)
        located.append(
            {
                "name": name,
                "type": entity["type"],
                "start": start,
                "end": start + len(name),
            }
        )

    return located


def postprocess(sentence, model_output):
    entities, parse_ok = parse_prediction(model_output)
    return locate_entities(sentence, entities), parse_ok

