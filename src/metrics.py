from src.postprocess import locate_entities, parse_prediction


class EntityMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.total = 0
        self.parse_errors = 0
        self.invalid_entities = 0
        self.total_pred_entities = 0
        self.total_true_entities = 0
        self.records = []

    @staticmethod
    def _gold_set(entities):
        return {
            (entity["type"], entity["pos"][0], entity["pos"][1])
            for entity in entities
        }

    @staticmethod
    def _pred_set(entities):
        return {
            (entity["type"], entity["start"], entity["end"])
            for entity in entities
        }

    def add_sample(self, sentence, gold_entities, model_output):
        gold = self._gold_set(gold_entities)

        raw_predictions, parse_ok = parse_prediction(model_output)
        located_predictions = locate_entities(sentence, raw_predictions)
        predicted = self._pred_set(located_predictions)

        tp = len(gold & predicted)
        total_pred_entities = len(raw_predictions)
        total_true_entities = len(gold)
        fp = total_pred_entities - tp
        fn = total_true_entities - tp

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.total_pred_entities += total_pred_entities
        self.total_true_entities += total_true_entities
        self.total += 1

        if not parse_ok:
            self.parse_errors += 1
        self.invalid_entities += len(raw_predictions) - len(located_predictions)

        self.records.append(
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

    def compute(self):
        precision = (
            self.tp / self.total_pred_entities
            if self.total_pred_entities > 0
            else 0.0
        )
        recall = (
            self.tp / self.total_true_entities
            if self.total_true_entities > 0
            else 0.0
        )
        denominator = self.total_pred_entities + self.total_true_entities
        f1 = 2 * self.tp / denominator if denominator > 0 else 0.0
        parse_error_rate = self.parse_errors / self.total if self.total else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "total_pred_entities": self.total_pred_entities,
            "total_true_entities": self.total_true_entities,
            "total_samples": self.total,
            "json_parse_errors": self.parse_errors,
            "json_parse_error_rate": parse_error_rate,
            "invalid_entities": self.invalid_entities,
            "samples": self.records,
        }

