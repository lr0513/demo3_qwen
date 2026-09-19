import json
from pathlib import Path


class ExperimentConfig:
    def __init__(self, values, project_root):
        self.values = values
        self.project_root = Path(project_root).resolve()

    @classmethod
    def from_file(cls, config_path):
        path = Path(config_path).expanduser().resolve()
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls(values=values, project_root=path.parent.parent)

    def get(self, key, default=None):
        return self.values.get(key, default)

    def require(self, key):
        if key not in self.values:
            raise KeyError(f"Missing config key: {key}")
        return self.values[key]

    def resolve_path(self, value):
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return self.project_root / path

    def set(self, key, value):
        self.values[key] = value

