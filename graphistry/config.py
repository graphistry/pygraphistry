from typing import Any, Dict

class GraphistryConfig:
    def __init__(self):
        self.config: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def unset(self, key: str) -> None:
        if key in self.config:
            del self.config[key]


###


config = GraphistryConfig()

def set(key: str, value: Any) -> None:
    config.set(key, value)

def get(key: str, default: Any = None) -> Any:
    return config.get(key, default)

def unset(key: str) -> None:
    config.unset(key)
