"""Configuration file parser supporting multiple formats."""
import json
import os
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when configuration is invalid."""
    pass


class ConfigParser:
    """Parses and merges configuration from files and environment."""

    def __init__(self, defaults: dict[str, Any] | None = None):
        self._config: dict[str, Any] = defaults or {}

    def load_json(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self._merge(data)

    def load_env(self, prefix: str = "APP_") -> None:
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                self._config[config_key] = self._coerce(value)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._config:
            raise ConfigError(f"Required config key missing: {key}")
        return self._config[key]

    def _merge(self, data: dict) -> None:
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(self._config.get(k), dict):
                self._config[k].update(v)
            else:
                self._config[k] = v

    @staticmethod
    def _coerce(value: str) -> Any:
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
