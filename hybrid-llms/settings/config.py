import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when application configuration is invalid."""


class Config:
    """Loads configuration once and exposes small reusable helpers."""

    def __init__(self, config_file: str = "config.json"):
        self.settings_dir = Path(__file__).resolve().parent
        self.project_root = self.settings_dir.parent
        self.config_path = self.settings_dir / config_file
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            load_dotenv(self.project_root / ".env")
            with self.config_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            self._validate(data)
            return data
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Unable to load configuration: {exc}") from exc

    @staticmethod
    def _validate(data: dict[str, Any]) -> None:
        required = (
            "ollama",
            "gemini",
            "models",
            "routing",
            "evaluation",
            "guardrail",
            "reporting",
            "logging",
            "execution",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ConfigError(f"Missing configuration sections: {missing}")

        review_threshold = float(data["evaluation"]["review_threshold"])
        pass_threshold = float(data["evaluation"]["pass_threshold"])
        if not 0 <= review_threshold <= pass_threshold <= 1:
            raise ConfigError("Evaluation thresholds must satisfy 0 <= review <= pass <= 1")

        if int(data["execution"].get("file_concurrency", 1)) < 1:
            raise ConfigError("file_concurrency must be >= 1")

    def get(self, *keys: str, default: Any = None) -> Any:
        value: Any = self._data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def gemini_api_key(self) -> str:
        env_name = self.get("gemini", "api_key_env", default="GEMINI_API_KEY")
        return os.getenv(env_name, "").strip()

    def gemini_model(self) -> str:
        env_name = self.get("gemini", "model_env", default="GEMINI_MODEL")
        return os.getenv(env_name, self.get("models", "general"))

    def version(self) -> str:
        return self.get("metadata", "config_version", default="project3-v2")
