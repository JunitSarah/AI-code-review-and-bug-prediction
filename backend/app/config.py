import json
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# lets you keep settings in backend/.env locally; on Render the dashboard env vars are used
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_PROVIDER = os.getenv("LLM_PROVIDER", "openai_compatible").strip().lower()


class Settings:
    # Local folder or Hugging Face Hub repo id holding the fine-tuned CodeBERT
    model_path: str = os.getenv("MODEL_PATH", "artifacts/codebert-defect")
    github_token: str | None = os.getenv("GITHUB_TOKEN")

    # --- optional LLM review ("anthropic" or "openai_compatible", see services/reviewer.py) ---
    llm_provider: str = _PROVIDER
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")
    llm_model: str = os.getenv("LLM_MODEL") or ("openai/gpt-oss-120b" if _PROVIDER == "openai_compatible" else "")

    cors_origins: list[str] = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]
    max_files: int = int(os.getenv("MAX_FILES", "15"))
    quantize: bool = os.getenv("QUANTIZE", "1") == "1"  # int8 dynamic quantisation, cuts RAM for free tiers
    # Explicit override; otherwise threshold.json (written by ml/evaluate.py) next to the model, else 0.5
    risk_threshold_env: str | None = os.getenv("RISK_THRESHOLD")


def load_threshold(model_path: str, env_value: str | None) -> float:
    if env_value:
        return float(env_value)
    f = Path(model_path) / "threshold.json"
    if f.is_file():
        try:
            return float(json.loads(f.read_text())["threshold"])
        except (ValueError, KeyError):
            pass
    return 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()