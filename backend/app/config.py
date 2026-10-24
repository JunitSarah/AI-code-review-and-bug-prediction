import os
from functools import lru_cache


class Settings:
    # Local folder or Hugging Face Hub repo id holding the fine-tuned CodeBERT
    model_path: str = os.getenv("MODEL_PATH", "artifacts/codebert-defect")
    github_token: str | None = os.getenv("GITHUB_TOKEN")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
    cors_origins: list[str] = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")]
    max_files: int = int(os.getenv("MAX_FILES", "15"))
    quantize: bool = os.getenv("QUANTIZE", "1") == "1"  # int8 dynamic quantisation, cuts RAM for free tiers
    risk_threshold: float = float(os.getenv("RISK_THRESHOLD", "0.5"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
