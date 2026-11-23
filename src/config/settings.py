from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """How the service runs (DESIGN.md section 9). Set each one as SANDBOX_<NAME> in the environment or in .env.

    What a chain may do with the data is not set here: that is the owner's policy, in `policy_file`.
    """

    model_config = SettingsConfigDict(env_prefix="SANDBOX_", env_file=".env", extra="ignore")

    app_name: str = "Code-on-demand sandbox"
    log_level: str = "INFO"

    # policy: switched off for now; the code stays (src/policy/), and true switches it back on
    policy_enabled: bool = False
    policy_file: Path = Path("policy.json")

    # the trained models (ML operators)
    model_dir: Path = Path("models")

    # sources
    fetch_timeout_seconds: float = 10.0
    max_response_bytes: int = 64 * 1024 * 1024
    max_records_per_source: int = 200_000
    record_time_fields: list[str] = ["ts", "timestamp", "time", "date"]  # the first one holding ISO text becomes ts

    # expressions
    max_expression_length: int = 500


@lru_cache
def get_settings() -> Settings:
    """Reads the settings once; every later call returns the same object."""
    return Settings()
