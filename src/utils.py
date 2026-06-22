"""Small helpers for paths, reading sample files, and saving outputs."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Project folders (used by notebooks and helper modules)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_TEXTS_DIR = DATA_DIR / "sample_texts"
OUTPUTS_DIR = DATA_DIR / "outputs"
PROMPT_LOGS_DIR = OUTPUTS_DIR / "prompt_logs"
APP_TRANSCRIPTS_DIR = OUTPUTS_DIR / "app_transcripts"
GENERATED_IMAGES_DIR = OUTPUTS_DIR / "generated_images"


def ensure_output_dirs() -> None:
    """Create output folders used by notebooks 2 and 3."""
    for folder in (PROMPT_LOGS_DIR, APP_TRANSCRIPTS_DIR, GENERATED_IMAGES_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    """Load API keys from the .env file in the project root."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def read_sample_text(filename: str) -> str:
    """Read teaching material from data/sample_texts/."""
    path = SAMPLE_TEXTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return path.read_text(encoding="utf-8")


def save_text_output(content: str, folder: Path, prefix: str, extension: str = "txt") -> Path:
    """Save notebook output to data/outputs/ with a date stamp."""
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{prefix}_{stamp}.{extension}"
    path.write_text(content, encoding="utf-8")
    return path


def get_env(key: str, default: str | None = None) -> str | None:
    load_env()
    return os.getenv(key, default)
