from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    db_path: Path
    sqlite_vec_path: str | None
    openai_api_key: str | None
    openai_model: str
    local_embedding_dim: int


def load_settings(db_path: str | Path | None = None) -> Settings:
    resolved_db = Path(db_path or os.getenv("PROMPT_MEMORY_DB", ".swarm/prompt_memory.db"))
    return Settings(
        db_path=resolved_db,
        sqlite_vec_path=os.getenv("PROMPT_MEMORY_SQLITE_VEC_PATH"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("PROMPT_MEMORY_OPENAI_MODEL", "text-embedding-3-small"),
        local_embedding_dim=int(os.getenv("PROMPT_MEMORY_EMBED_DIM", "256")),
    )
