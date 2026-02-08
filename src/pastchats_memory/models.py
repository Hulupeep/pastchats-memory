from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PromptTurn:
    source_path: str
    source_project: str
    conversation_id: str
    turn_index: int
    role: str
    content: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    prompt_id: int
    source_project: str
    source_path: str
    conversation_id: str
    turn_index: int
    role: str
    content: str
    timestamp: str | None
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0
    next_assistant: str | None = None
