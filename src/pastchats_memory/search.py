from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .embeddings import EmbeddingProvider
from .models import SearchHit
from .store import MemoryStore


def hybrid_search(
    store: MemoryStore,
    provider: EmbeddingProvider,
    query: str,
    *,
    limit: int = 8,
    lexical_k: int = 40,
    semantic_k: int = 40,
    semantic_weight: float = 0.6,
) -> list[SearchHit]:
    lexical_rows = store.lexical_search(query, lexical_k)
    semantic_rows = store.semantic_search(provider.embed(query), semantic_k)

    buckets: dict[int, SearchHit] = {}

    for rank, row in enumerate(lexical_rows):
        prompt_id = int(row["id"])
        lexical_score = 1.0 / (rank + 1)
        buckets[prompt_id] = SearchHit(
            prompt_id=prompt_id,
            source_project=str(row["source_project"]),
            source_path=str(row["source_path"]),
            conversation_id=str(row["conversation_id"]),
            turn_index=int(row["turn_index"]),
            role=str(row["role"]),
            content=str(row["content"]),
            timestamp=str(row["timestamp"]) if row["timestamp"] is not None else None,
            lexical_score=lexical_score,
            semantic_score=0.0,
            hybrid_score=0.0,
        )

    for rank, (prompt_id, similarity) in enumerate(semantic_rows):
        semantic_score = max(0.0, float(similarity))
        existing = buckets.get(prompt_id)
        if existing:
            existing.semantic_score = max(existing.semantic_score, semantic_score)
            continue

        row = store.prompt_by_id(prompt_id)
        if row is None:
            continue

        buckets[prompt_id] = SearchHit(
            prompt_id=prompt_id,
            source_project=str(row["source_project"]),
            source_path=str(row["source_path"]),
            conversation_id=str(row["conversation_id"]),
            turn_index=int(row["turn_index"]),
            role=str(row["role"]),
            content=str(row["content"]),
            timestamp=str(row["timestamp"]) if row["timestamp"] is not None else None,
            lexical_score=0.0,
            semantic_score=semantic_score,
            hybrid_score=0.0,
        )

    lexical_weight = 1.0 - semantic_weight
    hits = list(buckets.values())
    for hit in hits:
        hit.hybrid_score = (semantic_weight * hit.semantic_score) + (
            lexical_weight * hit.lexical_score
        )
        hit.next_assistant = store.next_assistant_for(hit.conversation_id, hit.turn_index)

    hits.sort(key=lambda item: item.hybrid_score, reverse=True)
    return hits[:limit]


def hits_as_json(hits: list[SearchHit]) -> list[dict[str, Any]]:
    return [asdict(hit) for hit in hits]
