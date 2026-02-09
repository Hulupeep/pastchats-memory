from __future__ import annotations

from pathlib import Path

from pastchats_memory.embeddings import LocalHashEmbeddingProvider
from pastchats_memory.store import MemoryStore


def test_store_turn_appends_and_dedupes_by_event_id(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    store = MemoryStore(db)
    provider = LocalHashEmbeddingProvider(dim=64)
    store.initialize()
    store.ensure_embedding_compatibility(provider)

    r1 = store.store_turn(
        source_path="live://test",
        source_project="proj",
        conversation_id="c1",
        role="user",
        content="hello",
        provider=provider,
        event_id="evt-1",
    )
    r2 = store.store_turn(
        source_path="live://test",
        source_project="proj",
        conversation_id="c1",
        role="user",
        content="hello",
        provider=provider,
        event_id="evt-1",
    )

    assert r1["inserted"] is True
    assert r2["inserted"] is False
    assert r1["prompt_id"] == r2["prompt_id"]

    # Turn index should auto-increment when no explicit index is provided.
    r3 = store.store_turn(
        source_path="live://test",
        source_project="proj",
        conversation_id="c1",
        role="assistant",
        content="world",
        provider=provider,
        event_id="evt-2",
    )
    assert r3["inserted"] is True
    row = store.conn.execute(
        "SELECT turn_index FROM prompts WHERE id = ?",
        (int(r3["prompt_id"]),),
    ).fetchone()
    assert int(row["turn_index"]) == 1

    store.close()

