from __future__ import annotations

from pathlib import Path

from pastchats_memory.embeddings import LocalHashEmbeddingProvider
from pastchats_memory.parsers import parse_history_file
from pastchats_memory.search import hybrid_search
from pastchats_memory.store import MemoryStore


def test_index_and_hybrid_search(tmp_path: Path) -> None:
    chat_file = tmp_path / "claude_history.md"
    chat_file.write_text(
        """
User: Build SQLite FTS5 prompt search
Assistant: Use FTS5 for lexical ranking and keep embeddings for semantic recall.
User: How do I combine both?
Assistant: Merge lexical BM25 and cosine scores into a hybrid score.
        """.strip(),
        encoding="utf-8",
    )

    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path)
    provider = LocalHashEmbeddingProvider(dim=128)

    store.initialize()
    store.ensure_embedding_compatibility(provider)
    turns = parse_history_file(chat_file, source_project="demo")

    inserted_first = store.upsert_turns(turns, provider)
    inserted_second = store.upsert_turns(turns, provider)

    assert inserted_first > 0
    assert inserted_second == 0

    hits = hybrid_search(store, provider, "hybrid ranking for fts5 and embeddings", limit=3)
    assert hits
    assert hits[0].source_project == "demo"
    assert "FTS5" in hits[0].content or "hybrid" in hits[0].content.lower()

    store.close()
