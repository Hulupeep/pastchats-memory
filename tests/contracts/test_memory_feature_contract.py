from __future__ import annotations

from pathlib import Path

from pastchats_memory.embeddings import LocalHashEmbeddingProvider
from pastchats_memory.models import PromptTurn
from pastchats_memory.search import hybrid_search
from pastchats_memory.store import MemoryStore


ROOT = Path(__file__).resolve().parents[2]


def test_mem_001_each_prompt_has_embedding(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.db"
    store = MemoryStore(db_path)
    provider = LocalHashEmbeddingProvider(dim=64)

    store.initialize()
    store.ensure_embedding_compatibility(provider)

    turns = [
        PromptTurn(
            source_path="/tmp/chat.json",
            source_project="alpha",
            conversation_id="c1",
            turn_index=0,
            role="user",
            content="How do I implement retries safely?",
        ),
        PromptTurn(
            source_path="/tmp/chat.json",
            source_project="alpha",
            conversation_id="c1",
            turn_index=1,
            role="assistant",
            content="Use idempotency keys and bounded exponential backoff.",
        ),
    ]

    store.upsert_turns(turns, provider)

    counts = store.conn.execute(
        "SELECT (SELECT COUNT(*) FROM prompts) AS prompts,"
        "       (SELECT COUNT(*) FROM prompt_embeddings) AS embeds"
    ).fetchone()

    assert counts["prompts"] == counts["embeds"], (
        "CONTRACT VIOLATION: MEM-001\n"
        "Rule: Every indexed prompt MUST have an embedding row\n"
        "See: docs/contracts/feature_prompt_memory.yml"
    )
    store.close()


def test_mem_002_hybrid_ranking_combines_lexical_and_semantic(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.db"
    store = MemoryStore(db_path)
    provider = LocalHashEmbeddingProvider(dim=128)

    store.initialize()
    store.ensure_embedding_compatibility(provider)
    store.upsert_turns(
        [
            PromptTurn(
                source_path="/tmp/chat1.md",
                source_project="alpha",
                conversation_id="x",
                turn_index=0,
                role="user",
                content="SQLite FTS5 exact keyword indexing",
            ),
            PromptTurn(
                source_path="/tmp/chat2.md",
                source_project="alpha",
                conversation_id="y",
                turn_index=0,
                role="assistant",
                content="Semantic retrieval with embeddings helps approximate matches",
            ),
        ],
        provider,
    )

    hits = hybrid_search(store, provider, "keyword indexing", limit=2)
    assert len(hits) == 2
    assert any(hit.lexical_score > 0 for hit in hits)
    assert any(hit.semantic_score > 0 for hit in hits)

    store.close()


def test_mem_003_vector_search_has_fallback_when_sqlite_vec_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "mem.db"
    store = MemoryStore(db_path, sqlite_vec_path="/definitely/missing/vec0.so")
    provider = LocalHashEmbeddingProvider(dim=64)

    store.initialize()
    assert store.maybe_load_sqlite_vec() is False
    store.ensure_embedding_compatibility(provider)

    store.upsert_turns(
        [
            PromptTurn(
                source_path="/tmp/chat.md",
                source_project="alpha",
                conversation_id="z",
                turn_index=0,
                role="user",
                content="persistent memory for agents",
            )
        ],
        provider,
    )

    results = store.semantic_search(provider.embed("agent memory"), limit=3)
    assert results

    store.close()


def test_mem_004_recall_output_contains_source_reference() -> None:
    cli_file = ROOT / "src" / "pastchats_memory" / "cli.py"
    content = cli_file.read_text(encoding="utf-8")
    assert "Source: {hit.source_path}" in content, (
        "CONTRACT VIOLATION: MEM-004\n"
        "Rule: Recall output MUST include source traceability\n"
        "Expected Source marker in recall formatter\n"
        "See: docs/contracts/feature_prompt_memory.yml"
    )
