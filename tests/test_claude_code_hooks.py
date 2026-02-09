from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from pastchats_memory.cli import _extract_last_assistant_from_transcript, cmd_hook_stop
from pastchats_memory.store import MemoryStore
from pastchats_memory.embeddings import LocalHashEmbeddingProvider


def test_extract_last_assistant_from_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "hi"}),
                json.dumps({"role": "assistant", "content": "hello there"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert _extract_last_assistant_from_transcript(str(transcript)) == "hello there"


def test_stop_hook_stores_assistant_turn(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "do x"}),
                json.dumps({"role": "assistant", "content": "did x"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    db = tmp_path / "memory.db"
    store = MemoryStore(db)
    provider = LocalHashEmbeddingProvider(dim=64)
    store.initialize()
    store.ensure_embedding_compatibility(provider)
    store.close()

    payload = {
        "session_id": "s1",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript),
        "hook_event_name": "Stop",
    }

    args = SimpleNamespace(db=str(db), embed_provider="local", openai_model=None, sqlite_vec=None)
    old_stdin = sys.stdin
    try:
        sys.stdin = io.StringIO(json.dumps(payload))
        assert cmd_hook_stop(args) == 0
    finally:
        sys.stdin = old_stdin

    store2 = MemoryStore(db)
    store2.initialize()
    row = store2.conn.execute(
        "SELECT role, content FROM prompts WHERE conversation_id = ? ORDER BY turn_index DESC LIMIT 1",
        ("s1",),
    ).fetchone()
    assert row["role"] == "assistant"
    assert row["content"] == "did x"
    store2.close()

