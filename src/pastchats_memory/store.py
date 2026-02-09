from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable

from .embeddings import EmbeddingProvider, cosine_similarity
from .models import PromptTurn


class MemoryStore:
    def __init__(self, db_path: Path, sqlite_vec_path: str | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_vec_path = sqlite_vec_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.sqlite_vec_loaded = False
        self.sqlite_vec_ready = False

    def close(self) -> None:
        self.conn.close()

    def initialize(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        sql = schema_path.read_text(encoding="utf-8")
        self.conn.executescript(sql)
        self.conn.commit()

    def maybe_load_sqlite_vec(self) -> bool:
        if not self.sqlite_vec_path:
            return False
        if self.sqlite_vec_loaded:
            return self.sqlite_vec_ready

        self.sqlite_vec_loaded = True
        try:
            self.conn.enable_load_extension(True)
            self.conn.load_extension(self.sqlite_vec_path)
            self.conn.execute("SELECT vec_version()")
            self.sqlite_vec_ready = True
        except sqlite3.Error:
            self.sqlite_vec_ready = False
        finally:
            try:
                self.conn.enable_load_extension(False)
            except sqlite3.Error:
                pass
        return self.sqlite_vec_ready

    def ensure_vec_table(self, dim: int) -> bool:
        if not self.sqlite_vec_ready:
            return False
        try:
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS prompt_vec USING vec0(embedding float[{dim}]);"
            )
            self.conn.commit()
            return True
        except sqlite3.Error:
            return False

    def create_index_run(self, inputs: list[str]) -> int:
        row = self.conn.execute(
            """
            INSERT INTO index_runs(started_at, input_roots_json)
            VALUES(datetime('now'), ?)
            """,
            (json.dumps(inputs),),
        )
        self.conn.commit()
        return int(row.lastrowid)

    def finish_index_run(
        self,
        run_id: int,
        *,
        files_scanned: int,
        prompts_indexed: int,
        notes: str = "",
    ) -> None:
        self.conn.execute(
            """
            UPDATE index_runs
            SET finished_at = datetime('now'),
                files_scanned = ?,
                prompts_indexed = ?,
                notes = ?
            WHERE id = ?
            """,
            (files_scanned, prompts_indexed, notes, run_id),
        )
        self.conn.commit()

    def get_setting(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM memory_settings WHERE key = ?",
            (key,),
        ).fetchone()
        if not row:
            return None
        return str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO memory_settings(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self.conn.commit()

    def ensure_embedding_compatibility(self, provider: EmbeddingProvider) -> None:
        existing_model = self.get_setting("embedding_model")
        existing_dim = self.get_setting("embedding_dim")
        if existing_model and existing_model != provider.model_name:
            raise RuntimeError(
                f"Embedding model mismatch: DB uses {existing_model}, requested {provider.model_name}."
            )

        if existing_dim and int(existing_dim) != provider.dim:
            raise RuntimeError(
                f"Embedding dim mismatch: DB uses {existing_dim}, requested {provider.dim}."
            )

        self.set_setting("embedding_model", provider.model_name)
        self.set_setting("embedding_dim", str(provider.dim))

    def upsert_turns(self, turns: Iterable[PromptTurn], provider: EmbeddingProvider) -> int:
        inserted = 0
        cursor = self.conn.cursor()
        vector_table_ready = self.ensure_vec_table(provider.dim)

        for turn in turns:
            content = turn.content.strip()
            if not content:
                continue

            digest = hashlib.sha256(
                "|".join(
                    [
                        turn.source_path,
                        turn.conversation_id,
                        str(turn.turn_index),
                        turn.role,
                        content,
                    ]
                ).encode("utf-8")
            ).hexdigest()

            cursor.execute(
                """
                INSERT OR IGNORE INTO prompts(
                  source_path, source_project, conversation_id, turn_index,
                  role, content, timestamp, content_hash, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn.source_path,
                    turn.source_project,
                    turn.conversation_id,
                    turn.turn_index,
                    turn.role,
                    content,
                    turn.timestamp,
                    digest,
                    json.dumps(turn.metadata, sort_keys=True),
                ),
            )
            was_inserted = cursor.rowcount == 1

            row = cursor.execute(
                "SELECT id FROM prompts WHERE content_hash = ?",
                (digest,),
            ).fetchone()
            if row is None:
                continue
            prompt_id = int(row["id"])
            if was_inserted:
                inserted += 1

            vector = provider.embed(content)
            cursor.execute(
                """
                INSERT INTO prompt_embeddings(prompt_id, model, dim, vector_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(prompt_id) DO UPDATE SET
                  model = excluded.model,
                  dim = excluded.dim,
                  vector_json = excluded.vector_json
                """,
                (prompt_id, provider.model_name, provider.dim, json.dumps(vector)),
            )

            if vector_table_ready:
                try:
                    cursor.execute(
                        "INSERT OR REPLACE INTO prompt_vec(rowid, embedding) VALUES (?, ?)",
                        (prompt_id, json.dumps(vector)),
                    )
                except sqlite3.Error:
                    vector_table_ready = False

        self.conn.commit()
        return inserted

    def next_turn_index(self, conversation_id: str) -> int:
        row = self.conn.execute(
            """
            SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_index
            FROM prompts
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        return int(row["next_index"]) if row else 0

    def store_turn(
        self,
        *,
        source_path: str,
        source_project: str,
        conversation_id: str,
        role: str,
        content: str,
        provider: EmbeddingProvider,
        timestamp: str | None = None,
        metadata: dict | None = None,
        turn_index: int | None = None,
        event_id: str | None = None,
    ) -> dict[str, int | bool]:
        """
        Store one prompt turn directly (for hooks/MCP/live capture).

        Idempotency options:
        - Provide `event_id` for stable dedupe even if you re-send the same turn.
        - Otherwise, content is deduped per (source_path, conversation_id, turn_index, role, content).
        """
        content = content.strip()
        if not content:
            return {"inserted": False, "prompt_id": 0}

        if turn_index is None:
            turn_index = self.next_turn_index(conversation_id)

        digest_parts = [
            source_path,
            conversation_id,
            str(event_id) if event_id is not None else str(turn_index),
            role,
            content,
        ]
        digest = hashlib.sha256("|".join(digest_parts).encode("utf-8")).hexdigest()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO prompts(
              source_path, source_project, conversation_id, turn_index,
              role, content, timestamp, content_hash, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_path,
                source_project,
                conversation_id,
                int(turn_index),
                role,
                content,
                timestamp,
                digest,
                json.dumps(metadata or {}, sort_keys=True),
            ),
        )
        was_inserted = cursor.rowcount == 1

        row = cursor.execute(
            "SELECT id FROM prompts WHERE content_hash = ?",
            (digest,),
        ).fetchone()
        if row is None:
            self.conn.commit()
            return {"inserted": False, "prompt_id": 0}

        prompt_id = int(row["id"])

        vector = provider.embed(content)
        cursor.execute(
            """
            INSERT INTO prompt_embeddings(prompt_id, model, dim, vector_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(prompt_id) DO UPDATE SET
              model = excluded.model,
              dim = excluded.dim,
              vector_json = excluded.vector_json
            """,
            (prompt_id, provider.model_name, provider.dim, json.dumps(vector)),
        )

        vector_table_ready = self.ensure_vec_table(provider.dim)
        if vector_table_ready:
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO prompt_vec(rowid, embedding) VALUES (?, ?)",
                    (prompt_id, json.dumps(vector)),
                )
            except sqlite3.Error:
                pass

        self.conn.commit()
        return {"inserted": bool(was_inserted), "prompt_id": int(prompt_id)}

    def lexical_search(self, query: str, limit: int) -> list[sqlite3.Row]:
        sql = """
            SELECT p.id, p.source_project, p.source_path, p.conversation_id,
                   p.turn_index, p.role, p.content, p.timestamp,
                   bm25(prompts_fts) AS bm25_score
            FROM prompts_fts
            JOIN prompts p ON p.id = prompts_fts.rowid
            WHERE prompts_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
        """

        try:
            rows = self.conn.execute(sql, (query, limit)).fetchall()
            return list(rows)
        except sqlite3.Error:
            cleaned = " ".join(re.findall(r"[a-zA-Z0-9_]+", query))
            if not cleaned:
                return []
            rows = self.conn.execute(sql, (cleaned, limit)).fetchall()
            return list(rows)

    def semantic_search(self, query_vector: list[float], limit: int) -> list[tuple[int, float]]:
        # Path A: sqlite-vec ANN search.
        if self.sqlite_vec_ready and self.ensure_vec_table(len(query_vector)):
            try:
                rows = self.conn.execute(
                    """
                    SELECT rowid AS prompt_id, distance
                    FROM prompt_vec
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance ASC
                    """,
                    (json.dumps(query_vector), limit),
                ).fetchall()
                return [(int(row["prompt_id"]), 1.0 / (1.0 + float(row["distance"]))) for row in rows]
            except sqlite3.Error:
                pass

        # Path B: fallback cosine search over stored vectors.
        rows = self.conn.execute(
            "SELECT prompt_id, vector_json FROM prompt_embeddings"
        ).fetchall()
        scored: list[tuple[int, float]] = []
        for row in rows:
            vector = json.loads(row["vector_json"])
            similarity = cosine_similarity(query_vector, vector)
            scored.append((int(row["prompt_id"]), float(similarity)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def prompt_by_id(self, prompt_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT id, source_project, source_path, conversation_id,
                   turn_index, role, content, timestamp
            FROM prompts
            WHERE id = ?
            """,
            (prompt_id,),
        ).fetchone()

    def next_assistant_for(self, conversation_id: str, turn_index: int) -> str | None:
        row = self.conn.execute(
            """
            SELECT content
            FROM prompts
            WHERE conversation_id = ?
              AND turn_index > ?
              AND role = 'assistant'
            ORDER BY turn_index ASC
            LIMIT 1
            """,
            (conversation_id, turn_index),
        ).fetchone()
        if not row:
            return None
        return str(row["content"])

    def stats(self) -> dict[str, int | str | None]:
        prompt_count = self.conn.execute("SELECT COUNT(*) FROM prompts").fetchone()[0]
        emb_count = self.conn.execute("SELECT COUNT(*) FROM prompt_embeddings").fetchone()[0]
        project_count = self.conn.execute(
            "SELECT COUNT(DISTINCT source_project) FROM prompts"
        ).fetchone()[0]
        latest_index = self.conn.execute(
            "SELECT finished_at FROM index_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return {
            "prompts": int(prompt_count),
            "embeddings": int(emb_count),
            "projects": int(project_count),
            "latest_index_finished_at": latest_index[0] if latest_index else None,
        }
