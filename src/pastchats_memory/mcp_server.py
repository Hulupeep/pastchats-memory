from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_settings
from .embeddings import LocalHashEmbeddingProvider, OpenAIEmbeddingProvider
from .search import hits_as_json, hybrid_search
from .store import MemoryStore


def _get_provider(store: MemoryStore, settings, embed_provider: str, openai_model: str | None):
    if embed_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embed_provider=openai")
        model = openai_model or store.get_setting("embedding_model") or settings.openai_model
        return OpenAIEmbeddingProvider(model_name=model, api_key=settings.openai_api_key)

    # default: local hash embeddings (fast, offline)
    existing_dim = store.get_setting("embedding_dim")
    dim = int(existing_dim) if existing_dim else settings.local_embedding_dim
    return LocalHashEmbeddingProvider(dim=dim)


def build_mcp(db_path: str | None = None, sqlite_vec_path: str | None = None):
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "MCP server requires optional dependency. Install: pip install -e '.[mcp]'"
        ) from exc

    mcp = FastMCP("pastchats-memory")

    @mcp.tool()
    def recall(
        query: str,
        limit: int = 5,
        db: str | None = None,
        embed_provider: str = "local",
        openai_model: str | None = None,
    ) -> str:
        """
        Recall prior prompt lessons as a compact, readable block.

        This is the primary pre-task tool. Use once per task.
        """
        settings = load_settings(db or db_path)
        store = MemoryStore(settings.db_path, sqlite_vec_path=sqlite_vec_path or settings.sqlite_vec_path)
        try:
            store.initialize()
            store.maybe_load_sqlite_vec()
            provider = _get_provider(store, settings, embed_provider, openai_model)
            if store.get_setting("embedding_model") is None:
                return "No memory indexed yet. Run index first."

            hits = hybrid_search(store, provider, query, limit=limit)
            lines: list[str] = ["# Memory Recall", f"Query: {query}", ""]
            for idx, hit in enumerate(hits, start=1):
                lines.append(f"## Memory {idx} [{hit.hybrid_score:.3f}] - {hit.source_project}")
                lines.append(f"Prompt: {hit.content.strip()}")
                if hit.next_assistant:
                    answer = hit.next_assistant.strip()
                    if len(answer) > 600:
                        answer = answer[:597] + "..."
                    lines.append(f"What worked: {answer}")
                lines.append(f"Source: {hit.source_path} (turn {hit.turn_index})")
                lines.append("")
            return "\n".join(lines).rstrip() + "\n"
        finally:
            store.close()

    @mcp.tool()
    def search(
        query: str,
        limit: int = 8,
        db: str | None = None,
        embed_provider: str = "local",
        openai_model: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid keyword + semantic search. Returns structured hits."""
        settings = load_settings(db or db_path)
        store = MemoryStore(settings.db_path, sqlite_vec_path=sqlite_vec_path or settings.sqlite_vec_path)
        try:
            store.initialize()
            store.maybe_load_sqlite_vec()
            provider = _get_provider(store, settings, embed_provider, openai_model)
            if store.get_setting("embedding_model") is None:
                return []
            hits = hybrid_search(store, provider, query, limit=limit)
            return hits_as_json(hits)
        finally:
            store.close()

    @mcp.tool()
    def store_turn(
        conversation_id: str,
        role: str,
        content: str,
        source_project: str = "live",
        source_path: str = "live://mcp",
        timestamp: str | None = None,
        event_id: str | None = None,
        turn_index: int | None = None,
        metadata_json: str | None = None,
        db: str | None = None,
        embed_provider: str = "local",
        openai_model: str | None = None,
    ) -> dict[str, Any]:
        """
        Store one message turn directly into the memory DB.

        Use this for true \"always-on\" memory. Call it from a hook after each user/assistant turn.

        Idempotency:
        - Pass event_id (recommended) so re-sending the same turn is safe.
        """
        settings = load_settings(db or db_path)
        store = MemoryStore(settings.db_path, sqlite_vec_path=sqlite_vec_path or settings.sqlite_vec_path)
        try:
            store.initialize()
            store.maybe_load_sqlite_vec()
            provider = _get_provider(store, settings, embed_provider, openai_model)
            store.ensure_embedding_compatibility(provider)
            metadata = {}
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    metadata = {"raw_metadata": metadata_json}
            result = store.store_turn(
                source_path=source_path,
                source_project=source_project,
                conversation_id=conversation_id,
                role=role,
                content=content,
                provider=provider,
                timestamp=timestamp,
                metadata=metadata,
                turn_index=turn_index,
                event_id=event_id,
            )
            return {"ok": True, **result}
        finally:
            store.close()

    @mcp.tool()
    def index_paths(
        inputs_json: str,
        roles: str = "user,assistant,system,tool",
        db: str | None = None,
        embed_provider: str = "local",
        openai_model: str | None = None,
    ) -> dict[str, Any]:
        """
        Run file-based indexing on a set of inputs.

        inputs_json: JSON array of file/dir paths.
        """
        from .parsers import discover_history_files, infer_project, parse_history_file, project_roots

        inputs = json.loads(inputs_json)
        if not isinstance(inputs, list):
            raise RuntimeError("inputs_json must be a JSON list of paths")

        settings = load_settings(db or db_path)
        store = MemoryStore(settings.db_path, sqlite_vec_path=sqlite_vec_path or settings.sqlite_vec_path)
        try:
            store.initialize()
            store.maybe_load_sqlite_vec()
            provider = _get_provider(store, settings, embed_provider, openai_model)
            store.ensure_embedding_compatibility(provider)

            files = discover_history_files([str(x) for x in inputs])
            roots = project_roots([str(x) for x in inputs])
            allowed_roles = {r.strip().lower() for r in roles.split(",") if r.strip()}

            indexed_total = 0
            for file_path in files:
                project = infer_project(file_path, roots)
                turns = parse_history_file(file_path, project)
                turns = [turn for turn in turns if turn.role in allowed_roles]
                indexed_total += store.upsert_turns(turns, provider)

            return {"ok": True, "files_scanned": len(files), "prompts_indexed": indexed_total}
        finally:
            store.close()

    return mcp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pastchats-memory-mcp")
    parser.add_argument("--db", default=None, help="SQLite DB path (default: env PROMPT_MEMORY_DB or .swarm/prompt_memory.db)")
    parser.add_argument("--sqlite-vec", default=None, help="Path to sqlite-vec extension (optional)")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "streamable-http"],
        help="MCP transport. stdio is the most compatible default.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for streamable-http transport")
    parser.add_argument("--port", type=int, default=8765, help="Port for streamable-http transport")
    args = parser.parse_args(argv)

    mcp = build_mcp(db_path=args.db, sqlite_vec_path=args.sqlite_vec)

    if args.transport == "streamable-http":
        # Requires mcp[cli] extras. Exposes an HTTP endpoint for MCP clients.
        mcp.run(transport="streamable-http", host=args.host, port=args.port)  # type: ignore[attr-defined]
        return 0

    mcp.run()  # stdio default  # type: ignore[attr-defined]
    return 0

