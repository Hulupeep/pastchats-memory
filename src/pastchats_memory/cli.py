from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import hashlib

from .config import load_settings
from .embeddings import (
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from .parsers import discover_history_files, infer_project, parse_history_file, project_roots
from .search import hits_as_json, hybrid_search
from .store import MemoryStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pastchats-memory")
    sub = parser.add_subparsers(dest="command", required=True)

    init_cmd = sub.add_parser("init", help="Initialize memory database")
    init_cmd.add_argument("--db", default=None, help="SQLite DB path")
    init_cmd.add_argument("--sqlite-vec", default=None, help="Path to sqlite-vec extension")

    index_cmd = sub.add_parser("index", help="Index conversation history")
    index_cmd.add_argument("--db", default=None, help="SQLite DB path")
    index_cmd.add_argument("--input", nargs="+", required=True, help="Files or directories to scan")
    index_cmd.add_argument(
        "--roles",
        default="user,assistant,system,tool",
        help="Comma-separated roles to index",
    )
    index_cmd.add_argument(
        "--embed-provider",
        choices=["auto", "local", "openai"],
        default="auto",
    )
    index_cmd.add_argument("--openai-model", default=None)
    index_cmd.add_argument("--sqlite-vec", default=None)

    search_cmd = sub.add_parser("search", help="Run hybrid search")
    search_cmd.add_argument("--db", default=None, help="SQLite DB path")
    search_cmd.add_argument("--query", required=True)
    search_cmd.add_argument("--limit", type=int, default=8)
    search_cmd.add_argument(
        "--embed-provider",
        choices=["auto", "local", "openai"],
        default="auto",
    )
    search_cmd.add_argument("--openai-model", default=None)
    search_cmd.add_argument("--json", action="store_true")
    search_cmd.add_argument("--sqlite-vec", default=None)

    recall_cmd = sub.add_parser("recall", help="Search and print memory context for pre-task priming")
    recall_cmd.add_argument("--db", default=None, help="SQLite DB path")
    recall_cmd.add_argument("--query", required=True)
    recall_cmd.add_argument("--limit", type=int, default=5)
    recall_cmd.add_argument(
        "--embed-provider",
        choices=["auto", "local", "openai"],
        default="auto",
    )
    recall_cmd.add_argument("--openai-model", default=None)
    recall_cmd.add_argument("--json", action="store_true")
    recall_cmd.add_argument("--sqlite-vec", default=None)

    stats_cmd = sub.add_parser("stats", help="Show DB stats")
    stats_cmd.add_argument("--db", default=None, help="SQLite DB path")

    hook_user = sub.add_parser(
        "hook-user-prompt-submit",
        help="Claude Code hook: inject memory lessons on UserPromptSubmit (reads JSON from stdin)",
    )
    hook_user.add_argument("--db", default=None, help="SQLite DB path")
    hook_user.add_argument("--limit", type=int, default=3)
    hook_user.add_argument(
        "--embed-provider",
        choices=["auto", "local", "openai"],
        default="auto",
    )
    hook_user.add_argument("--openai-model", default=None)
    hook_user.add_argument("--sqlite-vec", default=None)

    hook_stop = sub.add_parser(
        "hook-stop",
        help="Claude Code hook: store latest assistant turn on Stop (reads JSON from stdin)",
    )
    hook_stop.add_argument("--db", default=None, help="SQLite DB path")
    hook_stop.add_argument(
        "--embed-provider",
        choices=["auto", "local", "openai"],
        default="auto",
    )
    hook_stop.add_argument("--openai-model", default=None)
    hook_stop.add_argument("--sqlite-vec", default=None)
    return parser


def select_provider(
    store: MemoryStore,
    mode: str,
    settings,
    openai_model_override: str | None,
) -> EmbeddingProvider:
    existing_model = store.get_setting("embedding_model")
    existing_dim = store.get_setting("embedding_dim")

    if mode == "local":
        dim = int(existing_dim) if existing_dim else settings.local_embedding_dim
        return LocalHashEmbeddingProvider(dim=dim)

    if mode == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when --embed-provider=openai")
        model = openai_model_override or existing_model or settings.openai_model
        return OpenAIEmbeddingProvider(model_name=model, api_key=settings.openai_api_key)

    # auto mode
    if existing_model:
        if existing_model.startswith("local-hash-"):
            dim = int(existing_dim) if existing_dim else settings.local_embedding_dim
            return LocalHashEmbeddingProvider(dim=dim)

        if settings.openai_api_key:
            return OpenAIEmbeddingProvider(
                model_name=existing_model,
                api_key=settings.openai_api_key,
            )

        # Existing OpenAI model but no API key; fallback to local provider.
        dim = int(existing_dim) if existing_dim else settings.local_embedding_dim
        return LocalHashEmbeddingProvider(dim=dim)

    # Fresh DB defaults to local for zero-config usage.
    return LocalHashEmbeddingProvider(dim=settings.local_embedding_dim)


def cmd_init(args: argparse.Namespace) -> int:
    settings = load_settings(args.db)
    sqlite_vec_path = args.sqlite_vec or settings.sqlite_vec_path
    store = MemoryStore(settings.db_path, sqlite_vec_path=sqlite_vec_path)
    try:
        store.initialize()
        vec_enabled = store.maybe_load_sqlite_vec()
        print(
            json.dumps(
                {
                    "db": str(settings.db_path),
                    "initialized": True,
                    "sqlite_vec_loaded": vec_enabled,
                },
                indent=2,
            )
        )
        return 0
    finally:
        store.close()


def cmd_index(args: argparse.Namespace) -> int:
    settings = load_settings(args.db)
    sqlite_vec_path = args.sqlite_vec or settings.sqlite_vec_path
    store = MemoryStore(settings.db_path, sqlite_vec_path=sqlite_vec_path)
    try:
        store.initialize()
        store.maybe_load_sqlite_vec()
        provider = select_provider(store, args.embed_provider, settings, args.openai_model)
        store.ensure_embedding_compatibility(provider)

        files = discover_history_files(args.input)
        roots = project_roots(args.input)
        allowed_roles = {r.strip().lower() for r in args.roles.split(",") if r.strip()}

        run_id = store.create_index_run(args.input)
        indexed_total = 0
        for file_path in files:
            project = infer_project(file_path, roots)
            turns = parse_history_file(file_path, project)
            turns = [turn for turn in turns if turn.role in allowed_roles]
            indexed_total += store.upsert_turns(turns, provider)

        notes = (
            "sqlite-vec enabled"
            if store.sqlite_vec_ready
            else "sqlite-vec not available; using fallback cosine"
        )
        store.finish_index_run(
            run_id,
            files_scanned=len(files),
            prompts_indexed=indexed_total,
            notes=notes,
        )

        print(
            json.dumps(
                {
                    "db": str(settings.db_path),
                    "files_scanned": len(files),
                    "prompts_indexed": indexed_total,
                    "embedding_model": provider.model_name,
                    "embedding_dim": provider.dim,
                    "sqlite_vec_loaded": store.sqlite_vec_ready,
                },
                indent=2,
            )
        )
        return 0
    finally:
        store.close()


def _run_query(args: argparse.Namespace, command: str) -> int:
    settings = load_settings(args.db)
    sqlite_vec_path = getattr(args, "sqlite_vec", None) or settings.sqlite_vec_path
    store = MemoryStore(settings.db_path, sqlite_vec_path=sqlite_vec_path)
    try:
        store.initialize()
        store.maybe_load_sqlite_vec()

        provider = select_provider(store, args.embed_provider, settings, args.openai_model)
        if store.get_setting("embedding_model") is None:
            raise RuntimeError("Database has no indexed memories yet. Run `pastchats-memory index ...` first.")

        hits = hybrid_search(store, provider, args.query, limit=args.limit)

        if getattr(args, "json", False):
            print(json.dumps(hits_as_json(hits), indent=2))
            return 0

        if command == "search":
            for idx, hit in enumerate(hits, start=1):
                snippet = hit.content.replace("\n", " ").strip()
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                print(
                    f"{idx}. [{hit.hybrid_score:.3f}] {hit.source_project}"
                    f" | {hit.role} | {hit.source_path}#{hit.turn_index}\n"
                    f"   {snippet}"
                )
            return 0

        # recall formatter
        print("# Memory Recall")
        print(f"Query: {args.query}\n")
        for idx, hit in enumerate(hits, start=1):
            print(f"## Memory {idx} [{hit.hybrid_score:.3f}] - {hit.source_project}")
            print(f"Prompt: {hit.content.strip()}")
            if hit.next_assistant:
                answer = hit.next_assistant.strip()
                if len(answer) > 600:
                    answer = answer[:597] + "..."
                print(f"What worked: {answer}")
            print(f"Source: {hit.source_path} (turn {hit.turn_index})\n")
        return 0
    finally:
        store.close()


def cmd_stats(args: argparse.Namespace) -> int:
    settings = load_settings(args.db)
    store = MemoryStore(settings.db_path, sqlite_vec_path=settings.sqlite_vec_path)
    try:
        store.initialize()
        print(json.dumps(store.stats(), indent=2))
        return 0
    finally:
        store.close()


def _read_hook_input() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


def cmd_hook_user_prompt_submit(args: argparse.Namespace) -> int:
    payload = _read_hook_input()
    prompt = str(payload.get("prompt") or "").strip()
    if len(prompt) < 20:
        return 0

    session_id = str(payload.get("session_id") or "unknown")
    cwd = str(payload.get("cwd") or "")
    transcript_path = str(payload.get("transcript_path") or "")

    settings = load_settings(args.db)
    sqlite_vec_path = getattr(args, "sqlite_vec", None) or settings.sqlite_vec_path
    store = MemoryStore(settings.db_path, sqlite_vec_path=sqlite_vec_path)
    try:
        store.initialize()
        store.maybe_load_sqlite_vec()
        provider = select_provider(store, args.embed_provider, settings, args.openai_model)
        store.ensure_embedding_compatibility(provider)

        # Store the user prompt itself so the session is captured even if Stop hook isn't installed.
        source_project = Path(cwd).name if cwd else "claude"
        # Stable idempotency across processes (built-in hash() is randomized per process).
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        event_id = f"UserPromptSubmit:{session_id}:{prompt_hash}"
        result = store.store_turn(
            source_path=transcript_path or "live://claude-code",
            source_project=source_project,
            conversation_id=session_id,
            role="user",
            content=prompt,
            provider=provider,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"hook_event": "UserPromptSubmit", "cwd": cwd},
            event_id=event_id,
        )

        if store.get_setting("embedding_model") is None:
            return 0

        hits = hybrid_search(store, provider, prompt, limit=args.limit)
        inserted_id = int(result.get("prompt_id") or 0)
        if inserted_id:
            hits = [h for h in hits if h.prompt_id != inserted_id]

        if not hits:
            return 0

        # For UserPromptSubmit, stdout is injected into context. Keep it short.
        lines: list[str] = ["Memory lessons (from past chats):"]
        for idx, hit in enumerate(hits[: args.limit], start=1):
            what_worked = (hit.next_assistant or "").strip().replace("\n", " ")
            if not what_worked:
                what_worked = hit.content.strip().replace("\n", " ")
            if len(what_worked) > 180:
                what_worked = what_worked[:177] + "..."
            lines.append(f"{idx}) {what_worked} (source: {hit.source_path})")
        print("\n".join(lines))
        return 0
    finally:
        store.close()


def _extract_last_assistant_from_transcript(transcript_path: str) -> str:
    # Transcript is JSONL. Read a tail window to avoid reprocessing huge files.
    path = Path(transcript_path)
    if not path.exists() or not path.is_file():
        return ""

    data = path.read_bytes()
    window = data[-256_000:] if len(data) > 256_000 else data
    text = window.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip().startswith("{")]

    last_text = ""
    for line in lines[::-1]:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        role = str(obj.get("role") or obj.get("author") or obj.get("speaker") or "").lower()
        if role not in {"assistant", "ai", "model"}:
            continue

        content = obj.get("content") or obj.get("text") or obj.get("message")
        if isinstance(content, str) and content.strip():
            last_text = content.strip()
            break
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            joined = "\n".join(p for p in parts if p).strip()
            if joined:
                last_text = joined
                break

    return last_text


def cmd_hook_stop(args: argparse.Namespace) -> int:
    payload = _read_hook_input()
    transcript_path = str(payload.get("transcript_path") or "")
    session_id = str(payload.get("session_id") or "unknown")
    cwd = str(payload.get("cwd") or "")

    assistant_text = _extract_last_assistant_from_transcript(transcript_path)
    if not assistant_text:
        return 0

    settings = load_settings(args.db)
    sqlite_vec_path = getattr(args, "sqlite_vec", None) or settings.sqlite_vec_path
    store = MemoryStore(settings.db_path, sqlite_vec_path=sqlite_vec_path)
    try:
        store.initialize()
        store.maybe_load_sqlite_vec()
        provider = select_provider(store, args.embed_provider, settings, args.openai_model)
        store.ensure_embedding_compatibility(provider)

        source_project = Path(cwd).name if cwd else "claude"
        assistant_hash = hashlib.sha256(assistant_text.encode("utf-8")).hexdigest()[:16]
        event_id = f"Stop:{session_id}:{assistant_hash}"
        store.store_turn(
            source_path=transcript_path or "live://claude-code",
            source_project=source_project,
            conversation_id=session_id,
            role="assistant",
            content=assistant_text,
            provider=provider,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"hook_event": "Stop", "cwd": cwd},
            event_id=event_id,
        )
        return 0
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            return cmd_init(args)
        if args.command == "index":
            return cmd_index(args)
        if args.command == "search":
            return _run_query(args, "search")
        if args.command == "recall":
            return _run_query(args, "recall")
        if args.command == "stats":
            return cmd_stats(args)
        if args.command == "hook-user-prompt-submit":
            return cmd_hook_user_prompt_submit(args)
        if args.command == "hook-stop":
            return cmd_hook_stop(args)
        parser.print_help()
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
