# PastChats Memory

Persistent prompt memory for agent workflows. Index conversation history into SQLite, then run hybrid keyword + semantic search before starting new tasks.

## Docs

- Help site: https://hulupeep.github.io/pastchats-memory-help/
- Help repo: https://github.com/Hulupeep/pastchats-memory-help

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

pastchats-memory init --db .swarm/prompt_memory.db
pastchats-memory index --db .swarm/prompt_memory.db --input ~/projects
pastchats-memory recall --db .swarm/prompt_memory.db --query "build webhook retry with idempotency"
```

## Optional sqlite-vec

If `sqlite-vec` is installed, set:

```bash
export PROMPT_MEMORY_SQLITE_VEC_PATH=/path/to/vec0.so
```

The CLI will auto-enable ANN search through `vec0`; otherwise it falls back to in-process cosine search over stored vectors.

## MCP server (always-on memory)

This repo ships an MCP server wrapper that exposes tools:

- `recall(query, limit, ...)`
- `search(query, limit, ...)`
- `store_turn(conversation_id, role, content, event_id, ...)` (for live capture)
- `index_paths(inputs_json, ...)`

Install with MCP extras:

```bash
pip install -e .[mcp]
```

Run (stdio transport):

```bash
pastchats-memory-mcp --db .swarm/prompt_memory.db
```
