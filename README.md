# PastChats Memory

Persistent prompt memory for agent workflows. Index conversation history into SQLite, then run hybrid keyword + semantic search before starting new tasks.

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
