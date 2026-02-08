#!/usr/bin/env bash
set -euo pipefail

DB_PATH="${PROMPT_MEMORY_DB:-.swarm/prompt_memory.db}"
LIMIT="${PROMPT_MEMORY_LIMIT:-5}"

if [ "$#" -lt 1 ]; then
  echo "usage: pre_task_recall.sh \"<task intent query>\"" >&2
  exit 2
fi

QUERY="$1"

pastchats-memory recall \
  --db "$DB_PATH" \
  --query "$QUERY" \
  --limit "$LIMIT"
