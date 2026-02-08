## Epic: Persistent Prompt Memory for Agent Recall

**Priority:** P0
**Primary Persona:** AI coding agent
**Secondary Personas:** Solo developer, platform maintainer
**Core Promise:** "Before starting work, the agent can recall what previously worked and what failed across projects."

> Build a durable memory subsystem that indexes conversation history and injects relevant prior outcomes into each new task context.

---

## Definitions
- **Memory Item:** A single indexed prompt turn with metadata and embedding.
- **Hybrid Search:** Combined lexical (FTS5) and semantic (vector) ranking.
- **Recall Block:** Structured output used to prime an agent before task execution.

---

## Scope (MVP)

### In Scope
1. SQLite schema for prompt turns, FTS5, embeddings, and indexing runs
2. Ingestion pipeline for JSON/JSONL/Markdown conversation history
3. Hybrid search CLI with `search` and `recall` commands
4. sqlite-vec integration with runtime fallback when extension is unavailable
5. Specflow contracts and contract tests for architectural and memory invariants

### Not In Scope
- Automatic ingestion from hosted APIs
- Multi-user auth or tenancy
- UI dashboard
- Cross-DB backends beyond SQLite

---

## Data Contracts

### Table: `prompts`
```sql
CREATE TABLE prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_path TEXT NOT NULL,
  source_project TEXT NOT NULL,
  conversation_id TEXT,
  turn_index INTEGER NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  timestamp TEXT,
  content_hash TEXT NOT NULL UNIQUE,
  metadata_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

### Table: `prompt_embeddings`
```sql
CREATE TABLE prompt_embeddings (
  prompt_id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
);
```

### Virtual Table: `prompts_fts`
```sql
CREATE VIRTUAL TABLE prompts_fts USING fts5(
  content,
  source_project,
  conversation_id UNINDEXED,
  content='prompts',
  content_rowid='id'
);
```

---

## Invariants
- **I-ARCH-001:** SQLite access MUST be isolated to store layer.
- **I-ARCH-002:** Prompt corpus MUST expose an FTS5 lexical search index.
- **I-ARCH-003:** CLI MUST expose `recall` pre-task entrypoint.
- **I-MEM-001:** Every indexed prompt MUST have exactly one embedding row.
- **I-MEM-002:** Final rank MUST combine lexical and semantic signals.
- **I-MEM-003:** Semantic search MUST function when sqlite-vec is unavailable.
- **I-MEM-004:** Recall output MUST include source traceability for each memory.

---

## Gherkin Scenarios
```gherkin
Feature: Persistent prompt memory recall

  Background:
    Given a memory database exists
    And past prompts are indexed from at least one project

  @ARCH-002 @MEM-002
  Scenario: Agent searches memory with hybrid ranking
    When the agent runs "pastchats-memory search --query 'retry strategy'"
    Then the system returns ranked results
    And each result has a hybrid score

  @ARCH-003 @MEM-004
  Scenario: Agent runs pre-task recall
    When the agent runs "pastchats-memory recall --query 'build webhook retries'"
    Then the output contains "# Memory Recall"
    And each memory entry contains "Source:"

  @MEM-003
  Scenario: sqlite-vec extension is not installed
    Given sqlite-vec fails to load
    When the agent runs semantic retrieval
    Then the system returns results using fallback cosine search

  @MEM-001
  Scenario Outline: Indexed prompts always receive embeddings
    Given a parsed prompt turn with role "<role>"
    When indexing is executed
    Then prompt rows and embedding rows remain equal

    Examples:
      | role      |
      | user      |
      | assistant |
      | system    |
```

---

## Journeys
1. Agent receives new task request.
2. Agent runs `recall` using a condensed query derived from task intent.
3. Agent reviews top memory entries with prior successful responses.
4. Agent starts implementation using recalled constraints and patterns.
5. Agent completes work and re-indexes new prompt history.

---

## Definition of Done
- [ ] Schema creates prompt, embedding, and FTS5 structures
- [ ] Index command ingests supported formats and writes embeddings
- [ ] Search command returns hybrid ranked results
- [ ] Recall command outputs source-traceable memories
- [ ] Contract tests enforce ARCH and MEM invariants
- [ ] Contract index includes all feature and journey contracts

---

## Build Slices (2 subtasks)
1. Indexer + schema + parser implementation
2. Hybrid retrieval + recall command + contract tests
