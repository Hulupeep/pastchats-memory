## Parent Epic
#1 — Persistent Prompt Memory for Agent Recall

## Build Slice 1 of 2

**Priority:** P0
**Persona:** AI coding agent
**Promise:** "Index prior prompts and recall relevant lessons before starting a new task."

---

## Scope

### In Scope
- Implement schema and ingestion pipeline
- Implement hybrid search and recall CLI output
- Implement sqlite-vec optional path with fallback
- Add contract tests for defined invariants

### Not In Scope
- Hosted API ingestion
- Frontend memory dashboard
- Multi-user permissions

---

## Data Contract

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

### Frontend Interface (if needed)
Not applicable for this slice.

---

## Invariants Referenced
- **I-ARCH-001:** SQLite access MUST be isolated to store layer.
- **I-ARCH-002:** Prompt corpus MUST expose an FTS5 lexical search index.
- **I-ARCH-003:** CLI MUST expose `recall` pre-task entrypoint.
- **I-MEM-001:** Every indexed prompt MUST have exactly one embedding row.
- **I-MEM-002:** Final rank MUST combine lexical and semantic signals.
- **I-MEM-003:** Semantic search MUST function when sqlite-vec is unavailable.
- **I-MEM-004:** Recall output MUST include source traceability for each memory.

---

## Acceptance Criteria
- [ ] `pastchats-memory init` creates schema with FTS5 structures
- [ ] `pastchats-memory index` ingests JSON/JSONL/Markdown conversation history
- [ ] Re-running index does not duplicate rows
- [ ] `pastchats-memory search` returns hybrid score ordered hits
- [ ] `pastchats-memory recall` includes `What worked` and `Source:` fields
- [ ] Contract tests pass for architecture and memory contracts

---

## Gherkin Scenarios
```gherkin
Feature: Index and recall from prompt memory

  Background:
    Given a clean memory database

  @ARCH-002 @MEM-001
  Scenario: Index prompt history into normalized memory tables
    Given a conversation file with user and assistant turns
    When I run the index command
    Then prompts are stored in the prompts table
    And each stored prompt has an embedding row
    And FTS5 can query the prompt content

  @MEM-002
  Scenario: Retrieve memory with hybrid ranking
    Given indexed prompts contain lexical and semantic diversity
    When I run the search command for "hybrid retrieval strategy"
    Then results are ordered by hybrid score
    And at least one result includes lexical score
    And at least one result includes semantic score

  @MEM-003
  Scenario: Fallback when sqlite-vec cannot load
    Given sqlite-vec extension path is invalid
    When semantic search executes
    Then the process does not fail
    And fallback cosine ranking returns hits

  @ARCH-003 @MEM-004
  Scenario: Pre-task recall returns source-traceable context
    Given indexed prompt history exists
    When I run recall for "retry safe webhook handling"
    Then the output starts with "# Memory Recall"
    And each memory entry includes "Source:"
```

---

## Journey: Pre-Task Memory Recall
1. User asks the agent to implement a new feature.
2. Agent derives a recall query from task intent.
3. Agent runs `pastchats-memory recall --query <intent>`.
4. Agent reads `What worked` snippets and source references.
5. Agent begins implementation with prior lessons in context.

---

## Definition of Done
- [ ] Migration/schema is applied
- [ ] Index/search/recall commands are operational
- [ ] sqlite-vec optional path is implemented
- [ ] Fallback path validated
- [ ] Contract tests pass
