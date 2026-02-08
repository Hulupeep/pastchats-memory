PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS prompts (
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

CREATE INDEX IF NOT EXISTS idx_prompts_project ON prompts(source_project);
CREATE INDEX IF NOT EXISTS idx_prompts_conversation ON prompts(conversation_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_prompts_role ON prompts(role);
CREATE INDEX IF NOT EXISTS idx_prompts_timestamp ON prompts(timestamp);

CREATE VIRTUAL TABLE IF NOT EXISTS prompts_fts USING fts5(
  content,
  source_project,
  conversation_id UNINDEXED,
  content='prompts',
  content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS prompts_ai AFTER INSERT ON prompts BEGIN
  INSERT INTO prompts_fts(rowid, content, source_project, conversation_id)
  VALUES (new.id, new.content, new.source_project, COALESCE(new.conversation_id, ''));
END;

CREATE TRIGGER IF NOT EXISTS prompts_ad AFTER DELETE ON prompts BEGIN
  INSERT INTO prompts_fts(prompts_fts, rowid, content, source_project, conversation_id)
  VALUES('delete', old.id, old.content, old.source_project, COALESCE(old.conversation_id, ''));
END;

CREATE TRIGGER IF NOT EXISTS prompts_au AFTER UPDATE ON prompts BEGIN
  INSERT INTO prompts_fts(prompts_fts, rowid, content, source_project, conversation_id)
  VALUES('delete', old.id, old.content, old.source_project, COALESCE(old.conversation_id, ''));
  INSERT INTO prompts_fts(rowid, content, source_project, conversation_id)
  VALUES (new.id, new.content, new.source_project, COALESCE(new.conversation_id, ''));
END;

CREATE TABLE IF NOT EXISTS prompt_embeddings (
  prompt_id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vector_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_prompt_embeddings_model ON prompt_embeddings(model);

CREATE TABLE IF NOT EXISTS index_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  input_roots_json TEXT NOT NULL,
  files_scanned INTEGER NOT NULL DEFAULT 0,
  prompts_indexed INTEGER NOT NULL DEFAULT 0,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS memory_settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
