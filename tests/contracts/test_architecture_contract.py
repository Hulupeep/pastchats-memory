from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "pastchats_memory"
SCHEMA = SRC / "schema.sql"
CLI = SRC / "cli.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_arch_001_sqlite_access_lives_in_store_only() -> None:
    violations: list[str] = []
    for file in SRC.glob("*.py"):
        if file.name in {"store.py", "__init__.py"}:
            continue
        content = _read(file)
        if re.search(r"\bimport\s+sqlite3\b", content):
            violations.append(str(file.relative_to(ROOT)))

    assert not violations, (
        "CONTRACT VIOLATION: ARCH-001\n"
        "Rule: SQLite access must be isolated to store.py\n"
        f"Found direct sqlite3 imports in: {violations}\n"
        "See: docs/contracts/feature_architecture.yml"
    )


def test_arch_002_schema_must_define_fts5_index() -> None:
    content = _read(SCHEMA)
    assert "CREATE VIRTUAL TABLE IF NOT EXISTS prompts_fts USING fts5" in content, (
        "CONTRACT VIOLATION: ARCH-002\n"
        "Rule: Memory index must expose FTS5 search surface\n"
        "Expected prompts_fts virtual table definition\n"
        "See: docs/contracts/feature_architecture.yml"
    )


def test_arch_003_cli_must_expose_recall_entrypoint() -> None:
    content = _read(CLI)
    assert 'sub.add_parser("recall"' in content, (
        "CONTRACT VIOLATION: ARCH-003\n"
        "Rule: Pre-task memory recall command must exist\n"
        "Expected recall subcommand in CLI\n"
        "See: docs/contracts/feature_architecture.yml"
    )
