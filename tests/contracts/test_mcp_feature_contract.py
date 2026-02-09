from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_mcp_001_entrypoint_exists() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "pastchats-memory-mcp" in pyproject, (
        "CONTRACT VIOLATION: MCP-001\n"
        "Rule: Project MUST ship an MCP server entrypoint\n"
        "Expected pastchats-memory-mcp script in pyproject.toml\n"
        "See: docs/contracts/feature_mcp.yml"
    )


def test_mcp_002_and_003_tools_exist() -> None:
    server_file = ROOT / "src" / "pastchats_memory" / "mcp_server.py"
    content = server_file.read_text(encoding="utf-8")
    assert "def recall(" in content, (
        "CONTRACT VIOLATION: MCP-002\n"
        "Rule: MCP server MUST expose recall tool\n"
        "Expected recall implementation in mcp_server.py\n"
        "See: docs/contracts/feature_mcp.yml"
    )
    assert "def store_turn(" in content, (
        "CONTRACT VIOLATION: MCP-003\n"
        "Rule: MCP server MUST expose store_turn tool\n"
        "Expected store_turn implementation in mcp_server.py\n"
        "See: docs/contracts/feature_mcp.yml"
    )


def test_mcp_004_store_supports_event_id() -> None:
    store_file = ROOT / "src" / "pastchats_memory" / "store.py"
    content = store_file.read_text(encoding="utf-8")
    assert "event_id" in content and "def store_turn" in content, (
        "CONTRACT VIOLATION: MCP-004\n"
        "Rule: Store layer MUST support direct turn ingestion with optional event_id idempotency\n"
        "Expected event_id support in store.py store_turn\n"
        "See: docs/contracts/feature_mcp.yml"
    )

