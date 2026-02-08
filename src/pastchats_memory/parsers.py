from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import re
from typing import Any

from .models import PromptTurn


_ALLOWED_EXTENSIONS = {".json", ".jsonl", ".md", ".markdown", ".txt"}
_HINT_TOKENS = ("claude", "chat", "conversation", "prompt", "history")
_ROLE_RE = re.compile(r"^(user|assistant|system|tool)\s*:\s*$", flags=re.IGNORECASE)
_INLINE_ROLE_RE = re.compile(r"^(user|assistant|system|tool)\s*:\s*(.+)$", flags=re.IGNORECASE)


def discover_history_files(inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        candidate = Path(raw).expanduser().resolve()
        if candidate.is_file() and candidate.suffix.lower() in _ALLOWED_EXTENSIONS:
            files.append(candidate)
            continue
        if not candidate.is_dir():
            continue

        for path in candidate.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _ALLOWED_EXTENSIONS:
                continue
            lowered = path.name.lower()
            if any(tok in lowered for tok in _HINT_TOKENS):
                files.append(path)
    return sorted(set(files))


def infer_project(path: Path, roots: list[Path]) -> str:
    for root in roots:
        root = root.expanduser().resolve()
        if root == path:
            return root.stem
        if root in path.parents:
            relative = path.relative_to(root)
            if relative.parts:
                return relative.parts[0]
            return root.stem
    return path.parent.name


def parse_history_file(path: Path, source_project: str) -> list[PromptTurn]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        turns = _parse_jsonish(path, source_project)
    else:
        turns = _parse_markdownish(path, source_project)

    # Ensure stable turn indexes per conversation.
    per_conv_counter: dict[str, int] = {}
    for turn in turns:
        key = turn.conversation_id
        idx = per_conv_counter.get(key, 0)
        turn.turn_index = idx
        per_conv_counter[key] = idx + 1
    return turns


def _parse_jsonish(path: Path, source_project: str) -> list[PromptTurn]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.strip()
    if not text:
        return []

    conversation_default = path.stem
    nodes: list[Any] = []

    if path.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                nodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        nodes.append(parsed)

    turns: list[PromptTurn] = []
    for node in nodes:
        turns.extend(
            _extract_turns_from_node(
                node=node,
                source_path=str(path),
                source_project=source_project,
                conversation_id=conversation_default,
            )
        )
    return turns


def _extract_turns_from_node(
    node: Any,
    source_path: str,
    source_project: str,
    conversation_id: str,
) -> list[PromptTurn]:
    turns: list[PromptTurn] = []

    if isinstance(node, list):
        for item in node:
            turns.extend(
                _extract_turns_from_node(item, source_path, source_project, conversation_id)
            )
        return turns

    if not isinstance(node, dict):
        return turns

    current_conversation = (
        str(
            node.get("conversation_id")
            or node.get("id")
            or node.get("uuid")
            or conversation_id
        )
        if node
        else conversation_id
    )

    role = _coerce_role(node.get("role") or node.get("author") or node.get("speaker"))
    content = _extract_text(node)
    timestamp = _coerce_timestamp(node.get("timestamp") or node.get("created_at") or node.get("time"))

    if content and role:
        turns.append(
            PromptTurn(
                source_path=source_path,
                source_project=source_project,
                conversation_id=current_conversation,
                turn_index=0,
                role=role,
                content=content,
                timestamp=timestamp,
                metadata={"format": "json"},
            )
        )

    for key in ("messages", "turns", "entries", "events", "chat_messages", "conversation"):
        value = node.get(key)
        if isinstance(value, (list, dict)):
            turns.extend(
                _extract_turns_from_node(
                    value,
                    source_path=source_path,
                    source_project=source_project,
                    conversation_id=current_conversation,
                )
            )

    return turns


def _extract_text(node: dict[str, Any]) -> str:
    direct = node.get("content") or node.get("text") or node.get("message")
    if isinstance(direct, str):
        return direct.strip()

    if isinstance(direct, list):
        pieces: list[str] = []
        for item in direct:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("value")
                if isinstance(txt, str):
                    pieces.append(txt)
        return "\n".join(p for p in pieces if p).strip()

    if isinstance(direct, dict):
        txt = direct.get("text") or direct.get("value")
        if isinstance(txt, str):
            return txt.strip()

    return ""


def _parse_markdownish(path: Path, source_project: str) -> list[PromptTurn]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    turns: list[PromptTurn] = []
    current_role = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_role
        if current_role and buffer:
            content = "\n".join(buffer).strip()
            if content:
                turns.append(
                    PromptTurn(
                        source_path=str(path),
                        source_project=source_project,
                        conversation_id=path.stem,
                        turn_index=0,
                        role=current_role,
                        content=content,
                        timestamp=None,
                        metadata={"format": "markdown"},
                    )
                )
        buffer = []

    for line in lines:
        line = line.rstrip()
        match_header = _ROLE_RE.match(line)
        match_inline = _INLINE_ROLE_RE.match(line)

        if match_inline:
            flush()
            current_role = match_inline.group(1).lower()
            buffer = [match_inline.group(2)]
            continue

        if match_header:
            flush()
            current_role = match_header.group(1).lower()
            buffer = []
            continue

        if line.startswith("## "):
            maybe_role = line[3:].strip().rstrip(":").lower()
            if maybe_role in {"user", "assistant", "system", "tool"}:
                flush()
                current_role = maybe_role
                buffer = []
                continue

        if current_role:
            buffer.append(line)

    flush()
    return turns


def _coerce_role(value: Any) -> str:
    if not value:
        return ""
    role = str(value).strip().lower()
    if role in {"human", "user_prompt"}:
        return "user"
    if role in {"ai", "assistant_response", "model"}:
        return "assistant"
    if role in {"system", "user", "assistant", "tool"}:
        return role
    return ""


def _coerce_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def project_roots(inputs: Iterable[str]) -> list[Path]:
    roots: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.exists():
            roots.append(path)
    return roots
