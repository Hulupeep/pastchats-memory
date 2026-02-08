# Prompt Memory Recall Workflow

## Query Construction

Use one-line intent queries with:
- core task noun: `retry policy`, `migration strategy`, `auth middleware`
- constraint: `idempotent`, `safe`, `backward compatible`
- stack hint: `python`, `typescript`, `sqlite`

Example:
`idempotent webhook retry strategy python sqlite`

## Result Triage

Select memories that satisfy at least one:
- identical failure mode
- identical integration point
- same persistence or API constraints

Drop memories that are old, generic, or cross-domain noise.

## Context Injection Pattern

Use this compact format:

```text
Memory lessons:
1) <pattern> (source: <path>)
2) <anti-pattern> (source: <path>)
3) <implementation hint> (source: <path>)
```

Keep the injected block under 12 lines.
