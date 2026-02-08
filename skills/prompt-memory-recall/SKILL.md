---
name: prompt-memory-recall
description: Retrieve and apply lessons from previously solved prompts before starting a new implementation task. Use when coding tasks may benefit from prior attempts, known fixes, anti-patterns, or project-specific context stored in the PastChats memory index.
---

# Prompt Memory Recall

## Workflow

1. Derive a short intent query from the incoming task.
2. Run:
   ```bash
   pastchats-memory recall --db .swarm/prompt_memory.db --query "<intent>" --limit 5
   ```
3. Extract only actionable lessons:
   - proven patterns (`What worked`)
   - constraints and pitfalls
   - source locations to inspect
4. Apply lessons to the implementation plan.
5. If results are weak or empty, proceed normally without forcing memory context.

## Indexing Commands

Initialize DB once:

```bash
pastchats-memory init --db .swarm/prompt_memory.db
```

Index chat history (repeatable):

```bash
pastchats-memory index --db .swarm/prompt_memory.db --input <project-or-history-path>
```

## Response Rules

- Prefer top 3 memories unless confidence is low.
- Include source references when using recalled information.
- Treat memory as advisory context, not authoritative truth.
- Never fabricate missing details from memory snippets.

## References

- Retrieval and ranking details: `references/workflow.md`
- Helper script: `scripts/pre_task_recall.sh`
