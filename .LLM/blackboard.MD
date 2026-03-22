# Agent Blackboard

This file is a shared coordination space for multiple coding agents working in the repository at the same time.

## Purpose

Use this document to communicate work that matters to other agents, including:

- current ownership,
- files being edited,
- important assumptions,
- blockers,
- and handoff notes.

The goal is to reduce duplicate work, merge conflicts, and hidden assumptions.

## Working Rules

- Keep updates short and concrete.
- Prefer append-only updates instead of rewriting another agent's notes.
- Claim file ownership before making substantial edits.
- Note blockers as soon as they are discovered.
- Record decisions that affect other parts of the repo.
- If a task is complete, mark it clearly.

## Recommended Update Format

Each update should use this structure:

```md
## [Agent or Task Name]

- Status: planned | in progress | blocked | complete
- Owner: agent name or identifier
- Scope: short description of the task
- Files: `path/to/file.py`, `path/to/other.py`
- Notes: key decisions, assumptions, or findings
- Blockers: none | short blocker description
- Next: immediate next step or handoff instruction
```

## Usage Guidance

Use the blackboard when:

- starting a non-trivial task,
- touching files another agent may also edit,
- discovering a constraint or bug others need to know,
- handing work off,
- or finishing work that changes the plan.

Do not use the blackboard for:

- long design essays,
- raw command output,
- or low-value status noise.

## Example Entry

```md
## Runtime Validation Docs

- Status: in progress
- Owner: agent-docs
- Scope: define documentation requirements for validation helpers
- Files: `.LLM/documentation.MD`, `bsnet/src/validation/__init__.py`
- Notes: standard requires Google-style docstrings plus Preconditions and Postconditions
- Blockers: none
- Next: update function templates and verify wording is consistent with Python 3.11 typing
```

## Conflict Prevention

Before editing a shared file:

- check whether another agent has already claimed it,
- avoid overlapping write scopes when possible,
- and leave a handoff note if the file still needs follow-up work.

If overlap is unavoidable, document:

- which part of the file you changed,
- what assumption you made,
- and what another agent should review before merging.

## Default Standard

Unless there is a better task-specific format, use the recommended update structure above and optimize for clarity over completeness.

NEVER modify anything above the start of the blackboard.

### --- START OF BLACKBOARD --- ###