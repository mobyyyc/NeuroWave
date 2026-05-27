# MiniSynth Daily Workflow

This file tells future Codex sessions how to work on MiniSynth.

## New Session Startup Prompt

If starting from a fresh Codex session, use this prompt:

```text
Go to /Users/moby/Desktop/Coding/MiniSynth and start the MiniSynth daily workflow.
Read WORKFLOW.md, PLAN.md, and PROGRESS.md first.
Do not modify anything until I approve today's task.
```

## Required Startup Steps

When the user asks to start the MiniSynth daily workflow:

1. Change context to `/Users/moby/Desktop/Coding/MiniSynth`.
2. Read `WORKFLOW.md`.
3. Read `PLAN.md` for the long-term architecture.
4. Read `PROGRESS.md` for current status and the next task list.
5. Run `git status --short` and `git log --oneline --decorate -6`.
6. Report the current project status to the user.
7. Identify the next small unchecked task from the current active phase in `PROGRESS.md`.
8. Explain exactly how the task will be completed.
9. Wait for explicit user approval before editing files, running long installs, or committing.

## Permission Rule

Do not modify project files until the user explicitly approves the proposed task.

Allowed before approval:

- Reading files.
- Checking git status.
- Inspecting logs.
- Explaining the next task.
- Proposing an implementation plan.

Not allowed before approval:

- Editing files.
- Creating commits.
- Installing dependencies.
- Running destructive cleanup.
- Changing `PROGRESS.md`.

## Daily Work Cycle

After the user approves the proposed task:

1. Mark the selected task as `[~]` in `PROGRESS.md` only if the task will take multiple steps.
2. Make the smallest coherent code or documentation change.
3. Run focused verification.
4. Update `PROGRESS.md`:
   - Change the completed task from `[ ]` or `[~]` to `[x]`.
   - Add a dated entry under "Progress Log".
   - Include the commit hash after committing if possible.
5. Commit the completed work.
6. If the progress log needs the final commit hash, amend or add a follow-up tracker commit only when useful.
7. Report:
   - What changed.
   - What verification ran.
   - Commit message and hash.
   - What the next likely task is.

## Commit Rules

- Commit after each completed task or coherent group of tiny tasks.
- Keep commit messages short and specific.
- Do not commit generated outputs unless the user explicitly asks.
- The repo is configured locally with `commit.gpgsign=false`, so normal commits should not require GPG signing.

Suggested commit message style:

```text
Add render smoke test
Move ADSR envelope into package
Add SynthConfig normalization helpers
Update MiniSynth progress tracker
```

## Progress Tracker Rules

`PROGRESS.md` is the source of truth for daily task state.

Status meanings:

- `[ ]` Not started.
- `[x]` Done and committed.
- `[~]` In progress.
- `[!]` Blocked.

When completing a task:

- Update only the relevant task unless the work truly completes more than one item.
- Add a concise progress log entry.
- Keep the next task clear for the next session.

## Planning Rules

`PLAN.md` is the source of truth for long-term direction.

Update `PLAN.md` when:

- The architecture changes.
- The ML strategy changes.
- A milestone is added, removed, or reordered.
- The meaning of a major component changes.

Update `PROGRESS.md` after updating `PLAN.md` so daily tasks match the plan.

## Current Project Goal

MiniSynth is moving toward an ML-controllable synthesizer that can listen to a short audio clip and approximate it by predicting editable synthesizer parameters.

The first priority is not ML. The first priority is a deterministic, parameterized synth engine with a stable config schema.

Current active phase is tracked in `PROGRESS.md`.
