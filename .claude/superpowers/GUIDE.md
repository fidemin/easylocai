# Superpowers Guide

## File Locations

| Artifact | Path |
|:---------|:-----|
| Design specs | `.claude/superpowers/specs/YYYY-MM-DD-<topic>-design.md` |
| Implementation plans | `.claude/superpowers/plans/YYYY-MM-DD-<topic>-plan.md` |

## Skills

Invoke skills with the `Skill` tool before acting. Even a 1% chance a skill applies means you must invoke it.

| Skill | When to use |
|:------|:------------|
| `superpowers:brainstorming` | Before planning any non-trivial feature |
| `superpowers:writing-plans` | After brainstorming; produces the implementation plan |
| `superpowers:subagent-driven-development` | Executing a plan task-by-task via subagents (recommended) |
| `superpowers:executing-plans` | Executing a plan inline in the current session |
| `superpowers:using-git-worktrees` | Setting up an isolated workspace before implementation |
| `superpowers:finishing-a-development-branch` | After all tasks complete; handles merge/PR/discard |
| `superpowers:test-driven-development` | Writing code — follow TDD per task |
| `superpowers:requesting-code-review` | Requesting a code review from a subagent |

## Typical Workflow

```
brainstorming → writing-plans → using-git-worktrees → subagent-driven-development → finishing-a-development-branch
```
