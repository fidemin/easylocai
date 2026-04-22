# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- **Never run `git commit` or `git push` without explicit user request.** Only commit or push when the user explicitly asks.
- **Update `docs/ARCHITECTURE.md`** whenever the main structure or flow changes (e.g. new agent, workflow variant, LLM call, tool management change, or core loop modification).
- **Python type annotations are required only when the type is not obvious.** Skip annotations when the type is evident from context (e.g. `x = 0`, `name = "foo"`, simple list/dict literals). Always annotate function signatures, dataclass fields, and any variable where the type is ambiguous.
- **Never use `cd <dir> && git <cmd>` compound commands.** Use `git -C <path> <cmd>` instead to avoid bare repository attack detection and unnecessary permission prompts.
- **Never use multiline `python -c` commands.** Use a single-line `-c` argument only, or write a proper script file. Never use `/tmp` for temporary files.
- **Never write files to `/tmp`.**

## References

| Topic | Document |
|:------|:---------|
| Project overview, features | [`README.md`](README.md) |
| System, OS, software requirements | [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) |
| Installation, initialization, running | [`docs/PRODUCTION.md`](docs/PRODUCTION.md) |
| MCP server configuration | [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) |
| Development mode, testing, adding workflow variants, key code patterns | [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) |
| Architecture, agent responsibilities, data flow diagrams | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |

## Superpowers

### Spec 저장 위치

Superpowers brainstorming을 통해 생성된 design spec은 `.claude/superpowers/specs/` 에 저장한다.

```
.claude/superpowers/specs/YYYY-MM-DD-<topic>-design.md
```

Implementation plan은 `.claude/superpowers/plans/` 에 저장한다.

```
.claude/superpowers/plans/YYYY-MM-DD-<topic>-plan.md
```
