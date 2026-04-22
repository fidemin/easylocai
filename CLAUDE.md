# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

### claude code rules
- **Never run `git commit` or `git push` without explicit user request.** Only commit or push when the user explicitly asks.
- **Never use `cd <dir> && git <cmd>` compound commands.** Use `git -C <path> <cmd>` instead to avoid bare repository attack detection and unnecessary permission prompts.
- **Never use multiline `python -c` commands.** Use a single-line `-c` argument only, or write a proper script file. Never use `/tmp` for temporary files.
- **Never write files to `/tmp`.**

## References

Read the relevant document before starting any task that touches the corresponding area. Update it when your changes affect the content it covers.

| Topic | Document |
|:------|:---------|
| Project overview, features | [`README.md`](README.md) |
| System, OS, software requirements | [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) |
| Installation, initialization, running | [`docs/PRODUCTION.md`](docs/PRODUCTION.md) |
| MCP server configuration | [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) |
| Development mode, testing, adding workflow variants, key code patterns | [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) |
| Architecture, agent responsibilities, data flow diagrams | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |

## Superpowers

See [`.claude/superpowers/GUIDE.md`](.claude/superpowers/GUIDE.md) for skill usage, file locations, and the typical workflow.
