# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- **Never run `git commit` or `git push` without explicit user request.** Only commit or push when the user explicitly asks.
- **Update `docs/ARCHITECTURE.md`** whenever the main structure or flow changes (e.g. new agent, workflow variant, LLM call, tool management change, or core loop modification).
- **Python type annotations are required only when the type is not obvious.** Skip annotations when the type is evident from context (e.g. `x = 0`, `name = "foo"`, simple list/dict literals). Always annotate function signatures, dataclass fields, and any variable where the type is ambiguous.
- **Never use `cd <dir> && git <cmd>` compound commands.** Use `git -C <path> <cmd>` instead to avoid bare repository attack detection and unnecessary permission prompts.
- **Never use multiline `python -c` commands.** Use a single-line `-c` argument only, or write a proper script file. Never use `/tmp` for temporary files.
- **Never write files to `/tmp`.**

## Commands

```bash
# Install (uses pipx with Python 3.12)
bash install.sh

# Run the agent (after install)
easylocai

# Run during development (without install)
python -m easylocai.run

# Run with a workflow flag (beta, alpha, or any registered variant)
easylocai --flag=<flag>
python -m easylocai.run --flag=<flag>

# Initialize user config (~/.config/easylocai/config.json)
easylocai init
easylocai init --force

# Run tests (all async, auto mode)
pytest

# Run a single test file
pytest tests/integration_tests/agents/test_single_task_agent.py

# Format code
black .
```

**Prerequisites:** Ollama must be running with `gpt-oss:20b` pulled (`ollama pull gpt-oss:20b`).

## Architecture

The system implements a **Plan-Execute-Replan** loop for autonomous local task execution using the `gpt-oss:20b` model via Ollama and MCP tools.

### Core Loop (`workflow.py`)

```
User Input
  → PlanAgent     — decomposes query into a task list
  → [loop]
      SingleTaskAgent  — executes one task using tools or reasoning
      ReplanAgent      — checks if goal is met; returns answer OR new tasks
```

### Agent Layer (`easylocai/agents/`)

- **`PlanAgent`** — runs `QueryReformatter` (pronoun resolution) → `Planner` → task list
- **`SingleTaskAgent`** — searches tool candidates, then loops: `TaskRouter` decides tool-call vs reasoning per subtask until task complete, then `TaskResultFilter` cleans output
- **`ReplanAgent`** — runs `Replanner` → either final response or new tasks


### LLM Abstraction (`easylocai/core/llm_call.py`, `easylocai/llm_calls/`)

`LLMCallV2` is the base class: loads a Jinja2 prompt template from `resources/prompts/`, calls Ollama, and parses the response into a Pydantic output model. Each logical LLM call (Planner, Replanner, TaskRouter, ToolSelector, etc.) is a concrete subclass.

### Tool Management (`easylocai/core/tool_manager.py`)

`ToolManager` manages MCP server lifecycles (via `ServerManager`) and wraps discovered tools as `Tool` objects. Tools are configured in `~/.config/easylocai/config.json` under `mcpServers`. ToolSelector finds relevant tools by semantic search over tool descriptions.

### Search (`easylocai/search_engines/`, `easylocai/core/search_engine.py`)

`AdvancedSearchEngine` combines `KeywordSearchEngine` (BM25) and `SemanticSearchEngine` (ChromaDB embeddings) using Reciprocal Rank Fusion (RRF). Used by `SingleTaskAgent` to find relevant tool candidates before execution.

### Workflow Flags

`--flag=<name>` selects a workflow variant at runtime. Flags are registered in `workflow_registry` in `easylocai/main.py`:

```python
workflow_registry = {
    "main": run_agent_workflow_main,
    "beta": run_agent_workflow_main_beta,   # example
    "alpha": run_agent_workflow_main_alpha, # example
}
```

To add a new variant: create a workflow class (e.g. `easylocai/workflow_<name>.py`), a runner function in `easylocai/main_<name>.py`, and register it in `workflow_registry`.

### Key Patterns

- **Agent base class** (`easylocai/core/agent.py`): `Agent[InModel, OutModel]` with `async _run()` — all agents are fully async
- **Prompt templates** in `resources/prompts/*.jinja2` — loaded at runtime via `installed_resources_dir()` in `easylocai/utlis/resource_util.py`
- **Pydantic v2** for all input/output models in `easylocai/schemas/`
- Tests use `pytest-asyncio` with `asyncio_mode = "auto"`; integration tests cover agent behavior end-to-end

---

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
