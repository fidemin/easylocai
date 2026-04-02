# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (uses pipx with Python 3.12)
bash install.sh

# Run the agent (after install)
easylocai

# Run during development (without install)
python -m easylocai.run

# Initialize user config (~/.config/easylocai/config.json)
easylocai init
easylocai init --force

# Run beta workflow variant (after install)
easylocai --flag=beta

# Run beta workflow variant during development
python -m easylocai.run --flag=beta

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

Beta variants (`*_beta.py`) are experimental and toggled via `--flag=beta`.

### LLM Abstraction (`easylocai/core/llm_call.py`, `easylocai/llm_calls/`)

`LLMCallV2` is the base class: loads a Jinja2 prompt template from `resources/prompts/`, calls Ollama, and parses the response into a Pydantic output model. Each logical LLM call (Planner, Replanner, TaskRouter, ToolSelector, etc.) is a concrete subclass.

### Tool Management (`easylocai/core/tool_manager.py`)

`ToolManager` manages MCP server lifecycles (via `ServerManager`) and wraps discovered tools as `Tool` objects. Tools are configured in `~/.config/easylocai/config.json` under `mcpServers`. ToolSelector finds relevant tools by semantic search over tool descriptions.

### Search (`easylocai/search_engines/`, `easylocai/core/search_engine.py`)

`AdvancedSearchEngine` combines `KeywordSearchEngine` (BM25) and `SemanticSearchEngine` (ChromaDB embeddings) using Reciprocal Rank Fusion (RRF). Used by `SingleTaskAgent` to find relevant tool candidates before execution.

### Key Patterns

- **Agent base class** (`easylocai/core/agent.py`): `Agent[InModel, OutModel]` with `async _run()` — all agents are fully async
- **Prompt templates** in `resources/prompts/*.jinja2` — loaded at runtime via `installed_resources_dir()` in `easylocai/utlis/resource_util.py`
- **Pydantic v2** for all input/output models in `easylocai/schemas/`
- Tests use `pytest-asyncio` with `asyncio_mode = "auto"`; integration tests cover agent behavior end-to-end