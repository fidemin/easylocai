# Development

## Adding a Workflow Variant

Variants are registered in `workflow_registry` in `easylocai/main.py`. To add a new variant:
1. Create a workflow class (e.g. `easylocai/workflow_<name>.py`)
2. Create a runner function in `easylocai/main_<name>.py`)
3. Register it in `workflow_registry`

## Initialize config
### first time initialization
```bash
python -m easylocai.run init
```

Config file is created at `~/.config/easylocai/config.json` with an empty `mcpServers` dict. If the file already exists, this is a no-op.

### force re-initialization
```bash
python -m easylocai.run init --force
```
> **Warning:** `--force` overwrites the existing config with `{"mcpServers": {}}`, permanently deleting all MCP server entries you have configured. Back up `~/.config/easylocai/config.json` before using this flag.

## Configuration

See [`docs/CONFIGURATION.md`](CONFIGURATION.md) for MCP server configuration details.

## Run

```bash
python -m easylocai.run

# With a workflow flag
python -m easylocai.run --flag=<flag>
```

## Testing

- `pytest-asyncio` with `asyncio_mode = "auto"` — all tests are async
- No mocking of LLM calls; tests hit a live Ollama instance

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/integration_tests/agents/test_single_task_agent.py
```

### Unit Tests
Located in `tests/unittests/`

### Integration Tests
**Prerequisite:** Ollama must be running with `gpt-oss:20b` pulled.

Located in `tests/integration_tests/`, organized by component:

| Test File | Coverage |
|:----------|:---------|
| `agents/test_plan_agent.py` | PlanAgent — query reformatting, task planning, conversation history |
| `agents/test_replan_agent.py` | ReplanAgent — replanning, completion detection, conversation history |
| `agents/test_single_task_agent.py` | SingleTaskAgent — tool execution, multi-turn context |
| `agents/test_reasoning_agent.py` | ReasoningAgent — reasoning, context usage, no re-fetching from conversation history |
| `search_engines/test_keyword_search_engine.py` | KeywordSearchEngine — BM25 search, incremental indexing |

Fixtures are defined in `tests/conftest.py`: `ollama_client`, `tool_manager` (spins up a live MCP filesystem server), `search_engine`.

## Formatting

```bash
black .
```

## Key Development Patterns

### Context Schemas (`easylocai/schemas/context.py`)

Three flat Pydantic models carry state through the workflow:

| Model | Owner | Scope |
|:------|:------|:------|
| `GlobalContext` | `main.py` | Persists across conversations; holds `conversation_histories` |
| `WorkflowContext` | `EasylocaiWorkflow` | Single user query lifecycle |
| `SingleTaskAgentContext` | `SingleTaskAgent` | Single task execution; includes all WorkflowContext fields + agent-internal fields |

`ConversationHistory` is appended to `GlobalContext` at the end of each turn and threaded into every agent call (PlanAgent, SingleTaskAgent, ReplanAgent) for multi-turn context resolution.

### LLM Call Pattern (`easylocai/core/llm_call.py`)

`LLMCallV2[InModel, OutModel]` is the base class for all LLM calls:
- Loads Jinja2 prompt templates from `resources/prompts/`
- Calls Ollama with structured output (JSON schema from Pydantic model)
- Parses the response into `OutModel` via Pydantic
- Retries up to 3 times on empty or unparseable responses

Each LLM call (Planner, Replanner, TaskRouter, ToolSelector, Reasoning, etc.) is a concrete subclass in `easylocai/llm_calls/`. Field names in `InModel` map directly to Jinja2 template variables via `model_dump()`.

### Agent Pattern (`easylocai/core/agent.py`)

`Agent[InModel, OutModel]` — all agents are fully async; implement `async _run()`.

### Prompt Templates

- Location: `resources/prompts/*.jinja2`
- Loaded at runtime via `installed_resources_dir()` in `easylocai/utlis/resource_util.py`
- `StrictUndefined` is set — every template variable must exist in the input model or rendering will raise
