# Agent Integration Tests Design

**Date:** 2026-04-13
**Scope:** Integration tests for `PlanAgent` and `ReasoningAgent` in `tests/integration_tests/agents/`

---

## Goal

Create integration tests for the two agents that currently lack them (`PlanAgent`, `ReasoningAgent`), and introduce a shared `conftest.py` to eliminate fixture duplication.

**Out of scope:** Modifying `test_single_task_agent.py` or `test_replan_agent.py`.

---

## Decisions

- **Structural tests only** — assert output shape and field types, not answer correctness. Behavioral verification requires deterministic LLM output, which is not feasible here.
- **Real external dependencies** — Ollama must be running. `PlanAgent` and `ReasoningAgent` need only Ollama; no MCP server required.
- **Coverage level** — happy path + key input variants + one complex case per agent.

---

## File Structure

```
tests/integration_tests/agents/
  conftest.py                  ← new: shared fixtures
  test_plan_agent.py           ← replace existing file
  test_reasoning_agent.py      ← replace existing file
  test_single_task_agent.py    ← unchanged
  test_replan_agent.py         ← unchanged
```

---

## conftest.py

Two shared fixtures available to all tests in this directory:

- `ollama_client` — `AsyncClient(host="http://localhost:11434")`
- `mcp_servers_config` — filesystem MCP server config (used by `SingleTaskAgent` tests)

---

## test_plan_agent.py

**Agent:** `PlanAgent(client=ollama_client)`
**Input:** `PlanAgentInput(user_query: str, user_conversations: list[UserConversation] = [])`
**Output:** `PlanAgentOutput(context: str | None, tasks: list[str])`

| Test | Input variant | Assertions |
|------|--------------|------------|
| `test_basic_query` | query only | `tasks` non-empty `list[str]`, each item non-empty string; `context` is `str\|None` |
| `test_query_with_conversation_history` | query + 1 `UserConversation` | same shape |
| `test_multi_turn_conversation` | query + 3 `UserConversation` entries | same shape |

---

## test_reasoning_agent.py

**Agent:** `ReasoningAgent(client=ollama_client)`
**Input:** `ReasoningAgentInput(task: dict, user_context: str|None, previous_task_results: list[dict], previous_subtask_results: list[dict] = [])`
**Output:** `ReasoningAgentOutput(reasoning: str, final: str, confidence: int)`

| Test | Input variant | Assertions |
|------|--------------|------------|
| `test_without_context_or_prior_results` | no context, empty results | `reasoning`, `final` non-empty strings; `confidence` is `int` |
| `test_with_user_context` | `user_context` populated | same shape |
| `test_with_previous_task_results` | `previous_task_results` non-empty | same shape |
| `test_with_previous_subtask_results` | `previous_subtask_results` non-empty | same shape |
| `test_all_context_combined` | all optional inputs populated | same shape |
