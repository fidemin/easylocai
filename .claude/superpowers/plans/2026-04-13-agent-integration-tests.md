# Agent Integration Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the improperly-created `test_plan_agent.py` and `test_reasoning_agent.py` with clean, structurally-correct integration tests, and introduce a shared `conftest.py` for fixture reuse.

**Architecture:** One `conftest.py` provides shared `ollama_client` and `mcp_servers_config` fixtures. Two new test files (`test_plan_agent.py`, `test_reasoning_agent.py`) replace the existing ones. All tests assert output shape only — no behavioral/correctness assertions.

**Tech Stack:** pytest, pytest-asyncio, ollama AsyncClient, easylocai agents

---

## File Map

| Action | Path |
|--------|------|
| Create | `tests/integration_tests/agents/conftest.py` |
| Replace | `tests/integration_tests/agents/test_plan_agent.py` |
| Replace | `tests/integration_tests/agents/test_reasoning_agent.py` |
| Unchanged | `tests/integration_tests/agents/test_single_task_agent.py` |
| Unchanged | `tests/integration_tests/agents/test_replan_agent.py` |

---

## Task 1: Create conftest.py with shared fixtures

**Files:**
- Create: `tests/integration_tests/agents/conftest.py`

- [ ] **Step 1: Create the file**

```python
import os

import pytest
from ollama import AsyncClient


@pytest.fixture
def ollama_client():
    return AsyncClient(host="http://localhost:11434")


@pytest.fixture
def mcp_servers_config():
    return {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                os.getcwd(),
            ],
        }
    }
```

- [ ] **Step 2: Verify pytest can collect the fixtures**

Run: `pytest tests/integration_tests/agents/ --collect-only -q 2>&1 | head -20`

Expected: no errors about fixture resolution; existing tests still listed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration_tests/agents/conftest.py
git commit -m "test: add shared conftest fixtures for agent integration tests"
```

---

## Task 2: Replace test_plan_agent.py

**Files:**
- Replace: `tests/integration_tests/agents/test_plan_agent.py`

**Agent under test:** `PlanAgent(client: AsyncClient)`
**Input:** `PlanAgentInput(user_query: str, user_conversations: list[UserConversation] = [])`
**Output:** `PlanAgentOutput(context: str | None, tasks: list[str])`

- [ ] **Step 1: Delete the existing file and write the replacement**

```python
import pytest

from easylocai.agents.plan_agent import PlanAgent, PlanAgentInput, PlanAgentOutput
from easylocai.schemas.common import UserConversation


def assert_plan_output(output: PlanAgentOutput) -> None:
    assert isinstance(output.tasks, list)
    assert len(output.tasks) > 0
    for task in output.tasks:
        assert isinstance(task, str)
        assert len(task) > 0
    assert output.context is None or isinstance(output.context, str)


class TestPlanAgent:

    @pytest.mark.asyncio
    async def test_basic_query(self, ollama_client):
        """Happy path: query with no conversation history."""
        agent = PlanAgent(client=ollama_client)
        input_ = PlanAgentInput(
            user_query="Find all Python files in the current directory and count them",
        )
        output: PlanAgentOutput = await agent.run(input_)
        assert_plan_output(output)

    @pytest.mark.asyncio
    async def test_query_with_conversation_history(self, ollama_client):
        """Variant: query accompanied by one prior conversation turn."""
        agent = PlanAgent(client=ollama_client)
        input_ = PlanAgentInput(
            user_query="Summarize it",
            user_conversations=[
                UserConversation(
                    user_query="Find the README.md file",
                    assistant_answer="Found README.md in the root directory. It contains project setup instructions.",
                )
            ],
        )
        output: PlanAgentOutput = await agent.run(input_)
        assert_plan_output(output)

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, ollama_client):
        """Complex case: query with three prior conversation turns."""
        agent = PlanAgent(client=ollama_client)
        input_ = PlanAgentInput(
            user_query="Now delete them",
            user_conversations=[
                UserConversation(
                    user_query="Find all log files",
                    assistant_answer="Found 3 log files: app.log, error.log, debug.log",
                ),
                UserConversation(
                    user_query="Show me the largest one",
                    assistant_answer="error.log is the largest at 2MB",
                ),
                UserConversation(
                    user_query="What does it contain?",
                    assistant_answer="It contains database connection errors from last week",
                ),
            ],
        )
        output: PlanAgentOutput = await agent.run(input_)
        assert_plan_output(output)
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/integration_tests/agents/test_plan_agent.py -v`

Expected: 3 tests PASSED (requires Ollama running with `gpt-oss:20b`).

- [ ] **Step 3: Commit**

```bash
git add tests/integration_tests/agents/test_plan_agent.py
git commit -m "test: add PlanAgent integration tests (structural)"
```

---

## Task 3: Replace test_reasoning_agent.py

**Files:**
- Replace: `tests/integration_tests/agents/test_reasoning_agent.py`

**Agent under test:** `ReasoningAgent(client: AsyncClient)`
**Input:** `ReasoningAgentInput(task: dict, user_context: str | None, previous_task_results: list[dict], previous_subtask_results: list[dict] = [])`
**Output:** `ReasoningAgentOutput(reasoning: str, final: str, confidence: int)`

- [ ] **Step 1: Delete the existing file and write the replacement**

```python
import pytest

from easylocai.agents.reasoning_agent import (
    ReasoningAgent,
    ReasoningAgentInput,
    ReasoningAgentOutput,
)


def assert_reasoning_output(output: ReasoningAgentOutput) -> None:
    assert isinstance(output.reasoning, str)
    assert len(output.reasoning) > 0
    assert isinstance(output.final, str)
    assert len(output.final) > 0
    assert isinstance(output.confidence, int)


class TestReasoningAgent:

    @pytest.mark.asyncio
    async def test_without_context_or_prior_results(self, ollama_client):
        """Happy path: no user_context, no previous results."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            task={"description": "What is 2 + 2?"},
            user_context=None,
            previous_task_results=[],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_with_user_context(self, ollama_client):
        """Variant: user_context provided, no prior results."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            task={"description": "Summarize the user's goal"},
            user_context="The user wants to refactor their Python codebase to use async/await.",
            previous_task_results=[],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_with_previous_task_results(self, ollama_client):
        """Variant: previous_task_results provided, no user_context."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            task={"description": "How many Python files were found?"},
            user_context=None,
            previous_task_results=[
                {
                    "task": "List all files",
                    "result": "Found: main.py, utils.py, config.py, README.md",
                }
            ],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_with_previous_subtask_results(self, ollama_client):
        """Variant: previous_subtask_results provided, no user_context or task results."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            task={"description": "Combine counts into a total"},
            user_context=None,
            previous_task_results=[],
            previous_subtask_results=[
                {"subtask": "Count files in src/", "result": "5 files"},
                {"subtask": "Count files in tests/", "result": "3 files"},
            ],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)

    @pytest.mark.asyncio
    async def test_all_context_combined(self, ollama_client):
        """Complex case: user_context + previous_task_results + previous_subtask_results all populated."""
        agent = ReasoningAgent(client=ollama_client)
        input_ = ReasoningAgentInput(
            task={"description": "Write a summary of all findings"},
            user_context="User is auditing the project structure before a refactor.",
            previous_task_results=[
                {"task": "List top-level directories", "result": "src/, tests/, docs/"},
            ],
            previous_subtask_results=[
                {"subtask": "Count Python files in src/", "result": "12 files"},
                {"subtask": "Count test files", "result": "8 test files"},
            ],
        )
        output: ReasoningAgentOutput = await agent.run(input_)
        assert_reasoning_output(output)
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/integration_tests/agents/test_reasoning_agent.py -v`

Expected: 5 tests PASSED (requires Ollama running with `gpt-oss:20b`).

- [ ] **Step 3: Run all agent tests together**

Run: `pytest tests/integration_tests/agents/ -v`

Expected: all tests PASSED, no fixture errors.

- [ ] **Step 4: Commit**

```bash
git add tests/integration_tests/agents/test_reasoning_agent.py
git commit -m "test: add ReasoningAgent integration tests (structural)"
```
