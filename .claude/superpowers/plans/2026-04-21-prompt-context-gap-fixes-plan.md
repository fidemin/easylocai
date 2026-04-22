# Prompt Context Gap Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the context gaps identified in the spec so that each LLM call receives all the context variables it needs for correct multi-turn and subtask-aware behavior.

**Architecture:** Five sequential changes: (1) pass `conversation_histories` to Planner, (2) pass `conversation_histories` to Replanner, (3) rename `user_context` → `query_context` across all prompts and input models, (4) rename Reasoning's `task` → `subtask` in template and model, (5) add `original_task` to ToolSelector and Reasoning.

**Tech Stack:** Python (Pydantic v2), Jinja2 prompt templates, pytest-asyncio integration tests, Ollama (gpt-oss:20b).

---

## File Structure

| Task | Files Modified |
|------|---------------|
| 1 | `resources/prompts/planner_user_prompt.jinja2`, `resources/prompts/planner_system_prompt.jinja2`, `easylocai/llm_calls/planner.py`, `easylocai/agents/plan_agent.py`, `tests/integration_tests/agents/test_plan_agent.py` |
| 2 | `resources/prompts/replanner_user_prompt_c1_decision_tree_verbatim_plan.jinja2`, `resources/prompts/replanner_system_prompt_c1_decision_tree_verbatim_plan.jinja2`, `easylocai/llm_calls/replanner.py`, `easylocai/agents/replan_agent.py`, `tests/integration_tests/agents/test_replan_agent.py` |
| 3 | All 6 `*_user_prompt.jinja2` templates, 5 system prompt templates, 6 `easylocai/llm_calls/*.py` input models, `easylocai/agents/plan_agent.py`, `easylocai/agents/replan_agent.py`, `easylocai/agents/single_task_agent.py`, `easylocai/agents/reasoning_agent.py` |
| 4 | `resources/prompts/reasoning_user_prompt.jinja2`, `resources/prompts/reasoning_system_prompt.jinja2`, `easylocai/llm_calls/reasoning.py`, `easylocai/agents/reasoning_agent.py` |
| 5 | `resources/prompts/tool_selector_user_prompt.jinja2`, `resources/prompts/tool_selector_system_prompt.jinja2`, `resources/prompts/reasoning_user_prompt.jinja2`, `resources/prompts/reasoning_system_prompt.jinja2`, `easylocai/llm_calls/tool_selector.py`, `easylocai/llm_calls/reasoning.py`, `easylocai/agents/reasoning_agent.py`, `easylocai/agents/single_task_agent.py` |

---

## Task 1: Add `conversation_histories` to Planner

Without this, the Planner has no awareness of previous turns and may re-plan already-completed work in multi-turn sessions.

**Files:**
- Modify: `resources/prompts/planner_user_prompt.jinja2`
- Modify: `resources/prompts/planner_system_prompt.jinja2`
- Modify: `easylocai/llm_calls/planner.py`
- Modify: `easylocai/agents/plan_agent.py`
- Modify: `tests/integration_tests/agents/test_plan_agent.py`

- [ ] **Step 1.1: Write a failing test that checks `PlannerInput` has `conversation_histories` field**

Add to `tests/integration_tests/agents/test_plan_agent.py`:

```python
import pytest

from easylocai.agents.plan_agent import (
    PlanAgent,
    PlanAgentInput,
    PlanAgentOutput,
)
from easylocai.llm_calls.planner import PlannerInput
from easylocai.schemas.context import ConversationHistory, WorkflowContext


def assert_output(output: PlanAgentOutput) -> None:
    assert isinstance(output.task_list, list)
    assert len(output.task_list) > 0
    for task in output.task_list:
        assert isinstance(task, str)
        assert len(task) > 0
    assert output.reformatted_user_query
    assert output.query_context is None or isinstance(output.query_context, str)


class TestPlanAgent:

    @pytest.mark.asyncio
    async def test_basic_query(self, ollama_client):
        agent = PlanAgent(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="Find all Python files in the current directory and count them",
        )
        input_ = PlanAgentInput(workflow_context=workflow_context)
        output = await agent.run(input_)
        assert_output(output)

    @pytest.mark.asyncio
    async def test_query_with_conversation_history(self, ollama_client):
        agent = PlanAgent(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="Summarize it",
            conversation_histories=[
                ConversationHistory(
                    original_user_query="Find the README.md file",
                    reformatted_user_query="Find the README.md file",
                    response="Found README.md in the root directory.",
                )
            ],
        )
        input_ = PlanAgentInput(workflow_context=workflow_context)
        output = await agent.run(input_)
        assert_output(output)

    def test_planner_input_has_conversation_histories_field(self):
        # PlannerInput must accept conversation_histories before this test passes
        histories = [
            ConversationHistory(
                original_user_query="Find alice.txt",
                reformatted_user_query="Find alice.txt",
                response="Found alice.txt with 42 lines.",
            )
        ]
        inp = PlannerInput(
            user_query="Summarize alice.txt",
            user_context=None,
            conversation_histories=histories,
        )
        assert len(inp.conversation_histories) == 1
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
pytest tests/integration_tests/agents/test_plan_agent.py::TestPlanAgent::test_planner_input_has_conversation_histories_field -v
```

Expected: FAIL — `PlannerInput` does not accept `conversation_histories`.

- [ ] **Step 1.3: Add `conversation_histories` to `PlannerInput`**

Full content of `easylocai/llm_calls/planner.py`:

```python
from pydantic import BaseModel, Field

from easylocai.constants.model import GPT_OSS_20B
from easylocai.core.llm_call import LLMCallV2
from easylocai.schemas.context import ConversationHistory


class PlannerInput(BaseModel):
    user_query: str = Field(
        title="User Query",
        description="The user's objective/query that needs to be planned.",
    )
    user_context: str | None = Field(
        title="User Context",
        description="Additional context provided by the user.",
    )
    conversation_histories: list[ConversationHistory] = Field(
        default_factory=list,
        title="Conversation Histories",
        description="Previous conversation turns for multi-turn context.",
    )


class PlannerOutput(BaseModel):
    tasks: list[str] = Field(
        title="Tasks",
        description="A list of atomic, independent, simple, and semantic tasks",
    )


class Planner(LLMCallV2[PlannerInput, PlannerOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/planner_system_prompt.jinja2"
        user_prompt_path = "prompts/planner_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=PlannerOutput,
            options=options,
        )
```

- [ ] **Step 1.4: Update `planner_user_prompt.jinja2` to render conversation history**

Full content of `resources/prompts/planner_user_prompt.jinja2`:

```
{% if conversation_histories %}
CONVERSATION_HISTORY:
{% for h in conversation_histories %}
user: {{ h.original_user_query }}
assistant: {{ h.response }}

{% endfor %}
{% endif %}
USER_QUERY:
{{ user_query }}

USER_CONTEXT:
{{ user_context }}
```

- [ ] **Step 1.5: Update `planner_system_prompt.jinja2` to mention CONVERSATION_HISTORY**

Find the line:
```
- Only USER_CONTEXT is available information for planning. Do not use your own information.
```

Replace with (full file for reference — only this one line changes):

```
You are a plan agent.

Your goal is to produce a step-by-step tasks to accomplish the user query.
You must NOT solve the problem yourself — only output tasks.

- Each task should be atomic, independent, simple, semantic (Do one progress at a time).
 - If tasks can be executed in parallel as subtasks, merge them into a single task that indicates parallel execution.
  - e.g. "Read a.txt, b.txt, and c.txt files in parallel" instead of separate tasks for each file.
- USER_CONTEXT provides useful information about the user query. You should utilize USER_CONTEXT to create better tasks.
  - If USER_CONTEXT already contains the data needed (e.g. file contents, query results), **skip the data-acquisition task entirely** and proceed directly to the processing task.
  e.g. If USER_CONTEXT is "The contents of names.txt is 'Alice, Bob, Charlie'", and the user query is "Count the number of names in names.txt", you should create a task like "Count the number of names in names.txt in USER_CONTEXT contents" instead of creating multiple tasks like "Read names.txt", "Extract names", "Count names".
- CONVERSATION_HISTORY (if provided) shows previous turns of the session. Use it to understand follow-up queries and avoid re-fetching data already retrieved in prior turns.
  - If a prior turn already retrieved the needed data (e.g. file contents, search results), **skip the data-acquisition task** and use the existing result directly.
- Only USER_CONTEXT and CONVERSATION_HISTORY are available information for planning. Do not use your own information.
- The number of tasks should be **minimal but sufficient** — no superfluous actions.
- You will only use tools or reasoning agents that are available in the system. So, assume the tools and reasoning agents available in the system can be used to accomplish the tasks.
- DO NOT asks back to the user for clarification.
- If the user query includes logical problem solving such as math, physics problems, logical question, you should not reason task to solve the problem or solve it by yourself.
  e.g. If the user query is "Convert 72 Fahrenheit to Celsius", the plan should not be ["Subtract 32 from 72", "Multiply result by 5/9"] but rather ["Convert 72 Fahrenheit to Celsius"].
  - **IMPORTANT**: A math/logic computation is always its own atomic task. Each distinct action that follows (saving, sending, storing, etc.) is also a separate task. Do NOT merge the computation with any subsequent action.
  e.g. If the user query is "Convert 72 Fahrenheit to Celsius and save the result to output.txt", the plan should be ["Convert 72 Fahrenheit to Celsius", "Save the conversion result to output.txt"].
  e.g. If the user query is "Compute the area of a circle with radius 5, save it to area.txt, and send it via Slack", the plan should be ["Compute the area of a circle with radius 5", "Save the result to area.txt", "Send the result via Slack"].
- If the user query is a simple factual or knowledge question (e.g. "What is X?", "Where is Y?", "Who is Z?"), produce a single task to answer it. Do NOT add a separate "present the answer" or "retrieve more information" task.
  e.g. If the user query is "What is the population of Australia?", the plan should be ["Find the population of Australia"], not ["Retrieve the population of Australia", "Present the answer"].
- If the user query is a **pure** generation or creative task with no required data-acquisition step and no required file I/O step (e.g. "Create a to-do list", "Generate a meal plan", "Write an essay"), produce a single task. Do NOT decompose it into sub-tasks like "research", "create template", "populate", "format".
  e.g. If the user query is "Write a cover letter for a software engineer position", the plan should be ["Write a cover letter for a software engineer position"], not multiple sub-steps.
  e.g. If the user query is "Draft a weekly workout schedule for a beginner", the plan should be ["Draft a weekly workout schedule for a beginner"].
- When a task requires fetching external data (read files, search a database, call an API) **and that data is NOT already in USER_CONTEXT**, produce exactly two tasks: one to acquire the data, and one to process/use it.
  - The data-acquisition task covers all fetching needed (reading, searching, retrieving) — do NOT split it further.
  - The processing task covers all downstream work (summarizing, transforming, writing to a file) — do NOT split it further.
  e.g. "Read config.yaml and requirements.txt files in parallel" is task 1; "Summarize the contents of config.yaml and requirements.txt" is task 2.
  e.g. "Search Slack messages about the deployment outage" is task 1; "Summarize the found messages and save to outage_report.txt" is task 2.
- Do NOT add a "present the answer to the user" or "format the output" task — the system handles output presentation automatically.
```

- [ ] **Step 1.6: Pass `conversation_histories` in `plan_agent.py`**

Full content of `easylocai/agents/plan_agent.py`:

```python
import logging

from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.llm_calls.planner import Planner, PlannerInput, PlannerOutput
from easylocai.llm_calls.query_reformatter import (
    QueryReformatter,
    QueryReformatterInput,
    QueryReformatterOutput,
)
from easylocai.schemas.common import UserConversation
from easylocai.schemas.context import WorkflowContext

logger = logging.getLogger(__name__)


class PlanAgentInput(BaseModel):
    workflow_context: WorkflowContext


class PlanAgentOutput(BaseModel):
    query_context: str | None
    reformatted_user_query: str
    task_list: list[str]


class PlanAgent(Agent[PlanAgentInput, PlanAgentOutput]):
    def __init__(self, *, client: AsyncClient):
        self._ollama_client = client

    async def _run(self, input_: PlanAgentInput) -> PlanAgentOutput:
        ctx = input_.workflow_context

        previous_conversations = [
            UserConversation(
                user_query=h.original_user_query,
                assistant_answer=h.response,
            )
            for h in ctx.conversation_histories
        ]

        reformatter_input = QueryReformatterInput(
            user_query=ctx.original_user_query,
            previous_conversations=previous_conversations,
        )
        reformatter: QueryReformatter = QueryReformatter(client=self._ollama_client)
        reformatter_output: QueryReformatterOutput = await reformatter.call(reformatter_input)

        planner_input = PlannerInput(
            user_query=reformatter_output.reformed_query,
            user_context=reformatter_output.query_context,
            conversation_histories=ctx.conversation_histories,
        )
        planner = Planner(client=self._ollama_client)
        planner_output: PlannerOutput = await planner.call(planner_input)

        return PlanAgentOutput(
            query_context=reformatter_output.query_context,
            reformatted_user_query=reformatter_output.reformed_query,
            task_list=planner_output.tasks,
        )
```

- [ ] **Step 1.7: Run test to verify it passes**

```bash
pytest tests/integration_tests/agents/test_plan_agent.py -v
```

Expected: All 3 tests PASS.

- [ ] **Step 1.8: Commit**

```bash
git add resources/prompts/planner_user_prompt.jinja2 \
        resources/prompts/planner_system_prompt.jinja2 \
        easylocai/llm_calls/planner.py \
        easylocai/agents/plan_agent.py \
        tests/integration_tests/agents/test_plan_agent.py
git commit -m "feat: add conversation_histories to Planner for multi-turn context"
```

---

## Task 2: Add `conversation_histories` to Replanner

Without this, Replanner cannot see prior turns when deciding if the query is fully answered.

**Files:**
- Modify: `resources/prompts/replanner_user_prompt_c1_decision_tree_verbatim_plan.jinja2`
- Modify: `resources/prompts/replanner_system_prompt_c1_decision_tree_verbatim_plan.jinja2`
- Modify: `easylocai/llm_calls/replanner.py`
- Modify: `easylocai/agents/replan_agent.py`
- Modify: `tests/integration_tests/agents/test_replan_agent.py`

- [ ] **Step 2.1: Write a failing test that checks `ReplannerInput` has `conversation_histories` field**

Full content of `tests/integration_tests/agents/test_replan_agent.py`:

```python
import pytest

from easylocai.agents.replan_agent import (
    ReplanAgent,
    ReplanAgentInput,
    ReplanAgentOutput,
)
from easylocai.llm_calls.replanner import ReplannerInput
from easylocai.schemas.context import ConversationHistory, ExecutedTaskResult, WorkflowContext


class TestReplanAgent:

    @pytest.mark.asyncio
    async def test_replan_with_remaining_tasks(self, ollama_client):
        agent = ReplanAgent(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="Find all Python files and count lines of code",
            query_context=None,
            reformatted_user_query="Find all Python files and count lines of code",
            task_list=[
                "Find all Python files in the project",
                "Count total lines of code",
            ],
            executed_task_results=[
                ExecutedTaskResult(
                    executed_task="Find all Python files in the project",
                    result="Found 15 Python files",
                )
            ],
        )
        input_ = ReplanAgentInput(workflow_context=workflow_context)
        output: ReplanAgentOutput = await agent.run(input_)

        assert isinstance(output.tasks, list)
        assert output.response is None or isinstance(output.response, str)

    @pytest.mark.asyncio
    async def test_replan_all_tasks_complete(self, ollama_client):
        agent = ReplanAgent(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="List files in /tmp",
            query_context=None,
            reformatted_user_query="List files in /tmp",
            task_list=["List files in /tmp directory"],
            executed_task_results=[
                ExecutedTaskResult(
                    executed_task="List files in /tmp directory",
                    result="Files: a.txt, b.txt, c.txt",
                )
            ],
        )
        input_ = ReplanAgentInput(workflow_context=workflow_context)
        output: ReplanAgentOutput = await agent.run(input_)

        assert output.response is not None or len(output.tasks) > 0

    @pytest.mark.asyncio
    async def test_replan_with_conversation_history(self, ollama_client):
        agent = ReplanAgent(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="Now count how many there are",
            query_context=None,
            reformatted_user_query="Count the number of Python files",
            task_list=["Count the number of Python files"],
            executed_task_results=[
                ExecutedTaskResult(
                    executed_task="Count the number of Python files",
                    result="There are 15 Python files",
                )
            ],
            conversation_histories=[
                ConversationHistory(
                    original_user_query="Find all Python files",
                    reformatted_user_query="Find all Python files",
                    response="Found 15 Python files in the project.",
                )
            ],
        )
        input_ = ReplanAgentInput(workflow_context=workflow_context)
        output: ReplanAgentOutput = await agent.run(input_)

        assert output.response is not None

    def test_replanner_input_has_conversation_histories_field(self):
        histories = [
            ConversationHistory(
                original_user_query="Find alice.txt",
                reformatted_user_query="Find alice.txt",
                response="Found alice.txt.",
            )
        ]
        inp = ReplannerInput(
            user_context=None,
            original_user_query="Summarize alice.txt",
            previous_plan=["Summarize alice.txt"],
            task_results=[{"task": "Summarize alice.txt", "result": "It is a short file."}],
            conversation_histories=histories,
        )
        assert len(inp.conversation_histories) == 1
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
pytest tests/integration_tests/agents/test_replan_agent.py::TestReplanAgent::test_replanner_input_has_conversation_histories_field -v
```

Expected: FAIL — `ReplannerInput` does not accept `conversation_histories`.

- [ ] **Step 2.3: Add `conversation_histories` to `ReplannerInput`**

Full content of `easylocai/llm_calls/replanner.py`:

```python
from pydantic import BaseModel, Field

from easylocai.constants.model import GPT_OSS_20B
from easylocai.core.llm_call import LLMCallV2
from easylocai.schemas.context import ConversationHistory


class ReplannerInput(BaseModel):
    user_context: str | None
    original_user_query: str
    previous_plan: list[str]
    task_results: list[dict]
    conversation_histories: list[ConversationHistory] = Field(default_factory=list)


class ReplannerOutput(BaseModel):
    tasks: list[str] = Field(
        description="A list of remaining tasks to complete the user query"
    )
    response: str | None = Field(
        description="Final response to the user if all tasks are completed. None if tasks remain."
    )


class Replanner(LLMCallV2[ReplannerInput, ReplannerOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/replanner_system_prompt_c1_decision_tree_verbatim_plan.jinja2"
        user_prompt_path = "prompts/replanner_user_prompt_c1_decision_tree_verbatim_plan.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ReplannerOutput,
            options=options,
        )
```

- [ ] **Step 2.4: Update replanner user prompt to render conversation history**

Full content of `resources/prompts/replanner_user_prompt_c1_decision_tree_verbatim_plan.jinja2`:

```
{% if conversation_histories %}
CONVERSATION_HISTORY:
{% for h in conversation_histories %}
user: {{ h.original_user_query }}
assistant: {{ h.response }}

{% endfor %}
{% endif %}
USER_CONTEXT:
{{ user_context }}

ORIGINAL_USER_QUERY:
{{ original_user_query }}

PREVIOUS_PLAN:
{% for task in previous_plan %}
{{ loop.index }}. {{ task }}
{% endfor %}

COMPLETED_TASKS ({{ task_results | length }} of {{ previous_plan | length }} done):
{% if task_results %}
{% for task_result in task_results %}
<task_result>
- task: {{ task_result["task"] }}
- result: {{ task_result["result"] }}
</task_result>
{% endfor %}
{% else %}
(none completed yet)
{% endif %}

Based on the above, determine what to do next. Return ONLY remaining tasks from PREVIOUS_PLAN unless the query is already answerable.
```

- [ ] **Step 2.5: Update replanner system prompt to mention CONVERSATION_HISTORY**

Full content of `resources/prompts/replanner_system_prompt_c1_decision_tree_verbatim_plan.jinja2`:

```
You are a re-plan agent.

Your goal is to determine what tasks remain to complete the user's original query, based on the progress so far.

## Decision Rules (apply in order)

1. **Check CONVERSATION_HISTORY and USER_CONTEXT first.** If CONVERSATION_HISTORY or USER_CONTEXT already contains the information needed to answer the original query directly, set `tasks` to an empty list and set `response` to the answer drawn from that context. Do NOT create tasks to re-fetch information that is already present.

2. **Check if all tasks are done.** Compare COMPLETED_TASKS against PREVIOUS_PLAN. If every task in PREVIOUS_PLAN is completed and the result is ready to deliver to the user:
   - Set `tasks` to an empty list
   - Set `response` to the final answer synthesized from the completed task results

3. **Return remaining tasks from the original plan.** If tasks still remain, identify which tasks from PREVIOUS_PLAN have NOT yet been completed. Return ONLY those remaining tasks — do NOT invent new tasks or rephrase the original tasks unless a completed result makes a remaining task clearly unnecessary or impossible.

4. **Missing information.** If the plan cannot proceed due to genuinely missing information (not resolvable from CONVERSATION_HISTORY, USER_CONTEXT, or completed results), return an empty task list and explain in `response`.

## Output Rules

- When tasks remain: set `tasks` to the list of remaining task descriptions (from PREVIOUS_PLAN), set `response` to null
- When done: set `tasks` to an empty list, set `response` to the final answer
- NEVER set both `tasks` (non-empty) and `response` (non-null) at the same time
- Tasks must be simple string descriptions, not structured objects
- Each task should be atomic and specific

## What NOT to do

- Do NOT solve tasks yourself — only identify what remains
- Do NOT add new tasks that weren't in the original plan unless absolutely required
- Do NOT re-fetch information already present in CONVERSATION_HISTORY, USER_CONTEXT, or completed task results
- Do NOT split a logical-reasoning task (math, physics) into sub-steps — keep it as one task
```

- [ ] **Step 2.6: Pass `conversation_histories` in `replan_agent.py`**

Full content of `easylocai/agents/replan_agent.py`:

```python
import logging

from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.llm_calls.replanner import Replanner, ReplannerInput, ReplannerOutput
from easylocai.schemas.context import WorkflowContext

logger = logging.getLogger(__name__)


class ReplanAgentInput(BaseModel):
    workflow_context: WorkflowContext


class ReplanAgentOutput(BaseModel):
    tasks: list[str]
    response: str | None


class ReplanAgent(Agent[ReplanAgentInput, ReplanAgentOutput]):
    def __init__(self, *, client: AsyncClient):
        self._ollama_client = client

    async def _run(self, input_: ReplanAgentInput) -> ReplanAgentOutput:
        ctx = input_.workflow_context

        task_results = [
            {"task": r.executed_task, "result": r.result}
            for r in ctx.executed_task_results
        ]

        replanner_input = ReplannerInput(
            user_context=ctx.query_context,
            original_user_query=ctx.original_user_query,
            previous_plan=ctx.task_list,
            task_results=task_results,
            conversation_histories=ctx.conversation_histories,
        )

        replanner = Replanner(client=self._ollama_client)
        replanner_output: ReplannerOutput = await replanner.call(replanner_input)

        logger.debug(f"ReplanAgent output: {replanner_output}")

        return ReplanAgentOutput(
            tasks=replanner_output.tasks,
            response=replanner_output.response,
        )
```

- [ ] **Step 2.7: Run all replan agent tests**

```bash
pytest tests/integration_tests/agents/test_replan_agent.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 2.8: Commit**

```bash
git add resources/prompts/replanner_user_prompt_c1_decision_tree_verbatim_plan.jinja2 \
        resources/prompts/replanner_system_prompt_c1_decision_tree_verbatim_plan.jinja2 \
        easylocai/llm_calls/replanner.py \
        easylocai/agents/replan_agent.py \
        tests/integration_tests/agents/test_replan_agent.py
git commit -m "feat: add conversation_histories to Replanner for multi-turn context"
```

---

## Task 3: Rename `user_context` → `query_context` Everywhere

`user_context` is a misleading name — the value is `query_context` from `QueryReformatter` (preamble only, usually null). Rename it throughout so that the template variable name matches what it actually contains.

**Files:** All 6 `*_user_prompt.jinja2`, system prompts that mention `USER_CONTEXT`, 6 `LLMCallV2` input models, and 4 agent files.

- [ ] **Step 3.1: Run full test suite to capture baseline (all must pass)**

```bash
pytest -v
```

Expected: All tests PASS. Record this as the baseline.

- [ ] **Step 3.2: Rename in `planner_user_prompt.jinja2`**

Full content of `resources/prompts/planner_user_prompt.jinja2`:

```
{% if conversation_histories %}
CONVERSATION_HISTORY:
{% for h in conversation_histories %}
user: {{ h.original_user_query }}
assistant: {{ h.response }}

{% endfor %}
{% endif %}
USER_QUERY:
{{ user_query }}

QUERY_CONTEXT:
{{ query_context }}
```

- [ ] **Step 3.3: Rename in `planner_system_prompt.jinja2`**

Replace all occurrences of `USER_CONTEXT` with `QUERY_CONTEXT`. The relevant lines to change (rest of file unchanged):

- Line: `- USER_CONTEXT provides useful information about the user query. You should utilize USER_CONTEXT to create better tasks.`  
  → `- QUERY_CONTEXT provides useful information about the user query. You should utilize QUERY_CONTEXT to create better tasks.`

- Line: `  - If USER_CONTEXT already contains the data needed...`  
  → `  - If QUERY_CONTEXT already contains the data needed...`

- Line: `  e.g. If USER_CONTEXT is "The contents of names.txt is 'Alice, Bob, Charlie'"...`  
  → `  e.g. If QUERY_CONTEXT is "The contents of names.txt is 'Alice, Bob, Charlie'"...`

- Line: `- Only USER_CONTEXT and CONVERSATION_HISTORY are available information for planning. Do not use your own information.`  
  → `- Only QUERY_CONTEXT and CONVERSATION_HISTORY are available information for planning. Do not use your own information.`

- Line: `  - If a prior turn already retrieved the needed data...`  
  → unchanged

Full content after update:

```
You are a plan agent.

Your goal is to produce a step-by-step tasks to accomplish the user query.
You must NOT solve the problem yourself — only output tasks.

- Each task should be atomic, independent, simple, semantic (Do one progress at a time).
 - If tasks can be executed in parallel as subtasks, merge them into a single task that indicates parallel execution.
  - e.g. "Read a.txt, b.txt, and c.txt files in parallel" instead of separate tasks for each file.
- QUERY_CONTEXT provides useful information about the user query. You should utilize QUERY_CONTEXT to create better tasks.
  - If QUERY_CONTEXT already contains the data needed (e.g. file contents, query results), **skip the data-acquisition task entirely** and proceed directly to the processing task.
  e.g. If QUERY_CONTEXT is "The contents of names.txt is 'Alice, Bob, Charlie'", and the user query is "Count the number of names in names.txt", you should create a task like "Count the number of names in names.txt in QUERY_CONTEXT contents" instead of creating multiple tasks like "Read names.txt", "Extract names", "Count names".
- CONVERSATION_HISTORY (if provided) shows previous turns of the session. Use it to understand follow-up queries and avoid re-fetching data already retrieved in prior turns.
  - If a prior turn already retrieved the needed data (e.g. file contents, search results), **skip the data-acquisition task** and use the existing result directly.
- Only QUERY_CONTEXT and CONVERSATION_HISTORY are available information for planning. Do not use your own information.
- The number of tasks should be **minimal but sufficient** — no superfluous actions.
- You will only use tools or reasoning agents that are available in the system. So, assume the tools and reasoning agents available in the system can be used to accomplish the tasks.
- DO NOT asks back to the user for clarification.
- If the user query includes logical problem solving such as math, physics problems, logical question, you should not reason task to solve the problem or solve it by yourself.
  e.g. If the user query is "Convert 72 Fahrenheit to Celsius", the plan should not be ["Subtract 32 from 72", "Multiply result by 5/9"] but rather ["Convert 72 Fahrenheit to Celsius"].
  - **IMPORTANT**: A math/logic computation is always its own atomic task. Each distinct action that follows (saving, sending, storing, etc.) is also a separate task. Do NOT merge the computation with any subsequent action.
  e.g. If the user query is "Convert 72 Fahrenheit to Celsius and save the result to output.txt", the plan should be ["Convert 72 Fahrenheit to Celsius", "Save the conversion result to output.txt"].
  e.g. If the user query is "Compute the area of a circle with radius 5, save it to area.txt, and send it via Slack", the plan should be ["Compute the area of a circle with radius 5", "Save the result to area.txt", "Send the result via Slack"].
- If the user query is a simple factual or knowledge question (e.g. "What is X?", "Where is Y?", "Who is Z?"), produce a single task to answer it. Do NOT add a separate "present the answer" or "retrieve more information" task.
  e.g. If the user query is "What is the population of Australia?", the plan should be ["Find the population of Australia"], not ["Retrieve the population of Australia", "Present the answer"].
- If the user query is a **pure** generation or creative task with no required data-acquisition step and no required file I/O step (e.g. "Create a to-do list", "Generate a meal plan", "Write an essay"), produce a single task. Do NOT decompose it into sub-tasks like "research", "create template", "populate", "format".
  e.g. If the user query is "Write a cover letter for a software engineer position", the plan should be ["Write a cover letter for a software engineer position"], not multiple sub-steps.
  e.g. If the user query is "Draft a weekly workout schedule for a beginner", the plan should be ["Draft a weekly workout schedule for a beginner"].
- When a task requires fetching external data (read files, search a database, call an API) **and that data is NOT already in QUERY_CONTEXT**, produce exactly two tasks: one to acquire the data, and one to process/use it.
  - The data-acquisition task covers all fetching needed (reading, searching, retrieving) — do NOT split it further.
  - The processing task covers all downstream work (summarizing, transforming, writing to a file) — do NOT split it further.
  e.g. "Read config.yaml and requirements.txt files in parallel" is task 1; "Summarize the contents of config.yaml and requirements.txt" is task 2.
  e.g. "Search Slack messages about the deployment outage" is task 1; "Summarize the found messages and save to outage_report.txt" is task 2.
- Do NOT add a "present the answer to the user" or "format the output" task — the system handles output presentation automatically.
```

- [ ] **Step 3.4: Rename in `PlannerInput` model**

In `easylocai/llm_calls/planner.py`, change the `user_context` field to `query_context`:

```python
from pydantic import BaseModel, Field

from easylocai.constants.model import GPT_OSS_20B
from easylocai.core.llm_call import LLMCallV2
from easylocai.schemas.context import ConversationHistory


class PlannerInput(BaseModel):
    user_query: str = Field(
        title="User Query",
        description="The user's objective/query that needs to be planned.",
    )
    query_context: str | None = Field(
        title="Query Context",
        description="Preamble context extracted from the current query by QueryReformatter.",
    )
    conversation_histories: list[ConversationHistory] = Field(
        default_factory=list,
        title="Conversation Histories",
        description="Previous conversation turns for multi-turn context.",
    )


class PlannerOutput(BaseModel):
    tasks: list[str] = Field(
        title="Tasks",
        description="A list of atomic, independent, simple, and semantic tasks",
    )


class Planner(LLMCallV2[PlannerInput, PlannerOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/planner_system_prompt.jinja2"
        user_prompt_path = "prompts/planner_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=PlannerOutput,
            options=options,
        )
```

- [ ] **Step 3.5: Update `plan_agent.py` to use `query_context=`**

Change the `PlannerInput(...)` construction in `easylocai/agents/plan_agent.py`:

```python
        planner_input = PlannerInput(
            user_query=reformatter_output.reformed_query,
            query_context=reformatter_output.query_context,
            conversation_histories=ctx.conversation_histories,
        )
```

(Only this constructor call changes; the rest of the file is identical to Step 1.6.)

- [ ] **Step 3.6: Rename in `task_router_user_prompt.jinja2`**

Full content of `resources/prompts/task_router_user_prompt.jinja2`:

```
Task:
{{ task }}

QUERY_CONTEXT:
{{ query_context }}

TOOL_CANDIDATES:
{% for tool in tool_candidates %}
<tool>
- server_name: {{ tool["server_name"] }}
- tool_name: {{ tool["tool_name"] }}
- tool_description: {{ tool["tool_description"] }}
</tool>
{% endfor %}

PREVIOUS_TASK_RESULTS (chronological order):
{% for task_result in previous_task_results %}
<task_result>
- task: {{ task_result["task"] }}
- result: {{ task_result["result"] }}
</task_result>
{% endfor %}

PREVIOUS_SUBTASK_RESULTS (chronological order):
{% for iteration_result in iteration_results %}
<subtask_result>
    - subtask:
    {{ iteration_result["subtask"] }}
    - result:
    {{ iteration_result["result"] }}
</subtask_result>
{% endfor %}
```

- [ ] **Step 3.7: Rename `USER_CONTEXT` in `task_router_system_prompt.jinja2`**

Change this line:
```
- USER_CONTEXT: Additional context about the user's intent
```
To:
```
- QUERY_CONTEXT: Preamble context extracted from the current query (may be null)
```

Full content of `resources/prompts/task_router_system_prompt.jinja2`:

```
** DO NOT RETURN EMPTY RESPONSE IN ANY CASE **

You are a task router assistant.

Your job is to determine the next subtask and route it to the appropriate handler based on the given context.

## Input Context
- TASK: The main task to complete
- QUERY_CONTEXT: Preamble context extracted from the current query (may be null)
- TOOL_CANDIDATES: Available tools that can be used
- PREVIOUS_TASK_RESULTS: Results from previously completed tasks
- PREVIOUS_SUBTASK_RESULTS: Results from subtasks in the current task iteration

## Your Responsibilities
1. **Analyze**: the current state based on previous results
2. **Determine**: the next subtask needed to progress toward completing the task. Subtask should be specific and actionable. Do not refer tool name in the subtask.
3. **Route**: the subtask to the appropriate handler:
   - `"tool"`: When the subtask requires external actions (file operations, API calls, data retrieval)
   - `"reasoning"`: When the subtask requires analysis, summarization, or generating conclusions
4. **Finish** when the task is fully completed or failed

## Routing Guidelines

Choose `"tool"` when:
- Reading or writing files
- Searching for information
- Executing external operations
- Any action requiring tool capabilities
- If tool capability is conflicted with reasoning needs, prioritize tool usage

Choose `"reasoning"` when:
- Summarizing collected information
- Analyzing data or content
- Drawing conclusions
- Generating final answers

Set `finished=true` when:
- The task has been fully addressed or cannot be completed due to limitations.
- All necessary information has been gathered and processed
```

- [ ] **Step 3.8: Rename `user_context` field in `TaskRouterInput`**

Full content of `easylocai/llm_calls/task_router.py`:

```python
from typing import Literal

from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2


class TaskRouterInput(BaseModel):
    task: str
    query_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class Subtask(BaseModel):
    subtask: str = Field(
        description="The next subtask to execute. Do not refer to tool names directly here."
    )
    subtask_type: Literal["tool", "reasoning"] = Field(
        description="'tool' for tool-based execution, 'reasoning' for reasoning subtask (e.g. coding, problem solving, inference)."
    )


class TaskRouterOutput(BaseModel):
    subtask: str | None = Field(
        description="The next subtask to execute. None if finished=True. Do not refer to tool names directly here."
    )
    subtask_type: Literal["tool", "reasoning"] | None = Field(
        description="'tool' for tool-based execution, 'reasoning' for reasoning agent. None if finished=True"
    )
    finished: bool = Field(description="Whether the task is completed or failed.")
    finished_reason: str | None = Field(
        description="Explanation of why the task is finished. None if finished is False."
    )


class TaskRouterOutputV2(BaseModel):
    subtasks: list[Subtask] = Field(
        description="A list of next subtasks which can be executed in parallel to achieve the goal of task. Empty if finished=True."
    )
    finished: bool = Field(description="Whether the task is completed or failed.")
    finished_reason: str | None = Field(
        description="Explanation of why the task is finished. None if finished is False."
    )


class TaskRouter(LLMCallV2[TaskRouterInput, TaskRouterOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/task_router_system_prompt.jinja2"
        user_prompt_path = "prompts/task_router_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=TaskRouterOutput,
            options=options,
        )
```

- [ ] **Step 3.9: Rename in `tool_selector_user_prompt.jinja2`**

Full content of `resources/prompts/tool_selector_user_prompt.jinja2`:

```
SUBTASK:
{{ subtask }}

QUERY_CONTEXT:
{{ query_context }}

TOOL_CANDIDATES:
{% for tool in tool_candidates %}
<tool>
- server_name: {{ tool["server_name"] }}
- tool_name: {{ tool["tool_name"] }}
- tool_description: {{ tool["tool_description"] }}
- input_schema: {{ tool["tool_input_schema"] }}
</tool>
{% endfor %}

PREVIOUS_TASK_RESULTS (chronological order):
{% for task_result in previous_task_results %}
<task_result>
- task: {{ task_result["task"] }}
- result: {{ task_result["result"] }}
</task_result>
{% endfor %}

PREVIOUS_SUBTASK_RESULTS (chronological order):
{% for iteration_result in iteration_results %}
<subtask_result>
    - subtask:
    {{ iteration_result["subtask"] }}
    - result:
    {{ iteration_result["result"] }}
</subtask_result>
{% endfor %}
```

- [ ] **Step 3.10: Rename `USER_CONTEXT` in `tool_selector_system_prompt.jinja2`**

Full content of `resources/prompts/tool_selector_system_prompt.jinja2`:

```
You are a tool selector assistant.

Your job is to select the appropriate tool and configure its arguments to execute the given subtask.

## Input Context
- SUBTASK: The main task to complete
- QUERY_CONTEXT: Preamble context extracted from the current query (may be null)
- TOOL_CANDIDATES: Available tools that can be used
- PREVIOUS_TASK_RESULTS: Results from previously completed tasks
- PREVIOUS_SUBTASK_RESULTS: Results from subtasks in the current task iteration

## Your Responsibilities
1. **Analyze**: the subtask requirements and available tool capabilities
2. **Select**: one tool from TOOL_CANDIDATES that best accomplish the subtask
3. **Configure**: the arguments for the selected tool based on the context

## Selection Guidelines

When selecting a tool:
- Choose the tool whose description best matches the subtask requirements
- Refer to the tool's `input_schema` for required and optional parameters

When configuring arguments:
- Extract relevant values from the subtask description and context
- Provide all required parameters as defined in the tool's input_schema
- Include optional parameters only when explicitly needed

When no matching tool is found:
- Set `selected_tool` to null
- Provide a clear `failure_reason` explaining why no tool matches the subtask

## Output Format
Respond in JSON format with the given structure. Empty output is not allowed.
```

- [ ] **Step 3.11: Rename `user_context` field in `ToolSelectorInput`**

In `easylocai/llm_calls/tool_selector.py`, rename `user_context` → `query_context` in both `ToolSelectorV2Input` and `ToolSelectorInput`:

```python
from typing import Any

from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2


class ToolInput(BaseModel):
    server_name: str = Field(description="The server_name of the selected tool")
    tool_name: str = Field(
        description="The tool_name selected based on subtask requirements"
    )
    tool_args: dict[str, Any] = Field(
        description="Parameters for tool execution following the tool's input_schema. Do not include extra parameters not defined in the tool's input_schema."
    )


class ToolSelectorV2Input(BaseModel):
    subtask: str
    query_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class ToolSelectorOutput(BaseModel):
    selected_tool: ToolInput | None = Field(
        description="The tool selected to execute the given subtask. None if no matching tool found."
    )
    failure_reason: str | None = Field(
        description="Reason why no tool was selected. None if a tool was successfully selected."
    )


class ToolSelectorInput(BaseModel):
    subtask: str
    query_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class ToolSelectorInputV2(BaseModel):
    subtasks: list[str]
    query_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class SubtaskWithTool(BaseModel):
    subtask: str = Field(description="The actual subtask for which tool is selected.")
    selected_tool: ToolInput | None = Field(
        description="The tool selected to execute the given subtask. None if no matching tool found."
    )
    failure_reason: str | None = Field(
        description="Reason why tool can not be selected. None if a tool was successfully selected."
    )


class ToolSelectorOutputV2(BaseModel):
    results: list[SubtaskWithTool]


class ToolSelector(LLMCallV2[ToolSelectorInput, ToolSelectorOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/tool_selector_system_prompt.jinja2"
        user_prompt_path = "prompts/tool_selector_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ToolSelectorOutput,
            options=options,
        )
```

- [ ] **Step 3.12: Rename in `reasoning_user_prompt.jinja2`**

Full content of `resources/prompts/reasoning_user_prompt.jinja2`:

```
The Task:
{{ task }}

QUERY_CONTEXT:
{{ query_context }}

PREVIOUS_TASK_RESULTS:
{% for task_result in previous_task_results %}
<task_result>
- task: {{ task_result["task"] }}
- result: {{ task_result["result"] }}
</task_result>
{% endfor %}

PREVIOUS_SUBTASK_RESULTS:
{% for subtask_result in previous_subtask_results %}
<subtask_result>
    - subtask:
    {{ subtask_result["subtask"] }}
    - result:
    {{ subtask_result["result"] }}
</subtask_result>
{% endfor %}
```

- [ ] **Step 3.13: Rename `USER_CONTEXT` in `reasoning_system_prompt.jinja2`**

Change this line:
```
- Gather relevant information from USER_CONTEXT, PREVIOUS_TASK_RESULTS, and PREVIOUS_SUBTASK_RESULTS if needed
```
To:
```
- Gather relevant information from QUERY_CONTEXT, PREVIOUS_TASK_RESULTS, and PREVIOUS_SUBTASK_RESULTS if needed
```

Full content of `resources/prompts/reasoning_system_prompt.jinja2`:

```
You are a REASONING AGENT that solves tasks using deliberate, step-by-step thinking. You excel at arithmetic, algebra, geometry, and word problems. You never skip steps and always verify your arithmetic before answering.

## Goals
- Understand the task deeply
- Gather relevant information from QUERY_CONTEXT, PREVIOUS_TASK_RESULTS, and PREVIOUS_SUBTASK_RESULTS if needed
- Analyse options and trade-offs logically
- Produce the best final answer with high reliability

## Reasoning Rules
1. Think step-by-step with no big jumps. For math problems, show every arithmetic operation explicitly.
2. For math: compute intermediate results and double-check them before proceeding to the next step.
3. Consider multiple possibilities; eliminate wrong ones with justification.
4. If information is missing, state the assumptions explicitly.
5. If uncertain, say "Not enough information" instead of inventing facts. `confidence` should be 0 in such cases.
6. Keep reasoning inside `reasoning` field. Never include reasoning in `final`.
7. The `final` field must be short and directly answer the user. For math problems, state just the numeric result (with units if applicable).

## Output Format: STRICTLY JSON ONLY
{
  "reasoning": "<string> Step-by-step thinking, hypotheses, alternatives, and decision path",
  "final": "<string> Final answer for the user (no reasoning). Answer must be concise and actionable.",
  "confidence": <integer 0–100>
}

### Examples

Task: "Calculate (2 + 3) * 5"
Response:
{
  "reasoning": "Step 1: Add 2 + 3 = 5. Step 2: Multiply 5 * 5 = 25.",
  "final": "25",
  "confidence": 100
}

Task: "What is 15% of 200?"
Response:
{
  "reasoning": "Step 1: Convert 15% to decimal: 15/100 = 0.15. Step 2: Multiply 0.15 * 200 = 30.",
  "final": "30",
  "confidence": 100
}

Task: "A train travels 300 km in 4 hours. What is its average speed?"
Response:
{
  "reasoning": "Step 1: Average speed = distance / time = 300 / 4 = 75. Step 2: Units are km/h.",
  "final": "75 km/h",
  "confidence": 100
}
```

- [ ] **Step 3.14: Rename `user_context` field in `ReasoningInput`**

In `easylocai/llm_calls/reasoning.py`:

```python
from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2


class ReasoningInput(BaseModel):
    task: str = Field(
        title="Task",
        description="The task to reason about.",
    )
    query_context: str | None = Field(
        title="Query Context",
        description="Preamble context extracted from the current query by QueryReformatter.",
    )
    previous_task_results: list[dict] = Field(
        title="Previous Task Results",
        description="Results from previous tasks.",
    )
    previous_subtask_results: list[dict] = Field(
        default_factory=list,
        title="Previous Subtask Results",
        description="Results from previous subtasks within the current task.",
    )


class ReasoningOutput(BaseModel):
    reasoning: str = Field(
        title="Reasoning",
        description="The reasoning process.",
    )
    final: str = Field(
        title="Final",
        description="The final answer.",
    )
    confidence: int = Field(
        title="Confidence",
        description="Confidence level of the answer.",
    )


class Reasoning(LLMCallV2[ReasoningInput, ReasoningOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/reasoning_system_prompt.jinja2"
        user_prompt_path = "prompts/reasoning_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ReasoningOutput,
            options=options,
        )
```

- [ ] **Step 3.15: Rename in `task_result_filter_user_prompt.jinja2`**

Full content of `resources/prompts/task_result_filter_user_prompt.jinja2`:

```
TASK:
{{ task }}

QUERY_CONTEXT:
{{ query_context }}

SUBTASK_RESULTS:
{% for subtask_result in subtask_results %}
<subtask_result>
    - subtask:
    {{ subtask_result["subtask"] }}
    - result:
    {{ subtask_result["result"] }}
</subtask_result>
{% endfor %}
```

- [ ] **Step 3.16: Rename `USER_CONTEXT` in `task_result_filter_system_prompt.jinja2`**

Change `USER_CONTEXT` to `QUERY_CONTEXT`. Full content:

```
You are a precise task-result filter assistant.

Your job is to read the TASK, QUERY_CONTEXT, and an subtask results based on task, then return ONLY the information relevant to the task.

Filter Guidelines:
1. Parse the task result:
   - Try to interpret it as JSON (array, object, or JSONL).
   - If parsing fails, still attempt to extract useful records.
   - Handle arrays, nested objects, and mixed content gracefully.

2. Filter for relevance:
   - Derive inclusion criteria directly from the TASK.
   - Include only the items that satisfy those criteria.
   - If no items match, explicitly state original result and why they do not match.

Return filtered results based on Filter Guideline. Make it structured plain text.
If original tool result is about error or failure, return the failure reason.
If no results match, return structured text of actual result with a clear statement of "No relevant items found" is added.
```

- [ ] **Step 3.17: Rename `user_context` field in `TaskResultFilterInput`**

Full content of `easylocai/llm_calls/task_result_filter.py`:

```python
from pydantic import BaseModel
from pydantic import RootModel, Field

from easylocai.constants.model import GPT_OSS_20B
from easylocai.core.llm_call import LLMCallV2


class TaskResultFilterInput(BaseModel):
    task: str
    subtask_results: list[dict]
    query_context: str | None


class TaskResultFilterOutput(RootModel[str]):
    root: str = Field(description="Filtered task result")


class TaskResultFilter(LLMCallV2[TaskResultFilterInput, TaskResultFilterOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/task_result_filter_system_prompt.jinja2"
        user_prompt_path = "prompts/task_result_filter_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=TaskResultFilterOutput,
            options=options,
        )
```

- [ ] **Step 3.18: Rename in `replanner_user_prompt_c1_decision_tree_verbatim_plan.jinja2`**

Full content of `resources/prompts/replanner_user_prompt_c1_decision_tree_verbatim_plan.jinja2`:

```
{% if conversation_histories %}
CONVERSATION_HISTORY:
{% for h in conversation_histories %}
user: {{ h.original_user_query }}
assistant: {{ h.response }}

{% endfor %}
{% endif %}
QUERY_CONTEXT:
{{ query_context }}

ORIGINAL_USER_QUERY:
{{ original_user_query }}

PREVIOUS_PLAN:
{% for task in previous_plan %}
{{ loop.index }}. {{ task }}
{% endfor %}

COMPLETED_TASKS ({{ task_results | length }} of {{ previous_plan | length }} done):
{% if task_results %}
{% for task_result in task_results %}
<task_result>
- task: {{ task_result["task"] }}
- result: {{ task_result["result"] }}
</task_result>
{% endfor %}
{% else %}
(none completed yet)
{% endif %}

Based on the above, determine what to do next. Return ONLY remaining tasks from PREVIOUS_PLAN unless the query is already answerable.
```

- [ ] **Step 3.19: Rename `USER_CONTEXT` in `replanner_system_prompt_c1_decision_tree_verbatim_plan.jinja2`**

Replace all `USER_CONTEXT` with `QUERY_CONTEXT`. Full content:

```
You are a re-plan agent.

Your goal is to determine what tasks remain to complete the user's original query, based on the progress so far.

## Decision Rules (apply in order)

1. **Check CONVERSATION_HISTORY and QUERY_CONTEXT first.** If CONVERSATION_HISTORY or QUERY_CONTEXT already contains the information needed to answer the original query directly, set `tasks` to an empty list and set `response` to the answer drawn from that context. Do NOT create tasks to re-fetch information that is already present.

2. **Check if all tasks are done.** Compare COMPLETED_TASKS against PREVIOUS_PLAN. If every task in PREVIOUS_PLAN is completed and the result is ready to deliver to the user:
   - Set `tasks` to an empty list
   - Set `response` to the final answer synthesized from the completed task results

3. **Return remaining tasks from the original plan.** If tasks still remain, identify which tasks from PREVIOUS_PLAN have NOT yet been completed. Return ONLY those remaining tasks — do NOT invent new tasks or rephrase the original tasks unless a completed result makes a remaining task clearly unnecessary or impossible.

4. **Missing information.** If the plan cannot proceed due to genuinely missing information (not resolvable from CONVERSATION_HISTORY, QUERY_CONTEXT, or completed results), return an empty task list and explain in `response`.

## Output Rules

- When tasks remain: set `tasks` to the list of remaining task descriptions (from PREVIOUS_PLAN), set `response` to null
- When done: set `tasks` to an empty list, set `response` to the final answer
- NEVER set both `tasks` (non-empty) and `response` (non-null) at the same time
- Tasks must be simple string descriptions, not structured objects
- Each task should be atomic and specific

## What NOT to do

- Do NOT solve tasks yourself — only identify what remains
- Do NOT add new tasks that weren't in the original plan unless absolutely required
- Do NOT re-fetch information already present in CONVERSATION_HISTORY, QUERY_CONTEXT, or completed task results
- Do NOT split a logical-reasoning task (math, physics) into sub-steps — keep it as one task
```

- [ ] **Step 3.20: Rename `user_context` field in `ReplannerInput`**

In `easylocai/llm_calls/replanner.py`, rename `user_context` to `query_context`:

```python
from pydantic import BaseModel, Field

from easylocai.constants.model import GPT_OSS_20B
from easylocai.core.llm_call import LLMCallV2
from easylocai.schemas.context import ConversationHistory


class ReplannerInput(BaseModel):
    query_context: str | None
    original_user_query: str
    previous_plan: list[str]
    task_results: list[dict]
    conversation_histories: list[ConversationHistory] = Field(default_factory=list)


class ReplannerOutput(BaseModel):
    tasks: list[str] = Field(
        description="A list of remaining tasks to complete the user query"
    )
    response: str | None = Field(
        description="Final response to the user if all tasks are completed. None if tasks remain."
    )


class Replanner(LLMCallV2[ReplannerInput, ReplannerOutput]):
    def __init__(self, *, client):
        model = GPT_OSS_20B
        system_prompt_path = "prompts/replanner_system_prompt_c1_decision_tree_verbatim_plan.jinja2"
        user_prompt_path = "prompts/replanner_user_prompt_c1_decision_tree_verbatim_plan.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ReplannerOutput,
            options=options,
        )
```

- [ ] **Step 3.21: Update all caller agents — `replan_agent.py`**

Change `user_context=ctx.query_context` → `query_context=ctx.query_context` in the `ReplannerInput(...)` call:

```python
        replanner_input = ReplannerInput(
            query_context=ctx.query_context,
            original_user_query=ctx.original_user_query,
            previous_plan=ctx.task_list,
            task_results=task_results,
            conversation_histories=ctx.conversation_histories,
        )
```

(Full file otherwise identical to Step 2.6.)

- [ ] **Step 3.22: Update all caller agents — `reasoning_agent.py`**

Rename `user_context=` → `query_context=` in `ReasoningInput(...)` construction. Also rename `user_context` field in `ReasoningAgentInput`:

```python
import logging

from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.core.contants import DEFAULT_LLM_MODEL
from easylocai.llm_calls.reasoning import Reasoning, ReasoningInput, ReasoningOutput

logger = logging.getLogger(__name__)


class ReasoningAgentInput(BaseModel):
    task: dict
    query_context: str | None
    previous_task_results: list[dict]
    previous_subtask_results: list[dict] = []


class ReasoningAgentOutput(BaseModel):
    reasoning: str
    final: str
    confidence: int


class ReasoningAgent(Agent[ReasoningAgentInput, ReasoningAgentOutput]):
    def __init__(
        self,
        *,
        client: AsyncClient,
    ):
        self._ollama_client = client
        self._model = DEFAULT_LLM_MODEL

    async def run(self, input_: ReasoningAgentInput) -> ReasoningAgentOutput:
        reasoning_input = ReasoningInput(
            task=input_.task["description"],
            query_context=input_.query_context,
            previous_task_results=input_.previous_task_results,
            previous_subtask_results=input_.previous_subtask_results,
        )
        reasoning = Reasoning(client=self._ollama_client)
        reasoning_output: ReasoningOutput = await reasoning.call(
            reasoning_input, think="medium"
        )

        response = ReasoningAgentOutput(**reasoning_output.model_dump())

        logger.debug(f"{self.__class__.__name__} Response:\n{response}")
        return response
```

- [ ] **Step 3.23: Update all caller agents — `single_task_agent.py`**

Three changes in `single_task_agent.py`:
1. `_route_task()` parameter: `user_context` → `query_context`
2. `_execute_tool_subtask()` parameter: `user_context` → `query_context`
3. `_execute_reasoning_subtask()` parameter: `user_context` → `query_context`
4. All internal input model construction: `user_context=` → `query_context=`
5. Call sites in `_run()`: `user_context=ctx.query_context` → `query_context=ctx.query_context`
6. `_filter_task_result()` parameter: `user_context` → `query_context`

Full content of `easylocai/agents/single_task_agent.py`:

```python
import logging
from typing import Any

from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

from easylocai.agents.reasoning_agent import (
    ReasoningAgent,
    ReasoningAgentInput,
    ReasoningAgentOutput,
)
from easylocai.core.agent import Agent
from easylocai.core.tool_manager import ToolManager
from easylocai.llm_calls.subtask_result_filter import (
    SubtaskResultFilter,
    SubtaskResultFilterInput,
)
from easylocai.llm_calls.task_result_filter import (
    TaskResultFilter,
    TaskResultFilterInput,
)
from easylocai.llm_calls.task_router import TaskRouter, TaskRouterInput, TaskRouterOutput
from easylocai.llm_calls.tool_selector import (
    ToolInput,
    ToolSelector,
    ToolSelectorInput,
    ToolSelectorOutput,
)
from easylocai.schemas.context import SingleTaskAgentContext, SubtaskResult

logger = logging.getLogger(__name__)


class SingleTaskAgentOutput(BaseModel):
    executed_task: str
    result: str


class SingleTaskAgent(Agent[SingleTaskAgentContext, SingleTaskAgentOutput]):
    N_TOOL_RESULTS = 18

    def __init__(self, *, client: AsyncClient, tool_manager: ToolManager):
        self._ollama_client = client
        self._tool_manager = tool_manager

    async def _run(self, input_: SingleTaskAgentContext) -> SingleTaskAgentOutput:
        ctx = input_
        tool_candidates = await self._get_tool_candidates([ctx.original_task])

        previous_task_results = [
            {"task": r.executed_task, "result": r.result}
            for r in ctx.executed_task_results
        ]

        while True:
            iteration_results = [r.model_dump() for r in ctx.subtask_results]

            task_router_output = await self._route_task(
                task=ctx.original_task,
                query_context=ctx.query_context,
                tool_candidates=tool_candidates,
                previous_task_results=previous_task_results,
                iteration_results=iteration_results,
            )

            if task_router_output.finished:
                logger.debug(f"Task finished: {task_router_output.finished_reason}")
                break

            subtask = task_router_output.subtask
            subtask_type = task_router_output.subtask_type

            if subtask_type == "tool":
                result = await self._execute_tool_subtask(
                    subtask=subtask,
                    query_context=ctx.query_context,
                    previous_task_results=previous_task_results,
                    iteration_results=iteration_results,
                )
            elif subtask_type == "reasoning":
                result = await self._execute_reasoning_subtask(
                    subtask=subtask,
                    query_context=ctx.query_context,
                    previous_task_results=previous_task_results,
                    previous_subtask_results=iteration_results,
                )
            else:
                raise ValueError(f"Unknown subtask type: {subtask_type}")

            filtered_result = await self._filter_subtask_result(subtask=subtask, result=result)
            ctx.subtask_results.append(SubtaskResult(subtask=subtask, result=filtered_result))

        final_result = await self._filter_task_result(
            task=ctx.original_task,
            subtask_results=[r.model_dump() for r in ctx.subtask_results],
            query_context=ctx.query_context,
        )

        return SingleTaskAgentOutput(
            executed_task=ctx.original_task,
            result=final_result,
        )

    async def _get_tool_candidates(self, queries: list[str]) -> list[dict]:
        tools = await self._tool_manager.search_tools(queries, n_results=self.N_TOOL_RESULTS)
        return [
            {
                "server_name": t.server_name,
                "tool_name": t.name,
                "tool_description": t.description,
                "tool_input_schema": t.input_schema,
            }
            for t in tools
        ]

    async def _route_task(
        self,
        *,
        task: str,
        query_context: str | None,
        tool_candidates: list[dict],
        previous_task_results: list[dict],
        iteration_results: list[dict],
    ) -> TaskRouterOutput:
        task_router_input = TaskRouterInput(
            task=task,
            query_context=query_context,
            tool_candidates=tool_candidates,
            previous_task_results=previous_task_results,
            iteration_results=iteration_results,
        )
        task_router = TaskRouter(client=self._ollama_client)
        output: TaskRouterOutput = await task_router.call(task_router_input)
        logger.debug(f"TaskRouter output: {output}")
        return output

    async def _execute_tool_subtask(
        self,
        *,
        subtask: str,
        query_context: str | None,
        previous_task_results: list[dict],
        iteration_results: list[dict],
    ) -> dict[str, Any]:
        subtask_tool_candidates = await self._get_tool_candidates([subtask])
        tool_selector_input = ToolSelectorInput(
            subtask=subtask,
            query_context=query_context,
            tool_candidates=subtask_tool_candidates,
            previous_task_results=previous_task_results,
            iteration_results=iteration_results,
        )
        tool_selector = ToolSelector(client=self._ollama_client)
        try:
            tool_selector_output: ToolSelectorOutput = await tool_selector.call(tool_selector_input)
        except ValidationError:
            llm_call_response = tool_selector.current_llm_call_response
            logger.error(
                f"Failed to parse ToolSelector response: {llm_call_response['message']['content']}"
            )
            return {"error": "Failed to parse tool selector response"}

        if tool_selector_output.selected_tool is None:
            logger.warning(f"No tool selected for subtask: {subtask}")
            return {"error": tool_selector_output.failure_reason}

        return await self._call_tool(tool_selector_output.selected_tool)

    async def _execute_reasoning_subtask(
        self,
        *,
        subtask: str,
        query_context: str | None,
        previous_task_results: list[dict],
        previous_subtask_results: list[dict],
    ) -> dict[str, Any]:
        reasoning_agent_input = ReasoningAgentInput(
            task={"description": subtask},
            query_context=query_context,
            previous_task_results=previous_task_results,
            previous_subtask_results=previous_subtask_results,
        )
        reasoning_agent = ReasoningAgent(client=self._ollama_client)
        reasoning_agent_output: ReasoningAgentOutput = await reasoning_agent.run(reasoning_agent_input)
        logger.debug(f"ReasoningAgent output: {reasoning_agent_output}")
        return reasoning_agent_output.model_dump()

    async def _call_tool(self, tool_input: ToolInput) -> dict[str, Any]:
        tool_result = await self._tool_manager._server_manager.call_tool(
            tool_input.server_name,
            tool_input.tool_name,
            tool_input.tool_args,
        )
        logger.debug(f"Tool call result: {tool_result}")
        if tool_result.isError:
            return {"error": f"Error occurred when calling tool: {tool_result.content}"}
        if tool_result.structuredContent:
            return tool_result.structuredContent
        return {"content": tool_result.content}

    async def _filter_subtask_result(self, subtask: str, result: dict[str, Any]) -> str:
        subtask_result_filter_input = SubtaskResultFilterInput(subtask=subtask, result=result)
        subtask_result_filter = SubtaskResultFilter(client=self._ollama_client)
        output = await subtask_result_filter.call(subtask_result_filter_input)
        return output.root

    async def _filter_task_result(
        self,
        task: str,
        subtask_results: list[dict],
        query_context: str | None,
    ) -> str:
        task_result_filter_input = TaskResultFilterInput(
            task=task,
            subtask_results=subtask_results,
            query_context=query_context,
        )
        task_result_filter = TaskResultFilter(client=self._ollama_client)
        output = await task_result_filter.call(task_result_filter_input)
        return output.root
```

- [ ] **Step 3.24: Update tests that use `ReplannerInput` directly**

In `tests/integration_tests/agents/test_replan_agent.py`, change `user_context=None` → `query_context=None` in the `test_replanner_input_has_conversation_histories_field` test:

```python
    def test_replanner_input_has_conversation_histories_field(self):
        histories = [
            ConversationHistory(
                original_user_query="Find alice.txt",
                reformatted_user_query="Find alice.txt",
                response="Found alice.txt.",
            )
        ]
        inp = ReplannerInput(
            query_context=None,
            original_user_query="Summarize alice.txt",
            previous_plan=["Summarize alice.txt"],
            task_results=[{"task": "Summarize alice.txt", "result": "It is a short file."}],
            conversation_histories=histories,
        )
        assert len(inp.conversation_histories) == 1
```

Also update the test that uses `PlannerInput` directly:

```python
    def test_planner_input_has_conversation_histories_field(self):
        histories = [
            ConversationHistory(
                original_user_query="Find alice.txt",
                reformatted_user_query="Find alice.txt",
                response="Found alice.txt with 42 lines.",
            )
        ]
        inp = PlannerInput(
            user_query="Summarize alice.txt",
            query_context=None,
            conversation_histories=histories,
        )
        assert len(inp.conversation_histories) == 1
```

- [ ] **Step 3.25: Run full test suite**

```bash
pytest -v
```

Expected: All tests PASS. If any fail, they are due to a missed `user_context=` → `query_context=` rename — grep for the old name:

```bash
grep -r "user_context" easylocai/ resources/prompts/
```

Expected: No matches (all renamed).

- [ ] **Step 3.26: Commit**

```bash
git add resources/prompts/ easylocai/llm_calls/ easylocai/agents/ tests/
git commit -m "refactor: rename user_context -> query_context in all prompts and input models"
```

---

## Task 4: Rename Reasoning `task` → `subtask`

The `reasoning_user_prompt.jinja2` uses `{{ task }}` but actually receives a subtask string. This is inconsistent with `tool_selector_user_prompt.jinja2` which uses `{{ subtask }}`.

**Files:**
- Modify: `resources/prompts/reasoning_user_prompt.jinja2`
- Modify: `easylocai/llm_calls/reasoning.py`
- Modify: `easylocai/agents/reasoning_agent.py`

(Note: `reasoning_system_prompt.jinja2` does not contain a `{{ task }}` template variable — only prose text. No change needed there.)

- [ ] **Step 4.1: Rename `{{ task }}` → `{{ subtask }}` in `reasoning_user_prompt.jinja2`**

Full content of `resources/prompts/reasoning_user_prompt.jinja2`:

```
SUBTASK:
{{ subtask }}

QUERY_CONTEXT:
{{ query_context }}

PREVIOUS_TASK_RESULTS:
{% for task_result in previous_task_results %}
<task_result>
- task: {{ task_result["task"] }}
- result: {{ task_result["result"] }}
</task_result>
{% endfor %}

PREVIOUS_SUBTASK_RESULTS:
{% for subtask_result in previous_subtask_results %}
<subtask_result>
    - subtask:
    {{ subtask_result["subtask"] }}
    - result:
    {{ subtask_result["result"] }}
</subtask_result>
{% endfor %}
```

- [ ] **Step 4.2: Rename `task` → `subtask` field in `ReasoningInput`**

Full content of `easylocai/llm_calls/reasoning.py`:

```python
from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2


class ReasoningInput(BaseModel):
    subtask: str = Field(
        title="Subtask",
        description="The subtask to reason about.",
    )
    query_context: str | None = Field(
        title="Query Context",
        description="Preamble context extracted from the current query by QueryReformatter.",
    )
    previous_task_results: list[dict] = Field(
        title="Previous Task Results",
        description="Results from previous tasks.",
    )
    previous_subtask_results: list[dict] = Field(
        default_factory=list,
        title="Previous Subtask Results",
        description="Results from previous subtasks within the current task.",
    )


class ReasoningOutput(BaseModel):
    reasoning: str = Field(
        title="Reasoning",
        description="The reasoning process.",
    )
    final: str = Field(
        title="Final",
        description="The final answer.",
    )
    confidence: int = Field(
        title="Confidence",
        description="Confidence level of the answer.",
    )


class Reasoning(LLMCallV2[ReasoningInput, ReasoningOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/reasoning_system_prompt.jinja2"
        user_prompt_path = "prompts/reasoning_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ReasoningOutput,
            options=options,
        )
```

- [ ] **Step 4.3: Update `reasoning_agent.py` to pass `subtask=`**

Full content of `easylocai/agents/reasoning_agent.py`:

```python
import logging

from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.core.contants import DEFAULT_LLM_MODEL
from easylocai.llm_calls.reasoning import Reasoning, ReasoningInput, ReasoningOutput

logger = logging.getLogger(__name__)


class ReasoningAgentInput(BaseModel):
    task: dict
    query_context: str | None
    previous_task_results: list[dict]
    previous_subtask_results: list[dict] = []


class ReasoningAgentOutput(BaseModel):
    reasoning: str
    final: str
    confidence: int


class ReasoningAgent(Agent[ReasoningAgentInput, ReasoningAgentOutput]):
    def __init__(
        self,
        *,
        client: AsyncClient,
    ):
        self._ollama_client = client
        self._model = DEFAULT_LLM_MODEL

    async def run(self, input_: ReasoningAgentInput) -> ReasoningAgentOutput:
        reasoning_input = ReasoningInput(
            subtask=input_.task["description"],
            query_context=input_.query_context,
            previous_task_results=input_.previous_task_results,
            previous_subtask_results=input_.previous_subtask_results,
        )
        reasoning = Reasoning(client=self._ollama_client)
        reasoning_output: ReasoningOutput = await reasoning.call(
            reasoning_input, think="medium"
        )

        response = ReasoningAgentOutput(**reasoning_output.model_dump())

        logger.debug(f"{self.__class__.__name__} Response:\n{response}")
        return response
```

- [ ] **Step 4.4: Run full test suite**

```bash
pytest -v
```

Expected: All tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add resources/prompts/reasoning_user_prompt.jinja2 \
        easylocai/llm_calls/reasoning.py \
        easylocai/agents/reasoning_agent.py
git commit -m "refactor: rename task -> subtask in ReasoningInput and reasoning_user_prompt"
```

---

## Task 5: Add `original_task` to ToolSelector and Reasoning

Both ToolSelector and Reasoning only receive a subtask. When the subtask description is ambiguous, they lack the parent task context needed to make correct decisions.

**Files:**
- Modify: `resources/prompts/tool_selector_user_prompt.jinja2`
- Modify: `resources/prompts/tool_selector_system_prompt.jinja2`
- Modify: `easylocai/llm_calls/tool_selector.py`
- Modify: `resources/prompts/reasoning_user_prompt.jinja2`
- Modify: `resources/prompts/reasoning_system_prompt.jinja2`
- Modify: `easylocai/llm_calls/reasoning.py`
- Modify: `easylocai/agents/reasoning_agent.py`
- Modify: `easylocai/agents/single_task_agent.py`

- [ ] **Step 5.1: Add `ORIGINAL_TASK` to `tool_selector_user_prompt.jinja2`**

Full content of `resources/prompts/tool_selector_user_prompt.jinja2`:

```
ORIGINAL_TASK:
{{ original_task }}

SUBTASK:
{{ subtask }}

QUERY_CONTEXT:
{{ query_context }}

TOOL_CANDIDATES:
{% for tool in tool_candidates %}
<tool>
- server_name: {{ tool["server_name"] }}
- tool_name: {{ tool["tool_name"] }}
- tool_description: {{ tool["tool_description"] }}
- input_schema: {{ tool["tool_input_schema"] }}
</tool>
{% endfor %}

PREVIOUS_TASK_RESULTS (chronological order):
{% for task_result in previous_task_results %}
<task_result>
- task: {{ task_result["task"] }}
- result: {{ task_result["result"] }}
</task_result>
{% endfor %}

PREVIOUS_SUBTASK_RESULTS (chronological order):
{% for iteration_result in iteration_results %}
<subtask_result>
    - subtask:
    {{ iteration_result["subtask"] }}
    - result:
    {{ iteration_result["result"] }}
</subtask_result>
{% endfor %}
```

- [ ] **Step 5.2: Mention `ORIGINAL_TASK` in `tool_selector_system_prompt.jinja2`**

Full content of `resources/prompts/tool_selector_system_prompt.jinja2`:

```
You are a tool selector assistant.

Your job is to select the appropriate tool and configure its arguments to execute the given subtask.

## Input Context
- ORIGINAL_TASK: The parent task that this subtask belongs to — use it to understand the broader goal
- SUBTASK: The specific subtask to complete right now
- QUERY_CONTEXT: Preamble context extracted from the current query (may be null)
- TOOL_CANDIDATES: Available tools that can be used
- PREVIOUS_TASK_RESULTS: Results from previously completed tasks
- PREVIOUS_SUBTASK_RESULTS: Results from subtasks in the current task iteration

## Your Responsibilities
1. **Analyze**: the subtask requirements and available tool capabilities
2. **Select**: one tool from TOOL_CANDIDATES that best accomplish the subtask
3. **Configure**: the arguments for the selected tool based on the context

## Selection Guidelines

When selecting a tool:
- Choose the tool whose description best matches the subtask requirements
- Refer to the tool's `input_schema` for required and optional parameters
- Use ORIGINAL_TASK to resolve ambiguity in the subtask description

When configuring arguments:
- Extract relevant values from the subtask description and context
- Provide all required parameters as defined in the tool's input_schema
- Include optional parameters only when explicitly needed

When no matching tool is found:
- Set `selected_tool` to null
- Provide a clear `failure_reason` explaining why no tool matches the subtask

## Output Format
Respond in JSON format with the given structure. Empty output is not allowed.
```

- [ ] **Step 5.3: Add `original_task` field to `ToolSelectorInput`**

In `easylocai/llm_calls/tool_selector.py`, add `original_task: str` to `ToolSelectorInput` (and `ToolSelectorV2Input` for consistency):

```python
from typing import Any

from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2


class ToolInput(BaseModel):
    server_name: str = Field(description="The server_name of the selected tool")
    tool_name: str = Field(
        description="The tool_name selected based on subtask requirements"
    )
    tool_args: dict[str, Any] = Field(
        description="Parameters for tool execution following the tool's input_schema. Do not include extra parameters not defined in the tool's input_schema."
    )


class ToolSelectorV2Input(BaseModel):
    original_task: str
    subtask: str
    query_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class ToolSelectorOutput(BaseModel):
    selected_tool: ToolInput | None = Field(
        description="The tool selected to execute the given subtask. None if no matching tool found."
    )
    failure_reason: str | None = Field(
        description="Reason why no tool was selected. None if a tool was successfully selected."
    )


class ToolSelectorInput(BaseModel):
    original_task: str
    subtask: str
    query_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class ToolSelectorInputV2(BaseModel):
    original_task: str
    subtasks: list[str]
    query_context: str | None
    tool_candidates: list[dict]
    previous_task_results: list[dict]
    iteration_results: list[dict]


class SubtaskWithTool(BaseModel):
    subtask: str = Field(description="The actual subtask for which tool is selected.")
    selected_tool: ToolInput | None = Field(
        description="The tool selected to execute the given subtask. None if no matching tool found."
    )
    failure_reason: str | None = Field(
        description="Reason why tool can not be selected. None if a tool was successfully selected."
    )


class ToolSelectorOutputV2(BaseModel):
    results: list[SubtaskWithTool]


class ToolSelector(LLMCallV2[ToolSelectorInput, ToolSelectorOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/tool_selector_system_prompt.jinja2"
        user_prompt_path = "prompts/tool_selector_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ToolSelectorOutput,
            options=options,
        )
```

- [ ] **Step 5.4: Add `ORIGINAL_TASK` to `reasoning_user_prompt.jinja2`**

Full content of `resources/prompts/reasoning_user_prompt.jinja2`:

```
ORIGINAL_TASK:
{{ original_task }}

SUBTASK:
{{ subtask }}

QUERY_CONTEXT:
{{ query_context }}

PREVIOUS_TASK_RESULTS:
{% for task_result in previous_task_results %}
<task_result>
- task: {{ task_result["task"] }}
- result: {{ task_result["result"] }}
</task_result>
{% endfor %}

PREVIOUS_SUBTASK_RESULTS:
{% for subtask_result in previous_subtask_results %}
<subtask_result>
    - subtask:
    {{ subtask_result["subtask"] }}
    - result:
    {{ subtask_result["result"] }}
</subtask_result>
{% endfor %}
```

- [ ] **Step 5.5: Mention `ORIGINAL_TASK` in `reasoning_system_prompt.jinja2`**

Full content of `resources/prompts/reasoning_system_prompt.jinja2`:

```
You are a REASONING AGENT that solves tasks using deliberate, step-by-step thinking. You excel at arithmetic, algebra, geometry, and word problems. You never skip steps and always verify your arithmetic before answering.

## Goals
- Understand the subtask deeply in the context of the ORIGINAL_TASK
- Gather relevant information from QUERY_CONTEXT, PREVIOUS_TASK_RESULTS, and PREVIOUS_SUBTASK_RESULTS if needed
- Analyse options and trade-offs logically
- Produce the best final answer with high reliability

## Reasoning Rules
1. Think step-by-step with no big jumps. For math problems, show every arithmetic operation explicitly.
2. For math: compute intermediate results and double-check them before proceeding to the next step.
3. Consider multiple possibilities; eliminate wrong ones with justification.
4. If information is missing, state the assumptions explicitly.
5. If uncertain, say "Not enough information" instead of inventing facts. `confidence` should be 0 in such cases.
6. Keep reasoning inside `reasoning` field. Never include reasoning in `final`.
7. The `final` field must be short and directly answer the user. For math problems, state just the numeric result (with units if applicable).

## Output Format: STRICTLY JSON ONLY
{
  "reasoning": "<string> Step-by-step thinking, hypotheses, alternatives, and decision path",
  "final": "<string> Final answer for the user (no reasoning). Answer must be concise and actionable.",
  "confidence": <integer 0–100>
}

### Examples

Task: "Calculate (2 + 3) * 5"
Response:
{
  "reasoning": "Step 1: Add 2 + 3 = 5. Step 2: Multiply 5 * 5 = 25.",
  "final": "25",
  "confidence": 100
}

Task: "What is 15% of 200?"
Response:
{
  "reasoning": "Step 1: Convert 15% to decimal: 15/100 = 0.15. Step 2: Multiply 0.15 * 200 = 30.",
  "final": "30",
  "confidence": 100
}

Task: "A train travels 300 km in 4 hours. What is its average speed?"
Response:
{
  "reasoning": "Step 1: Average speed = distance / time = 300 / 4 = 75. Step 2: Units are km/h.",
  "final": "75 km/h",
  "confidence": 100
}
```

- [ ] **Step 5.6: Add `original_task` field to `ReasoningInput`**

Full content of `easylocai/llm_calls/reasoning.py`:

```python
from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2


class ReasoningInput(BaseModel):
    original_task: str = Field(
        title="Original Task",
        description="The parent task that this subtask belongs to.",
    )
    subtask: str = Field(
        title="Subtask",
        description="The subtask to reason about.",
    )
    query_context: str | None = Field(
        title="Query Context",
        description="Preamble context extracted from the current query by QueryReformatter.",
    )
    previous_task_results: list[dict] = Field(
        title="Previous Task Results",
        description="Results from previous tasks.",
    )
    previous_subtask_results: list[dict] = Field(
        default_factory=list,
        title="Previous Subtask Results",
        description="Results from previous subtasks within the current task.",
    )


class ReasoningOutput(BaseModel):
    reasoning: str = Field(
        title="Reasoning",
        description="The reasoning process.",
    )
    final: str = Field(
        title="Final",
        description="The final answer.",
    )
    confidence: int = Field(
        title="Confidence",
        description="Confidence level of the answer.",
    )


class Reasoning(LLMCallV2[ReasoningInput, ReasoningOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/reasoning_system_prompt.jinja2"
        user_prompt_path = "prompts/reasoning_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ReasoningOutput,
            options=options,
        )
```

- [ ] **Step 5.7: Add `original_task` to `ReasoningAgentInput` and update `reasoning_agent.py`**

Full content of `easylocai/agents/reasoning_agent.py`:

```python
import logging

from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.core.contants import DEFAULT_LLM_MODEL
from easylocai.llm_calls.reasoning import Reasoning, ReasoningInput, ReasoningOutput

logger = logging.getLogger(__name__)


class ReasoningAgentInput(BaseModel):
    original_task: str
    task: dict
    query_context: str | None
    previous_task_results: list[dict]
    previous_subtask_results: list[dict] = []


class ReasoningAgentOutput(BaseModel):
    reasoning: str
    final: str
    confidence: int


class ReasoningAgent(Agent[ReasoningAgentInput, ReasoningAgentOutput]):
    def __init__(
        self,
        *,
        client: AsyncClient,
    ):
        self._ollama_client = client
        self._model = DEFAULT_LLM_MODEL

    async def run(self, input_: ReasoningAgentInput) -> ReasoningAgentOutput:
        reasoning_input = ReasoningInput(
            original_task=input_.original_task,
            subtask=input_.task["description"],
            query_context=input_.query_context,
            previous_task_results=input_.previous_task_results,
            previous_subtask_results=input_.previous_subtask_results,
        )
        reasoning = Reasoning(client=self._ollama_client)
        reasoning_output: ReasoningOutput = await reasoning.call(
            reasoning_input, think="medium"
        )

        response = ReasoningAgentOutput(**reasoning_output.model_dump())

        logger.debug(f"{self.__class__.__name__} Response:\n{response}")
        return response
```

- [ ] **Step 5.8: Update `single_task_agent.py` to pass `original_task` to both ToolSelector and ReasoningAgent**

Two methods change: `_execute_tool_subtask()` adds `original_task` parameter, and `_execute_reasoning_subtask()` adds `original_task` parameter. Their call sites in `_run()` also update.

Full content of `easylocai/agents/single_task_agent.py`:

```python
import logging
from typing import Any

from ollama import AsyncClient
from pydantic import BaseModel, ValidationError

from easylocai.agents.reasoning_agent import (
    ReasoningAgent,
    ReasoningAgentInput,
    ReasoningAgentOutput,
)
from easylocai.core.agent import Agent
from easylocai.core.tool_manager import ToolManager
from easylocai.llm_calls.subtask_result_filter import (
    SubtaskResultFilter,
    SubtaskResultFilterInput,
)
from easylocai.llm_calls.task_result_filter import (
    TaskResultFilter,
    TaskResultFilterInput,
)
from easylocai.llm_calls.task_router import TaskRouter, TaskRouterInput, TaskRouterOutput
from easylocai.llm_calls.tool_selector import (
    ToolInput,
    ToolSelector,
    ToolSelectorInput,
    ToolSelectorOutput,
)
from easylocai.schemas.context import SingleTaskAgentContext, SubtaskResult

logger = logging.getLogger(__name__)


class SingleTaskAgentOutput(BaseModel):
    executed_task: str
    result: str


class SingleTaskAgent(Agent[SingleTaskAgentContext, SingleTaskAgentOutput]):
    N_TOOL_RESULTS = 18

    def __init__(self, *, client: AsyncClient, tool_manager: ToolManager):
        self._ollama_client = client
        self._tool_manager = tool_manager

    async def _run(self, input_: SingleTaskAgentContext) -> SingleTaskAgentOutput:
        ctx = input_
        tool_candidates = await self._get_tool_candidates([ctx.original_task])

        previous_task_results = [
            {"task": r.executed_task, "result": r.result}
            for r in ctx.executed_task_results
        ]

        while True:
            iteration_results = [r.model_dump() for r in ctx.subtask_results]

            task_router_output = await self._route_task(
                task=ctx.original_task,
                query_context=ctx.query_context,
                tool_candidates=tool_candidates,
                previous_task_results=previous_task_results,
                iteration_results=iteration_results,
            )

            if task_router_output.finished:
                logger.debug(f"Task finished: {task_router_output.finished_reason}")
                break

            subtask = task_router_output.subtask
            subtask_type = task_router_output.subtask_type

            if subtask_type == "tool":
                result = await self._execute_tool_subtask(
                    original_task=ctx.original_task,
                    subtask=subtask,
                    query_context=ctx.query_context,
                    previous_task_results=previous_task_results,
                    iteration_results=iteration_results,
                )
            elif subtask_type == "reasoning":
                result = await self._execute_reasoning_subtask(
                    original_task=ctx.original_task,
                    subtask=subtask,
                    query_context=ctx.query_context,
                    previous_task_results=previous_task_results,
                    previous_subtask_results=iteration_results,
                )
            else:
                raise ValueError(f"Unknown subtask type: {subtask_type}")

            filtered_result = await self._filter_subtask_result(subtask=subtask, result=result)
            ctx.subtask_results.append(SubtaskResult(subtask=subtask, result=filtered_result))

        final_result = await self._filter_task_result(
            task=ctx.original_task,
            subtask_results=[r.model_dump() for r in ctx.subtask_results],
            query_context=ctx.query_context,
        )

        return SingleTaskAgentOutput(
            executed_task=ctx.original_task,
            result=final_result,
        )

    async def _get_tool_candidates(self, queries: list[str]) -> list[dict]:
        tools = await self._tool_manager.search_tools(queries, n_results=self.N_TOOL_RESULTS)
        return [
            {
                "server_name": t.server_name,
                "tool_name": t.name,
                "tool_description": t.description,
                "tool_input_schema": t.input_schema,
            }
            for t in tools
        ]

    async def _route_task(
        self,
        *,
        task: str,
        query_context: str | None,
        tool_candidates: list[dict],
        previous_task_results: list[dict],
        iteration_results: list[dict],
    ) -> TaskRouterOutput:
        task_router_input = TaskRouterInput(
            task=task,
            query_context=query_context,
            tool_candidates=tool_candidates,
            previous_task_results=previous_task_results,
            iteration_results=iteration_results,
        )
        task_router = TaskRouter(client=self._ollama_client)
        output: TaskRouterOutput = await task_router.call(task_router_input)
        logger.debug(f"TaskRouter output: {output}")
        return output

    async def _execute_tool_subtask(
        self,
        *,
        original_task: str,
        subtask: str,
        query_context: str | None,
        previous_task_results: list[dict],
        iteration_results: list[dict],
    ) -> dict[str, Any]:
        subtask_tool_candidates = await self._get_tool_candidates([subtask])
        tool_selector_input = ToolSelectorInput(
            original_task=original_task,
            subtask=subtask,
            query_context=query_context,
            tool_candidates=subtask_tool_candidates,
            previous_task_results=previous_task_results,
            iteration_results=iteration_results,
        )
        tool_selector = ToolSelector(client=self._ollama_client)
        try:
            tool_selector_output: ToolSelectorOutput = await tool_selector.call(tool_selector_input)
        except ValidationError:
            llm_call_response = tool_selector.current_llm_call_response
            logger.error(
                f"Failed to parse ToolSelector response: {llm_call_response['message']['content']}"
            )
            return {"error": "Failed to parse tool selector response"}

        if tool_selector_output.selected_tool is None:
            logger.warning(f"No tool selected for subtask: {subtask}")
            return {"error": tool_selector_output.failure_reason}

        return await self._call_tool(tool_selector_output.selected_tool)

    async def _execute_reasoning_subtask(
        self,
        *,
        original_task: str,
        subtask: str,
        query_context: str | None,
        previous_task_results: list[dict],
        previous_subtask_results: list[dict],
    ) -> dict[str, Any]:
        reasoning_agent_input = ReasoningAgentInput(
            original_task=original_task,
            task={"description": subtask},
            query_context=query_context,
            previous_task_results=previous_task_results,
            previous_subtask_results=previous_subtask_results,
        )
        reasoning_agent = ReasoningAgent(client=self._ollama_client)
        reasoning_agent_output: ReasoningAgentOutput = await reasoning_agent.run(reasoning_agent_input)
        logger.debug(f"ReasoningAgent output: {reasoning_agent_output}")
        return reasoning_agent_output.model_dump()

    async def _call_tool(self, tool_input: ToolInput) -> dict[str, Any]:
        tool_result = await self._tool_manager._server_manager.call_tool(
            tool_input.server_name,
            tool_input.tool_name,
            tool_input.tool_args,
        )
        logger.debug(f"Tool call result: {tool_result}")
        if tool_result.isError:
            return {"error": f"Error occurred when calling tool: {tool_result.content}"}
        if tool_result.structuredContent:
            return tool_result.structuredContent
        return {"content": tool_result.content}

    async def _filter_subtask_result(self, subtask: str, result: dict[str, Any]) -> str:
        subtask_result_filter_input = SubtaskResultFilterInput(subtask=subtask, result=result)
        subtask_result_filter = SubtaskResultFilter(client=self._ollama_client)
        output = await subtask_result_filter.call(subtask_result_filter_input)
        return output.root

    async def _filter_task_result(
        self,
        task: str,
        subtask_results: list[dict],
        query_context: str | None,
    ) -> str:
        task_result_filter_input = TaskResultFilterInput(
            task=task,
            subtask_results=subtask_results,
            query_context=query_context,
        )
        task_result_filter = TaskResultFilter(client=self._ollama_client)
        output = await task_result_filter.call(task_result_filter_input)
        return output.root
```

- [ ] **Step 5.9: Run full test suite**

```bash
pytest -v
```

Expected: All tests PASS.

- [ ] **Step 5.10: Commit**

```bash
git add resources/prompts/tool_selector_user_prompt.jinja2 \
        resources/prompts/tool_selector_system_prompt.jinja2 \
        resources/prompts/reasoning_user_prompt.jinja2 \
        resources/prompts/reasoning_system_prompt.jinja2 \
        easylocai/llm_calls/tool_selector.py \
        easylocai/llm_calls/reasoning.py \
        easylocai/agents/reasoning_agent.py \
        easylocai/agents/single_task_agent.py
git commit -m "feat: add original_task context to ToolSelector and Reasoning"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 5 items from the spec's implementation priority table are covered: Planner conversation_histories (Task 1), Replanner conversation_histories (Task 2), user_context → query_context rename (Task 3), Reasoning task → subtask rename (Task 4), original_task for ToolSelector/Reasoning (Task 5).
- [x] **No placeholders:** All code steps contain complete, copy-pasteable content.
- [x] **Type consistency:** `conversation_histories: list[ConversationHistory]` uses the same `ConversationHistory` type from `easylocai/schemas/context.py` in all locations. `query_context: str | None` matches the existing field type in `WorkflowContext`. `original_task: str` matches `SingleTaskAgentContext.original_task`.
- [x] **Test updates:** Unit-style field tests (Steps 1.1, 2.1) fail before implementation and pass after. Integration tests run after each task to catch regressions. Task 3's test for `PlannerInput` and `ReplannerInput` updated to use renamed fields (Step 3.24).
- [x] **Dependency order:** Task 3 comes after Tasks 1 and 2 (which introduce `conversation_histories`) so the rename in Steps 3.18–3.19 captures the full final state of replanner prompts. Task 4 comes before Task 5 so `subtask` field is in place when Task 5 adds `original_task` alongside it.
