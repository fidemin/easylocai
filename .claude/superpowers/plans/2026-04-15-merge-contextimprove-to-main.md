# Merge contextimprove Flag Into Main Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old main workflow and agents with the contextimprove variants, renaming all classes and files to drop the `ContextImprove` suffix.

**Architecture:** `git rm` old files, `git mv` contextimprove files to canonical names, then do in-place renames of classes and imports. Tests follow the same pattern. Main loop is updated last to use `GlobalContext`.

**Tech Stack:** Python, pytest-asyncio, git

---

### Task 1: Replace agent files

**Files:**
- Delete: `easylocai/agents/plan_agent.py`, `easylocai/agents/replan_agent.py`, `easylocai/agents/single_task_agent.py`
- Rename: `easylocai/agents/plan_agent_contextimprove.py` → `easylocai/agents/plan_agent.py`
- Rename: `easylocai/agents/replan_agent_contextimprove.py` → `easylocai/agents/replan_agent.py`
- Rename: `easylocai/agents/single_task_agent_contextimprove.py` → `easylocai/agents/single_task_agent.py`

- [ ] **Step 1: git rm old agent files**

```bash
git -C /Users/yunhongmin/Programming/easylocai rm easylocai/agents/plan_agent.py easylocai/agents/replan_agent.py easylocai/agents/single_task_agent.py
```

- [ ] **Step 2: git mv contextimprove agents to canonical names**

```bash
git -C /Users/yunhongmin/Programming/easylocai mv easylocai/agents/plan_agent_contextimprove.py easylocai/agents/plan_agent.py
git -C /Users/yunhongmin/Programming/easylocai mv easylocai/agents/replan_agent_contextimprove.py easylocai/agents/replan_agent.py
git -C /Users/yunhongmin/Programming/easylocai mv easylocai/agents/single_task_agent_contextimprove.py easylocai/agents/single_task_agent.py
```

- [ ] **Step 3: Rename classes in plan_agent.py**

Replace the full content of `easylocai/agents/plan_agent.py`:

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
        )
        planner = Planner(client=self._ollama_client)
        planner_output: PlannerOutput = await planner.call(planner_input)

        return PlanAgentOutput(
            query_context=reformatter_output.query_context,
            reformatted_user_query=reformatter_output.reformed_query,
            task_list=planner_output.tasks,
        )
```

- [ ] **Step 4: Rename classes in replan_agent.py**

Replace the full content of `easylocai/agents/replan_agent.py`:

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
        )

        replanner = Replanner(client=self._ollama_client)
        replanner_output: ReplannerOutput = await replanner.call(replanner_input)

        logger.debug(f"ReplanAgent output: {replanner_output}")

        return ReplanAgentOutput(
            tasks=replanner_output.tasks,
            response=replanner_output.response,
        )
```

- [ ] **Step 5: Rename classes in single_task_agent.py**

Replace the full content of `easylocai/agents/single_task_agent.py`:

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

    async def run(self, input_: SingleTaskAgentContext) -> SingleTaskAgentOutput:
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
                user_context=ctx.query_context,
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
                    user_context=ctx.query_context,
                    previous_task_results=previous_task_results,
                    iteration_results=iteration_results,
                )
            elif subtask_type == "reasoning":
                result = await self._execute_reasoning_subtask(
                    subtask=subtask,
                    user_context=ctx.query_context,
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
            user_context=ctx.query_context,
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
        user_context: str | None,
        tool_candidates: list[dict],
        previous_task_results: list[dict],
        iteration_results: list[dict],
    ) -> TaskRouterOutput:
        task_router_input = TaskRouterInput(
            task=task,
            user_context=user_context,
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
        user_context: str | None,
        previous_task_results: list[dict],
        iteration_results: list[dict],
    ) -> dict[str, Any]:
        subtask_tool_candidates = await self._get_tool_candidates([subtask])
        tool_selector_input = ToolSelectorInput(
            subtask=subtask,
            user_context=user_context,
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
        user_context: str | None,
        previous_task_results: list[dict],
        previous_subtask_results: list[dict],
    ) -> dict[str, Any]:
        reasoning_agent_input = ReasoningAgentInput(
            task={"description": subtask},
            user_context=user_context,
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
        user_context: str | None,
    ) -> str:
        task_result_filter_input = TaskResultFilterInput(
            task=task,
            subtask_results=subtask_results,
            user_context=user_context,
        )
        task_result_filter = TaskResultFilter(client=self._ollama_client)
        output = await task_result_filter.call(task_result_filter_input)
        return output.root
```

- [ ] **Step 6: Commit agent renames**

```bash
git -C /Users/yunhongmin/Programming/easylocai add easylocai/agents/plan_agent.py easylocai/agents/replan_agent.py easylocai/agents/single_task_agent.py
git -C /Users/yunhongmin/Programming/easylocai commit -m "refactor: rename contextimprove agents to canonical names"
```

---

### Task 2: Replace workflow file

**Files:**
- Delete: `easylocai/workflow.py`
- Rename: `easylocai/workflow_contextimprove.py` → `easylocai/workflow.py`

- [ ] **Step 1: git rm old workflow**

```bash
git -C /Users/yunhongmin/Programming/easylocai rm easylocai/workflow.py
```

- [ ] **Step 2: git mv contextimprove workflow**

```bash
git -C /Users/yunhongmin/Programming/easylocai mv easylocai/workflow_contextimprove.py easylocai/workflow.py
```

- [ ] **Step 3: Update workflow.py — rename class and imports**

Replace the full content of `easylocai/workflow.py`:

```python
import logging
from contextlib import AsyncExitStack
from typing import AsyncGenerator

from ollama import AsyncClient

from easylocai.agents.plan_agent import (
    PlanAgent,
    PlanAgentInput,
    PlanAgentOutput,
)
from easylocai.agents.replan_agent import (
    ReplanAgent,
    ReplanAgentInput,
    ReplanAgentOutput,
)
from easylocai.agents.single_task_agent import (
    SingleTaskAgent,
    SingleTaskAgentOutput,
)
from easylocai.core.tool_manager import ToolManager
from easylocai.schemas.common import EasyLocaiWorkflowOutput
from easylocai.schemas.context import (
    ConversationHistory,
    ExecutedTaskResult,
    GlobalContext,
    SingleTaskAgentContext,
    WorkflowContext,
)
from easylocai.search_engines.advanced_search_engine import AdvancedSearchEngine

logger = logging.getLogger(__name__)


def ensure_initialized(func):
    async def wrapper(self, *args, **kwargs):
        if not self._initialized:
            raise RuntimeError(
                "EasylocaiWorkflow is not initialized. "
                "Please call 'initialize' before running."
            )
        async for item in func(self, *args, **kwargs):
            yield item

    return wrapper


class EasylocaiWorkflow:
    def __init__(
        self,
        *,
        config_dict: dict,
        search_engine: AdvancedSearchEngine,
        ollama_client: AsyncClient,
    ):
        self._tool_manager = ToolManager(
            search_engine, mpc_servers=config_dict["mcpServers"]
        )
        self._plan_agent = PlanAgent(client=ollama_client)
        self._replan_agent = ReplanAgent(client=ollama_client)
        self._single_task_agent = SingleTaskAgent(
            client=ollama_client,
            tool_manager=self._tool_manager,
        )
        self._initialized = False

    def initialize(self, stack: AsyncExitStack):
        self._initialized = True
        return self._tool_manager.initialize(stack)

    @ensure_initialized
    async def run(
        self,
        user_query: str,
        *,
        global_context: GlobalContext,
    ) -> AsyncGenerator[EasyLocaiWorkflowOutput, None]:
        workflow_context = WorkflowContext(
            conversation_histories=global_context.conversation_histories,
            original_user_query=user_query,
        )

        yield EasyLocaiWorkflowOutput(type="status", message="Thinking...")

        plan_output: PlanAgentOutput = await self._plan_agent.run(
            PlanAgentInput(workflow_context=workflow_context)
        )

        workflow_context.query_context = plan_output.query_context
        workflow_context.reformatted_user_query = plan_output.reformatted_user_query
        workflow_context.task_list = plan_output.task_list

        logger.debug(f"Plan output: {plan_output}")

        answer = None
        while True:
            next_task = workflow_context.task_list[0]
            yield EasyLocaiWorkflowOutput(type="status", message=next_task)

            single_task_context = SingleTaskAgentContext(
                conversation_histories=workflow_context.conversation_histories,
                original_user_query=workflow_context.original_user_query,
                query_context=workflow_context.query_context,
                reformatted_user_query=workflow_context.reformatted_user_query,
                task_list=workflow_context.task_list,
                executed_task_results=workflow_context.executed_task_results,
                original_task=next_task,
            )

            task_output: SingleTaskAgentOutput = await self._single_task_agent.run(
                single_task_context
            )

            workflow_context.executed_task_results.append(
                ExecutedTaskResult(
                    executed_task=task_output.executed_task,
                    result=task_output.result,
                )
            )

            yield EasyLocaiWorkflowOutput(type="status", message="Check for completion...")

            replan_output: ReplanAgentOutput = await self._replan_agent.run(
                ReplanAgentInput(workflow_context=workflow_context)
            )
            logger.debug(f"Replan output: {replan_output}")

            if replan_output.response is not None:
                answer = replan_output.response
                break

            workflow_context.task_list = replan_output.tasks

        global_context.conversation_histories.append(
            ConversationHistory(
                original_user_query=user_query,
                reformatted_user_query=workflow_context.reformatted_user_query or user_query,
                query_context=workflow_context.query_context,
                response=answer,
            )
        )

        yield EasyLocaiWorkflowOutput(type="result", message=answer)
```

- [ ] **Step 4: Commit workflow rename**

```bash
git -C /Users/yunhongmin/Programming/easylocai add easylocai/workflow.py
git -C /Users/yunhongmin/Programming/easylocai commit -m "refactor: rename EasylocaiWorkflowContextImprove to EasylocaiWorkflow"
```

---

### Task 3: Update main.py and delete main_contextimprove.py

**Files:**
- Modify: `easylocai/main.py`
- Delete: `easylocai/main_contextimprove.py`

- [ ] **Step 1: Replace main.py content**

```python
import json
import logging
from contextlib import AsyncExitStack

from ollama import AsyncClient
from rich import get_console

from easylocai.config import user_config_path
from easylocai.schemas.context import GlobalContext
from easylocai.search_engines.advanced_search_engine import AdvancedSearchEngine
from easylocai.utlis.console_util import ConsoleSpinner, multiline_input, render_chat
from easylocai.workflow import EasylocaiWorkflow

logger = logging.getLogger(__name__)


async def run_agent_workflow_main():
    console = get_console()

    ollama_client = AsyncClient(host="http://localhost:11434")
    search_engine = AdvancedSearchEngine()

    config_path = user_config_path()
    with open(config_path) as f:
        config_dict = json.load(f)

    workflow = EasylocaiWorkflow(
        config_dict=config_dict,
        search_engine=search_engine,
        ollama_client=ollama_client,
    )

    global_context = GlobalContext()
    messages = []

    stack = AsyncExitStack()
    async with stack:
        await workflow.initialize(stack)

        while True:
            render_chat(console, messages)
            try:
                user_input = await multiline_input("> ")
            except KeyboardInterrupt:
                logger.warning("User interrupted the input")
                print("\nExiting...")
                break
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            messages.append({"role": "user", "content": user_input})
            render_chat(console, messages)

            async_generator = workflow.run(user_input, global_context=global_context)

            answer = None
            with ConsoleSpinner(console) as spinner:
                async for output in async_generator:
                    if output.type == "status":
                        spinner.set_prefix(output.message)
                        continue
                    if output.type == "result":
                        answer = output.message
                        messages.append({"role": "assistant", "content": answer})


workflow_registry = {
    "main": run_agent_workflow_main,
}


async def run_agent_workflow(flag: str | None = None):
    if flag is None:
        flag = "main"

    workflow_function = workflow_registry.get(flag)
    if workflow_function is None:
        raise ValueError(f"Unknown workflow flag: {flag}")

    await workflow_function()
```

- [ ] **Step 2: git rm main_contextimprove.py**

```bash
git -C /Users/yunhongmin/Programming/easylocai rm easylocai/main_contextimprove.py
```

- [ ] **Step 3: Commit main.py update**

```bash
git -C /Users/yunhongmin/Programming/easylocai add easylocai/main.py
git -C /Users/yunhongmin/Programming/easylocai commit -m "refactor: update main.py to use new EasylocaiWorkflow and GlobalContext"
```

---

### Task 4: Rename test files

**Files:**
- Delete: `tests/integration_tests/agents/test_plan_agent.py`, `test_replan_agent.py`, `test_single_task_agent.py`
- Rename: `test_plan_agent_contextimprove.py` → `test_plan_agent.py`
- Rename: `test_replan_agent_contextimprove.py` → `test_replan_agent.py`
- Rename: `test_single_task_agent_contextimprove.py` → `test_single_task_agent.py`

- [ ] **Step 1: git rm old test files**

```bash
git -C /Users/yunhongmin/Programming/easylocai rm tests/integration_tests/agents/test_plan_agent.py tests/integration_tests/agents/test_replan_agent.py tests/integration_tests/agents/test_single_task_agent.py
```

- [ ] **Step 2: git mv contextimprove test files**

```bash
git -C /Users/yunhongmin/Programming/easylocai mv tests/integration_tests/agents/test_plan_agent_contextimprove.py tests/integration_tests/agents/test_plan_agent.py
git -C /Users/yunhongmin/Programming/easylocai mv tests/integration_tests/agents/test_replan_agent_contextimprove.py tests/integration_tests/agents/test_replan_agent.py
git -C /Users/yunhongmin/Programming/easylocai mv tests/integration_tests/agents/test_single_task_agent_contextimprove.py tests/integration_tests/agents/test_single_task_agent.py
```

- [ ] **Step 3: Update imports in test_plan_agent.py**

Replace the full content of `tests/integration_tests/agents/test_plan_agent.py`:

```python
import pytest

from easylocai.agents.plan_agent import (
    PlanAgent,
    PlanAgentInput,
    PlanAgentOutput,
)
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
```

- [ ] **Step 4: Update imports in test_replan_agent.py**

Replace the full content of `tests/integration_tests/agents/test_replan_agent.py`:

```python
import pytest

from easylocai.agents.replan_agent import (
    ReplanAgent,
    ReplanAgentInput,
    ReplanAgentOutput,
)
from easylocai.schemas.context import ExecutedTaskResult, WorkflowContext


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
```

- [ ] **Step 5: Update imports in test_single_task_agent.py**

Replace the full content of `tests/integration_tests/agents/test_single_task_agent.py`:

```python
import pytest

from easylocai.agents.single_task_agent import (
    SingleTaskAgent,
    SingleTaskAgentOutput,
)
from easylocai.schemas.context import ExecutedTaskResult, SingleTaskAgentContext


class TestSingleTaskAgent:

    @pytest.mark.asyncio
    async def test_basic_task_execution(self, ollama_client, tool_manager):
        agent = SingleTaskAgent(
            client=ollama_client,
            tool_manager=tool_manager,
        )
        context = SingleTaskAgentContext(
            original_user_query="What is 2 + 2?",
            original_task="Calculate 2 + 2",
        )
        output: SingleTaskAgentOutput = await agent.run(context)

        assert isinstance(output.executed_task, str)
        assert len(output.executed_task) > 0
        assert isinstance(output.result, str)
        assert len(output.result) > 0

    @pytest.mark.asyncio
    async def test_task_with_previous_results(self, ollama_client, tool_manager):
        agent = SingleTaskAgent(
            client=ollama_client,
            tool_manager=tool_manager,
        )
        context = SingleTaskAgentContext(
            original_user_query="Find Python files and count them",
            executed_task_results=[
                ExecutedTaskResult(
                    executed_task="Find all Python files",
                    result="Found files: a.py, b.py, c.py",
                )
            ],
            original_task="Count the total number of Python files found",
        )
        output: SingleTaskAgentOutput = await agent.run(context)

        assert isinstance(output.executed_task, str)
        assert isinstance(output.result, str)
```

- [ ] **Step 6: Commit test renames**

```bash
git -C /Users/yunhongmin/Programming/easylocai add tests/integration_tests/agents/test_plan_agent.py tests/integration_tests/agents/test_replan_agent.py tests/integration_tests/agents/test_single_task_agent.py
git -C /Users/yunhongmin/Programming/easylocai commit -m "refactor: rename contextimprove test files to canonical names"
```

---

### Task 5: Verify and push

- [ ] **Step 1: Check no contextimprove references remain**

```bash
grep -r "contextimprove\|ContextImprove" /Users/yunhongmin/Programming/easylocai/easylocai/ /Users/yunhongmin/Programming/easylocai/tests/
```

Expected: no output (zero matches)

- [ ] **Step 2: Run import check**

Write to `/tmp/check_imports.py`:
```python
from easylocai.agents.plan_agent import PlanAgent, PlanAgentInput, PlanAgentOutput
from easylocai.agents.replan_agent import ReplanAgent, ReplanAgentInput, ReplanAgentOutput
from easylocai.agents.single_task_agent import SingleTaskAgent, SingleTaskAgentOutput
from easylocai.workflow import EasylocaiWorkflow
from easylocai.main import run_agent_workflow, workflow_registry
print("All imports OK")
print("Registry keys:", list(workflow_registry.keys()))
```

Run: `python /tmp/check_imports.py`
Expected:
```
All imports OK
Registry keys: ['main']
```

- [ ] **Step 3: Push**

```bash
git -C /Users/yunhongmin/Programming/easylocai push
```
