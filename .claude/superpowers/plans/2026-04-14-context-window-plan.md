# Context Window 개선 (contextimprove) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `--flag=contextimprove` workflow를 새로 만들어 계층적 Context 객체(GlobalContext → WorkflowContext → SingleTaskAgentContext)를 도입한다. 기존 `main` workflow는 변경하지 않는다.

**Architecture:** 각 Context는 독립적인 flat Pydantic 모델이다. Agent는 결과만 반환하며 Context 업데이트 책임은 소유자(workflow)가 가진다. 기존 LLM call 클래스(Planner, Replanner, TaskRouter, ToolSelector 등)는 그대로 재사용한다.

**Tech Stack:** Python 3.12, Pydantic v2, pytest-asyncio, Ollama (gpt-oss:20b)

---

## File Map

| 파일 | 작업 |
|------|------|
| `easylocai/schemas/context.py` | 신규: 모든 Context 스키마 정의 |
| `easylocai/agents/plan_agent_contextimprove.py` | 신규: WorkflowContext 기반 PlanAgent |
| `easylocai/agents/replan_agent_contextimprove.py` | 신규: WorkflowContext 기반 ReplanAgent |
| `easylocai/agents/single_task_agent_contextimprove.py` | 신규: SingleTaskAgentContext 기반 SingleTaskAgent |
| `easylocai/workflow_contextimprove.py` | 신규: contextimprove workflow |
| `easylocai/main_contextimprove.py` | 신규: runner 함수 |
| `easylocai/main.py` | 수정: workflow_registry에 contextimprove 등록 |
| `tests/integration_tests/agents/test_plan_agent_contextimprove.py` | 신규: PlanAgent 통합 테스트 |
| `tests/integration_tests/agents/test_replan_agent_contextimprove.py` | 신규: ReplanAgent 통합 테스트 |
| `tests/integration_tests/agents/test_single_task_agent_contextimprove.py` | 신규: SingleTaskAgent 통합 테스트 |

---

## Task 1: Context 스키마 정의

**Files:**
- Create: `easylocai/schemas/context.py`

- [ ] **Step 1: `easylocai/schemas/context.py` 생성**

```python
from pydantic import BaseModel


class ConversationHistory(BaseModel):
    original_user_query: str
    reformatted_user_query: str
    query_context: str | None = None
    response: str


class GlobalContext(BaseModel):
    conversation_histories: list[ConversationHistory] = []


class ExecutedTaskResult(BaseModel):
    executed_task: str
    result: str


class SubtaskResult(BaseModel):
    subtask: str
    result: str


class WorkflowContext(BaseModel):
    conversation_histories: list[ConversationHistory] = []
    original_user_query: str
    query_context: str | None = None
    reformatted_user_query: str | None = None
    task_list: list[str] = []
    executed_task_results: list[ExecutedTaskResult] = []


class SingleTaskAgentContext(BaseModel):
    # WorkflowContext 필드 전체
    conversation_histories: list[ConversationHistory] = []
    original_user_query: str
    query_context: str | None = None
    reformatted_user_query: str | None = None
    task_list: list[str] = []
    executed_task_results: list[ExecutedTaskResult] = []
    # SingleTaskAgent 전용
    original_task: str
    subtask_results: list[SubtaskResult] = []
```

- [ ] **Step 2: import 확인**

```bash
python -c "from easylocai.schemas.context import GlobalContext, WorkflowContext, SingleTaskAgentContext, ConversationHistory, ExecutedTaskResult, SubtaskResult; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add easylocai/schemas/context.py
git commit -m "feat: add context schemas for contextimprove workflow"
```

---

## Task 2: PlanAgentContextImprove

기존 `PlanAgent`와 동일한 로직이지만 `WorkflowContext`를 받아 처리하고, 결과(query_context, reformatted_user_query, task_list)를 Output으로 반환한다. Context 업데이트는 workflow가 담당한다.

**참고 파일:**
- `easylocai/agents/plan_agent.py` — 기존 구현 참고
- `easylocai/llm_calls/query_reformatter.py` — `QueryReformatterInput(user_query, previous_conversations: list[UserConversation])`
- `easylocai/llm_calls/planner.py` — `PlannerInput(user_query, user_context)`
- `easylocai/schemas/common.py` — `UserConversation(user_query, assistant_answer)`

**주의:** `ConversationHistory`의 필드명(`original_user_query`, `response`)이 `UserConversation`(`user_query`, `assistant_answer`)과 달라 매핑이 필요하다.

**Files:**
- Create: `easylocai/agents/plan_agent_contextimprove.py`
- Create: `tests/integration_tests/agents/test_plan_agent_contextimprove.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/integration_tests/agents/test_plan_agent_contextimprove.py
import pytest

from easylocai.agents.plan_agent_contextimprove import (
    PlanAgentContextImprove,
    PlanAgentContextImproveInput,
    PlanAgentContextImproveOutput,
)
from easylocai.schemas.context import ConversationHistory, WorkflowContext


def assert_output(output: PlanAgentContextImproveOutput) -> None:
    assert isinstance(output.task_list, list)
    assert len(output.task_list) > 0
    for task in output.task_list:
        assert isinstance(task, str)
        assert len(task) > 0
    assert output.reformatted_user_query
    assert output.query_context is None or isinstance(output.query_context, str)


class TestPlanAgentContextImprove:

    @pytest.mark.asyncio
    async def test_basic_query(self, ollama_client):
        """Happy path: conversation_histories 없이 기본 query 처리."""
        agent = PlanAgentContextImprove(client=ollama_client)
        workflow_context = WorkflowContext(
            original_user_query="Find all Python files in the current directory and count them",
        )
        input_ = PlanAgentContextImproveInput(workflow_context=workflow_context)
        output = await agent.run(input_)
        assert_output(output)

    @pytest.mark.asyncio
    async def test_query_with_conversation_history(self, ollama_client):
        """Variant: 이전 대화 기록이 있는 경우."""
        agent = PlanAgentContextImprove(client=ollama_client)
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
        input_ = PlanAgentContextImproveInput(workflow_context=workflow_context)
        output = await agent.run(input_)
        assert_output(output)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

```bash
pytest tests/integration_tests/agents/test_plan_agent_contextimprove.py -v
```

Expected: `ImportError` (모듈 없음)

- [ ] **Step 3: `easylocai/agents/plan_agent_contextimprove.py` 구현**

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


class PlanAgentContextImproveInput(BaseModel):
    workflow_context: WorkflowContext


class PlanAgentContextImproveOutput(BaseModel):
    query_context: str | None
    reformatted_user_query: str
    task_list: list[str]


class PlanAgentContextImprove(Agent[PlanAgentContextImproveInput, PlanAgentContextImproveOutput]):
    def __init__(self, *, client: AsyncClient):
        self._ollama_client = client

    async def _run(self, input_: PlanAgentContextImproveInput) -> PlanAgentContextImproveOutput:
        ctx = input_.workflow_context

        # ConversationHistory → UserConversation 매핑
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

        return PlanAgentContextImproveOutput(
            query_context=reformatter_output.query_context,
            reformatted_user_query=reformatter_output.reformed_query,
            task_list=planner_output.tasks,
        )
```

- [ ] **Step 4: 테스트 실행**

```bash
pytest tests/integration_tests/agents/test_plan_agent_contextimprove.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add easylocai/agents/plan_agent_contextimprove.py tests/integration_tests/agents/test_plan_agent_contextimprove.py
git commit -m "feat: add PlanAgentContextImprove with WorkflowContext input"
```

---

## Task 3: ReplanAgentContextImprove

기존 `ReplanAgent`와 동일한 로직. `WorkflowContext`를 받아 `Replanner`에 전달한다.

**참고 파일:**
- `easylocai/agents/replan_agent.py`
- `easylocai/llm_calls/replanner.py` — `ReplannerInput(user_context, original_user_query, previous_plan: list[str], task_results: list[dict])`
- Jinja template(`replanner_user_prompt.jinja2`)에서 `task_result["task"]`와 `task_result["result"]` 키를 사용함
- `ExecutedTaskResult.executed_task` → dict의 `"task"` 키로 매핑 필요

**Files:**
- Create: `easylocai/agents/replan_agent_contextimprove.py`
- Create: `tests/integration_tests/agents/test_replan_agent_contextimprove.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/integration_tests/agents/test_replan_agent_contextimprove.py
import pytest

from easylocai.agents.replan_agent_contextimprove import (
    ReplanAgentContextImprove,
    ReplanAgentContextImproveInput,
    ReplanAgentContextImproveOutput,
)
from easylocai.schemas.context import ExecutedTaskResult, WorkflowContext


class TestReplanAgentContextImprove:

    @pytest.mark.asyncio
    async def test_replan_with_remaining_tasks(self, ollama_client):
        """Task가 남아있을 때 새 task list를 반환한다."""
        agent = ReplanAgentContextImprove(client=ollama_client)
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
        input_ = ReplanAgentContextImproveInput(workflow_context=workflow_context)
        output: ReplanAgentContextImproveOutput = await agent.run(input_)

        assert isinstance(output.tasks, list)
        assert output.response is None or isinstance(output.response, str)

    @pytest.mark.asyncio
    async def test_replan_all_tasks_complete(self, ollama_client):
        """모든 task 완료 시 response를 반환하고 tasks는 비어있다."""
        agent = ReplanAgentContextImprove(client=ollama_client)
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
        input_ = ReplanAgentContextImproveInput(workflow_context=workflow_context)
        output: ReplanAgentContextImproveOutput = await agent.run(input_)

        assert output.response is not None or len(output.tasks) > 0
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

```bash
pytest tests/integration_tests/agents/test_replan_agent_contextimprove.py -v
```

Expected: `ImportError`

- [ ] **Step 3: `easylocai/agents/replan_agent_contextimprove.py` 구현**

```python
import logging
from typing import Optional

from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.llm_calls.replanner import Replanner, ReplannerInput, ReplannerOutput
from easylocai.schemas.context import WorkflowContext

logger = logging.getLogger(__name__)


class ReplanAgentContextImproveInput(BaseModel):
    workflow_context: WorkflowContext


class ReplanAgentContextImproveOutput(BaseModel):
    tasks: list[str]
    response: Optional[str]


class ReplanAgentContextImprove(Agent[ReplanAgentContextImproveInput, ReplanAgentContextImproveOutput]):
    def __init__(self, *, client: AsyncClient):
        self._ollama_client = client

    async def _run(self, input_: ReplanAgentContextImproveInput) -> ReplanAgentContextImproveOutput:
        ctx = input_.workflow_context

        # ExecutedTaskResult → Replanner가 기대하는 dict 형식으로 변환
        # replanner_user_prompt.jinja2에서 task_result["task"], task_result["result"] 사용
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

        logger.debug(f"ReplanAgentContextImprove output: {replanner_output}")

        return ReplanAgentContextImproveOutput(
            tasks=replanner_output.tasks,
            response=replanner_output.response,
        )
```

- [ ] **Step 4: 테스트 실행**

```bash
pytest tests/integration_tests/agents/test_replan_agent_contextimprove.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add easylocai/agents/replan_agent_contextimprove.py tests/integration_tests/agents/test_replan_agent_contextimprove.py
git commit -m "feat: add ReplanAgentContextImprove with WorkflowContext input"
```

---

## Task 4: SingleTaskAgentContextImprove

기존 `SingleTaskAgent`와 동일한 루프 로직. `SingleTaskAgentContext`를 받아 subtask를 실행하고 `SingleTaskAgentOutput(executed_task, result)`을 반환한다.

**참고 파일:**
- `easylocai/agents/single_task_agent.py` — 전체 구현 참고
- `easylocai/llm_calls/task_router.py` — `TaskRouterInput(task, user_context, tool_candidates, previous_task_results, iteration_results)`
- `easylocai/llm_calls/tool_selector.py` — `ToolSelectorInput(subtask, user_context, tool_candidates, previous_task_results, iteration_results)`

**매핑 주의:**
- `TaskRouterInput.previous_task_results`와 `ToolSelectorInput.previous_task_results`는 `list[dict]`로 `{"task": ..., "result": ...}` 키를 가진다.
- `SingleTaskAgentContext.executed_task_results`는 `list[ExecutedTaskResult]`이므로 변환 필요: `{"task": r.executed_task, "result": r.result}`
- `TaskRouterInput.iteration_results`와 `ToolSelectorInput.iteration_results`는 `{"subtask": ..., "result": ...}` 형식이다.
- `SingleTaskAgentContext.subtask_results`는 `list[SubtaskResult]`로 `subtask_result.model_dump()`로 변환 가능.

**Files:**
- Create: `easylocai/agents/single_task_agent_contextimprove.py`
- Create: `tests/integration_tests/agents/test_single_task_agent_contextimprove.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/integration_tests/agents/test_single_task_agent_contextimprove.py
import pytest

from easylocai.agents.single_task_agent_contextimprove import (
    SingleTaskAgentContextImprove,
    SingleTaskAgentContextImproveOutput,
)
from easylocai.schemas.context import SingleTaskAgentContext


class TestSingleTaskAgentContextImprove:

    @pytest.mark.asyncio
    async def test_basic_task_execution(self, ollama_client, tool_manager):
        """Happy path: tool 없이 reasoning으로 단순 task 처리."""
        agent = SingleTaskAgentContextImprove(
            client=ollama_client,
            tool_manager=tool_manager,
        )
        context = SingleTaskAgentContext(
            original_user_query="What is 2 + 2?",
            original_task="Calculate 2 + 2",
        )
        output: SingleTaskAgentContextImproveOutput = await agent.run(context)

        assert isinstance(output.executed_task, str)
        assert len(output.executed_task) > 0
        assert isinstance(output.result, str)
        assert len(output.result) > 0

    @pytest.mark.asyncio
    async def test_task_with_previous_results(self, ollama_client, tool_manager):
        """Variant: 이전 task 결과가 있는 경우."""
        from easylocai.schemas.context import ExecutedTaskResult

        agent = SingleTaskAgentContextImprove(
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
        output: SingleTaskAgentContextImproveOutput = await agent.run(context)

        assert isinstance(output.executed_task, str)
        assert isinstance(output.result, str)
```

- [ ] **Step 2: 테스트가 실패하는지 확인**

```bash
pytest tests/integration_tests/agents/test_single_task_agent_contextimprove.py -v
```

Expected: `ImportError`

- [ ] **Step 3: `easylocai/agents/single_task_agent_contextimprove.py` 구현**

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


class SingleTaskAgentContextImproveOutput(BaseModel):
    executed_task: str
    result: str


class SingleTaskAgentContextImprove(Agent[SingleTaskAgentContext, SingleTaskAgentContextImproveOutput]):
    N_TOOL_RESULTS = 18

    def __init__(self, *, client: AsyncClient, tool_manager: ToolManager):
        self._ollama_client = client
        self._tool_manager = tool_manager

    async def run(self, input_: SingleTaskAgentContext) -> SingleTaskAgentContextImproveOutput:
        ctx = input_
        tool_candidates = await self._get_tool_candidates([ctx.original_task])

        # executed_task_results → TaskRouter/ToolSelector가 기대하는 dict 형식
        previous_task_results = [
            {"task": r.executed_task, "result": r.result}
            for r in ctx.executed_task_results
        ]

        while True:
            # subtask_results → iteration_results dict 형식
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

        return SingleTaskAgentContextImproveOutput(
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
            logger.error(f"Failed to parse ToolSelector response")
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

- [ ] **Step 4: 테스트 실행**

```bash
pytest tests/integration_tests/agents/test_single_task_agent_contextimprove.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add easylocai/agents/single_task_agent_contextimprove.py tests/integration_tests/agents/test_single_task_agent_contextimprove.py
git commit -m "feat: add SingleTaskAgentContextImprove with SingleTaskAgentContext input"
```

---

## Task 5: Workflow 및 Runner 생성 + Registry 등록

**Files:**
- Create: `easylocai/workflow_contextimprove.py`
- Create: `easylocai/main_contextimprove.py`
- Modify: `easylocai/main.py` (lines 77-79)

- [ ] **Step 1: `easylocai/workflow_contextimprove.py` 생성**

```python
import logging
from contextlib import AsyncExitStack
from typing import AsyncGenerator

from ollama import AsyncClient

from easylocai.agents.plan_agent_contextimprove import (
    PlanAgentContextImprove,
    PlanAgentContextImproveInput,
    PlanAgentContextImproveOutput,
)
from easylocai.agents.replan_agent_contextimprove import (
    ReplanAgentContextImprove,
    ReplanAgentContextImproveInput,
    ReplanAgentContextImproveOutput,
)
from easylocai.agents.single_task_agent_contextimprove import (
    SingleTaskAgentContextImprove,
    SingleTaskAgentContextImproveOutput,
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
                "EasylocaiWorkflowContextImprove is not initialized. "
                "Please call 'initialize' before running."
            )
        async for item in func(self, *args, **kwargs):
            yield item

    return wrapper


class EasylocaiWorkflowContextImprove:
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
        self._plan_agent = PlanAgentContextImprove(client=ollama_client)
        self._replan_agent = ReplanAgentContextImprove(client=ollama_client)
        self._single_task_agent = SingleTaskAgentContextImprove(
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

        plan_output: PlanAgentContextImproveOutput = await self._plan_agent.run(
            PlanAgentContextImproveInput(workflow_context=workflow_context)
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

            task_output: SingleTaskAgentContextImproveOutput = await self._single_task_agent.run(
                single_task_context
            )

            workflow_context.executed_task_results.append(
                ExecutedTaskResult(
                    executed_task=task_output.executed_task,
                    result=task_output.result,
                )
            )

            yield EasyLocaiWorkflowOutput(type="status", message="Check for completion...")

            replan_output: ReplanAgentContextImproveOutput = await self._replan_agent.run(
                ReplanAgentContextImproveInput(workflow_context=workflow_context)
            )
            logger.debug(f"Replan output: {replan_output}")

            if replan_output.response is not None:
                answer = replan_output.response
                break

            workflow_context.task_list = replan_output.tasks

        # GlobalContext 업데이트
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

- [ ] **Step 2: `easylocai/main_contextimprove.py` 생성**

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
from easylocai.workflow_contextimprove import EasylocaiWorkflowContextImprove

logger = logging.getLogger(__name__)


async def run_agent_workflow_contextimprove():
    console = get_console()

    ollama_client = AsyncClient(host="http://localhost:11434")
    search_engine = AdvancedSearchEngine()

    config_path = user_config_path()
    with open(config_path) as f:
        config_dict = json.load(f)

    workflow = EasylocaiWorkflowContextImprove(
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
```

- [ ] **Step 3: `easylocai/main.py`에 registry 등록**

`easylocai/main.py`의 `workflow_registry` dict를 아래와 같이 수정한다:

```python
from easylocai.main_contextimprove import run_agent_workflow_contextimprove

workflow_registry = {
    "main": run_agent_workflow_main,
    "contextimprove": run_agent_workflow_contextimprove,
}
```

- [ ] **Step 4: import 확인**

```bash
python -c "from easylocai.main import workflow_registry; print(list(workflow_registry.keys()))"
```

Expected: `['main', 'contextimprove']`

- [ ] **Step 5: 실행 smoke test**

```bash
python -m easylocai.run --flag=contextimprove
```

`> ` 프롬프트가 나타나면 성공. `Ctrl+C`로 종료.

- [ ] **Step 6: Commit**

```bash
git add easylocai/workflow_contextimprove.py easylocai/main_contextimprove.py easylocai/main.py
git commit -m "feat: add contextimprove workflow with hierarchical context objects"
```

---

## Self-Review

**Spec 커버리지:**
- [x] GlobalContext, WorkflowContext, SingleTaskAgentContext 스키마 정의 → Task 1
- [x] GlobalContext는 독립 데이터 객체, main_contextimprove.py가 소유 → Task 5
- [x] WorkflowContext 업데이트는 workflow가 담당 → Task 5
- [x] SingleTaskAgentContext는 WorkflowContext 필드 전체 포함 → Task 1, 4
- [x] subtask_tools는 context에 미포함, 로컬 변수로만 사용 → Task 4
- [x] `--flag=contextimprove` 등록 → Task 5
- [x] 기존 `main` workflow 미변경 → 신규 파일만 생성

**Type 일관성:**
- `ExecutedTaskResult.executed_task` → `{"task": ..., "result": ...}` 변환이 Task 3, 4, 5 모두에서 일관되게 적용됨
- `SubtaskResult.model_dump()` → `{"subtask": ..., "result": ...}` — Jinja 템플릿과 일치
- `ConversationHistory` → `UserConversation` 매핑이 Task 2에서만 발생하며 명시적으로 처리됨
