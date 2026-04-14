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
