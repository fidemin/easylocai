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
            query_context=ctx.query_context,
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
