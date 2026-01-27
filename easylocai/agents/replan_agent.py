import logging
from typing import Optional

from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.llm_calls.replanner import (
    Replanner,
    ReplannerInput,
    ReplannerOutput,
)

logger = logging.getLogger(__name__)


class ReplanAgentInput(BaseModel):
    user_query: str
    user_context: str | None
    task_results: list[dict]
    previous_plan: list[str]


class ReplanAgentOutput(BaseModel):
    tasks: list[str]
    response: Optional[str]


class ReplanAgent(Agent[ReplanAgentInput, ReplanAgentOutput]):
    def __init__(
        self,
        *,
        client: AsyncClient,
    ):
        self._ollama_client = client

    async def _run(self, input_: ReplanAgentInput) -> ReplanAgentOutput:
        replanner_input = ReplannerInput(
            user_context=input_.user_context,
            original_user_query=input_.user_query,
            previous_plan=input_.previous_plan,
            task_results=input_.task_results,
        )

        replanner = Replanner(client=self._ollama_client)
        replanner_output: ReplannerOutput = await replanner.call(replanner_input)

        logger.debug(f"ReplanAgentV2 output: {replanner_output}")

        return ReplanAgentOutput(
            tasks=replanner_output.tasks,
            response=replanner_output.response,
        )
