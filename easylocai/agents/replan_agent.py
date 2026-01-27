import logging
from typing import Optional

from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.llm_calls.replanner import (
    ReplannerV2,
    ReplannerV2Input,
    ReplannerV2Output,
)

logger = logging.getLogger(__name__)


class ReplanAgentV2Input(BaseModel):
    user_query: str
    user_context: str | None
    task_results: list[dict]
    previous_plan: list[str]


class ReplanAgentV2Output(BaseModel):
    tasks: list[str]
    response: Optional[str]


class ReplanAgentV2(Agent[ReplanAgentV2Input, ReplanAgentV2Output]):
    def __init__(
        self,
        *,
        client: AsyncClient,
    ):
        self._ollama_client = client

    async def _run(self, input_: ReplanAgentV2Input) -> ReplanAgentV2Output:
        replanner_input = ReplannerV2Input(
            user_context=input_.user_context,
            original_user_query=input_.user_query,
            previous_plan=input_.previous_plan,
            task_results=input_.task_results,
        )

        replanner = ReplannerV2(client=self._ollama_client)
        replanner_output: ReplannerV2Output = await replanner.call(replanner_input)

        logger.debug(f"ReplanAgentV2 output: {replanner_output}")

        return ReplanAgentV2Output(
            tasks=replanner_output.tasks,
            response=replanner_output.response,
        )
