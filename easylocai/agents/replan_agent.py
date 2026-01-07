import logging
from typing import Optional

from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.core.contants import DEFAULT_LLM_MODEL
from easylocai.core.tool_manager import ToolManager
from easylocai.llm_calls.replanner import Replanner, ReplannerInput, ReplannerOutput

logger = logging.getLogger(__name__)


class ReplanAgentInput(BaseModel):
    user_query: str
    user_context: str | None
    task_results: list[dict]
    previous_plan: list[str]


class ReplanAgentOutput(BaseModel):
    tasks: list[dict]
    response: Optional[str]


class ReplanAgent(Agent[ReplanAgentInput, ReplanAgentOutput]):
    def __init__(
        self,
        *,
        client: AsyncClient,
        tool_manager: ToolManager,
    ):
        self._ollama_client = client
        self._tool_manager = tool_manager
        self._model = DEFAULT_LLM_MODEL

    async def _run(self, input_: ReplanAgentInput) -> ReplanAgentOutput:
        previous_plan = input_.previous_plan
        tool_candidates = await self._fetch_tool_candidates(previous_plan)

        replanner_output = await self._replan(
            user_context=input_.user_context,
            user_query=input_.user_query,
            previous_plan=previous_plan,
            tool_candidates=tool_candidates,
            input_=input_,
        )

        revised_plan = ReplanAgentOutput(
            tasks=replanner_output.tasks,
            response=replanner_output.response,
        )

        return revised_plan

    async def _fetch_tool_candidates(self, previous_plan: list[str]) -> list[dict]:
        if not previous_plan:
            return []

        tools = self._tool_manager.search_tools(previous_plan, n_results=10)

        tool_candidates = [
            {
                "tool_name": tool.name,
                "tool_description": tool.description,
            }
            for tool in tools
        ]

        return tool_candidates

    async def _replan(
        self,
        *,
        input_: ReplanAgentInput,
        user_context: str | None,
        user_query: str,
        previous_plan: list[str],
        tool_candidates: list[dict],
    ) -> ReplannerOutput:
        replanner_input = ReplannerInput(
            user_context=user_context,
            original_user_query=user_query,
            previous_plan=previous_plan,
            tool_candidates=tool_candidates,
            task_results=input_.task_results,
        )
        replanner = Replanner(client=self._ollama_client)

        replanner_output = await replanner.call(replanner_input)
        return replanner_output
