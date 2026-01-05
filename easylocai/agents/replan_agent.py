import logging
from typing import Optional

from chromadb.types import Collection
from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.core.contants import DEFAULT_LLM_MODEL
from easylocai.core.server import ServerManager
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
        tool_collection: Collection,
        server_manager: ServerManager,
    ):
        self._ollama_client = client
        self._tool_collection = tool_collection
        self._server_manager = server_manager
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
        tool_candidates = []
        if previous_plan:
            tool_search_result = self._tool_collection.query(
                query_texts=previous_plan,
                n_results=10,
            )

            tool_candidates = []

            for metadatas in tool_search_result["metadatas"]:
                for metadata in metadatas:
                    server_name = metadata["server_name"]
                    tool_name = metadata["tool_name"]
                    tool = self._server_manager.get_server(server_name).get_tool(
                        tool_name
                    )
                    tool_candidates.append(
                        {
                            "tool_name": tool.name,
                            "tool_description": tool.description,
                        }
                    )
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
