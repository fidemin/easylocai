import logging
from typing import Optional, Any

from chromadb.types import Collection
from ollama import AsyncClient
from pydantic import BaseModel

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.core.server import ServerManager
from src.llm_calls.planner import Planner, PlannerInput, PlannerOutput
from src.llm_calls.query_normalizer import (
    QueryNormalizer,
    QueryNormalizerInput,
    Conversation,
    QueryNormalizerOutput,
)
from src.llm_calls.replanner import ReplannerInput, Replanner, ReplannerOutput

logger = logging.getLogger(__name__)


class PlanAgentInput(BaseModel):
    user_query: str
    user_conversations: list[dict] = []


class PlanAgentOutput(BaseModel):
    context: str | None
    tasks: list[dict]
    response: Optional[str]


class PlanAgent(Agent[PlanAgentInput, PlanAgentOutput]):
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

    async def _run(self, input_: PlanAgentInput) -> PlanAgentOutput:
        original_user_query = input_.user_query
        previous_conversations = input_.user_conversations

        normalizer_output = await self._normalize_query(
            original_user_query, previous_conversations
        )

        user_query = normalizer_output.user_query
        user_context = normalizer_output.user_context

        planner_output = await self._initial_plan(user_query, user_context)
        previous_plan = planner_output.tasks

        tool_candidates = await self._fetch_tool_candidates(previous_plan)

        replanner_output = await self._replan(
            user_context=user_context,
            user_query=user_query,
            previous_plan=previous_plan,
            tool_candidates=tool_candidates,
        )

        revised_plan = PlanAgentOutput(
            tasks=replanner_output.tasks,
            response=replanner_output.response,
            context=user_context,
        )

        return revised_plan

    async def _normalize_query(
        self, original_user_query: str, previous_conversations: list[dict]
    ) -> QueryNormalizerOutput:
        normalizer_input = QueryNormalizerInput(
            user_query=original_user_query,
            previous_conversations=[
                Conversation(
                    user_query=conversation_dict["user_query"],
                    answer=conversation_dict["answer"],
                )
                for conversation_dict in previous_conversations
            ],
        )

        query_normalizer: QueryNormalizer = QueryNormalizer(client=self._ollama_client)
        normalizer_output: QueryNormalizerOutput = await query_normalizer.call(
            normalizer_input
        )
        return normalizer_output

    async def _initial_plan(self, user_query: str, user_context: str) -> PlannerOutput:
        planner = Planner(client=self._ollama_client)
        planner_input = PlannerInput(
            user_query=user_query,
            user_context=user_context,
        )

        planner_output: PlannerOutput = await planner.call(planner_input)
        return planner_output

    async def _fetch_tool_candidates(self, previous_plan: list[str]) -> list[Any]:
        if not previous_plan:
            return []

        tool_search_result = self._tool_collection.query(
            query_texts=previous_plan,
            n_results=10,
        )

        tool_candidates = []

        for metadatas in tool_search_result["metadatas"]:
            for metadata in metadatas:
                server_name = metadata["server_name"]
                tool_name = metadata["tool_name"]
                tool = self._server_manager.get_server(server_name).get_tool(tool_name)
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
        user_context: str | None,
        user_query: str,
        previous_plan: list[str],
        tool_candidates: list[Any],
    ) -> ReplannerOutput:
        replanner_input = ReplannerInput(
            user_context=user_context,
            original_user_query=user_query,
            previous_plan=previous_plan,
            tool_candidates=tool_candidates,
            task_results=[],  # No task results in the initial planning phase
        )
        replanner = Replanner(client=self._ollama_client)

        replanner_output = await replanner.call(replanner_input)
        return replanner_output
