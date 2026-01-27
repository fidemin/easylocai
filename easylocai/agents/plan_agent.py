import logging

from ollama import AsyncClient
from pydantic import BaseModel

from easylocai.core.agent import Agent
from easylocai.llm_calls.planner import Planner, PlannerInput, PlannerOutput
from easylocai.llm_calls.query_reformatter import (
    QueryReformatterInput,
    QueryReformatter,
    QueryReformatterOutput,
)
from easylocai.schemas.common import UserConversation

logger = logging.getLogger(__name__)


class PlanAgentInput(BaseModel):
    user_query: str
    user_conversations: list[UserConversation] = []


class PlanAgentOutput(BaseModel):
    context: str | None
    tasks: list[str]


class PlanAgent(Agent[PlanAgentInput, PlanAgentOutput]):
    def __init__(
        self,
        *,
        client: AsyncClient,
    ):
        self._ollama_client = client

    async def _run(self, input_: PlanAgentInput) -> PlanAgentOutput:
        original_user_query = input_.user_query
        previous_conversations = input_.user_conversations

        reformatter_output: QueryReformatterOutput = await self._reformat_query(
            original_user_query, previous_conversations
        )

        user_query = reformatter_output.reformed_query
        user_context = reformatter_output.query_context

        planner_output = await self._initial_plan(user_query, user_context)

        revised_plan = PlanAgentOutput(
            tasks=planner_output.tasks,
            context=user_context,
        )

        return revised_plan

    async def _reformat_query(
        self, original_user_query: str, previous_conversations: list[UserConversation]
    ) -> QueryReformatterOutput:
        reformatter_input = QueryReformatterInput(
            user_query=original_user_query,
            previous_conversations=previous_conversations,
        )

        query_reformatter: QueryReformatter = QueryReformatter(
            client=self._ollama_client
        )

        reformatter_output: QueryReformatterOutput = await query_reformatter.call(
            reformatter_input
        )
        return reformatter_output

    async def _initial_plan(self, user_query: str, user_context: str) -> PlannerOutput:
        planner = Planner(client=self._ollama_client)
        planner_input = PlannerInput(
            user_query=user_query,
            user_context=user_context,
        )

        planner_output: PlannerOutput = await planner.call(planner_input)
        return planner_output
