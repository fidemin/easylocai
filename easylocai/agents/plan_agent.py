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
            conversation_histories=ctx.conversation_histories,
        )
        planner = Planner(client=self._ollama_client)
        planner_output: PlannerOutput = await planner.call(planner_input)

        return PlanAgentOutput(
            query_context=reformatter_output.query_context,
            reformatted_user_query=reformatter_output.reformed_query,
            task_list=planner_output.tasks,
        )
