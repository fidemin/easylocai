import logging

from ollama import AsyncClient
from pydantic import BaseModel, Field

from easylocai.core.agent import Agent
from easylocai.core.contants import DEFAULT_LLM_MODEL
from easylocai.llm_calls.reasoning import Reasoning, ReasoningInput, ReasoningOutput
from easylocai.schemas.context import ConversationHistory

logger = logging.getLogger(__name__)


class ReasoningAgentInput(BaseModel):
    original_task: str
    task: dict
    query_context: str | None
    previous_task_results: list[dict]
    previous_subtask_results: list[dict] = Field(default_factory=list)
    conversation_histories: list[ConversationHistory] = Field(default_factory=list)


class ReasoningAgentOutput(BaseModel):
    reasoning: str
    final: str
    confidence: int


class ReasoningAgent(Agent[ReasoningAgentInput, ReasoningAgentOutput]):
    def __init__(
        self,
        *,
        client: AsyncClient,
    ):
        self._ollama_client = client
        self._model = DEFAULT_LLM_MODEL

    async def run(self, input_: ReasoningAgentInput) -> ReasoningAgentOutput:
        reasoning_input = ReasoningInput(
            original_task=input_.original_task,
            subtask=input_.task["description"],
            query_context=input_.query_context,
            previous_task_results=input_.previous_task_results,
            previous_subtask_results=input_.previous_subtask_results,
            conversation_histories=input_.conversation_histories,
        )
        reasoning = Reasoning(client=self._ollama_client)
        # TODO: adjust think time based on task complexity
        reasoning_output: ReasoningOutput = await reasoning.call(
            reasoning_input, think="medium"
        )

        response = ReasoningAgentOutput(**reasoning_output.model_dump())

        logger.debug(f"{self.__class__.__name__} Response:\n{response}")
        return response
