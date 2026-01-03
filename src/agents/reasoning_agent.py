import logging

from ollama import AsyncClient
from pydantic import BaseModel

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.llm_calls.reasoning import Reasoning, ReasoningInput, ReasoningOutput

logger = logging.getLogger(__name__)


class ReasoningAgentInput(BaseModel):
    task: dict
    user_context: str | None


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
            task=input_.task["description"],
            user_context=input_.user_context,
        )
        reasoning = Reasoning(client=self._ollama_client)
        # TODO: adjust think time based on task complexity
        reasoning_output: ReasoningOutput = await reasoning.call(
            reasoning_input, think="medium"
        )

        response = ReasoningAgentOutput(**reasoning_output.model_dump())

        logger.debug(f"{self.__class__.__name__} Response:\n{response}")
        return response
