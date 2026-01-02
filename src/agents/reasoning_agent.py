import logging

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from ollama import AsyncClient
from pydantic import BaseModel

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.llm_calls.reasoning import Reasoning, ReasoningInput, ReasoningOutput

logger = logging.getLogger(__name__)


class ReasoningAgentInput(BaseModel):
    task: dict


class ReasoningAgentOutput(BaseModel):
    reasoning: str
    final: str
    confidence: int


class ReasoningAgent(Agent[ReasoningAgentInput, ReasoningAgentOutput]):
    _prompt_path = "resources/prompts/v2/reasoning_prompt.jinja2"

    def __init__(
        self,
        *,
        client: AsyncClient,
        prompt_path: str | None = None,
    ):
        if prompt_path is not None:
            self._prompt_path = prompt_path

        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""), undefined=StrictUndefined)
        prompt_template = env.get_template(self._prompt_path)
        self._prompt_template = prompt_template
        self._model = DEFAULT_LLM_MODEL

    async def run(self, input_: ReasoningAgentInput) -> ReasoningAgentOutput:
        reasoning_input = ReasoningInput(task=input_.task["description"])
        reasoning = Reasoning(client=self._ollama_client)
        # TODO: adjust think time based on task complexity
        reasoning_output: ReasoningOutput = await reasoning.call(
            reasoning_input, think="medium"
        )

        response = ReasoningAgentOutput(**reasoning_output.model_dump())

        logger.debug(f"{self.__class__.__name__} Response:\n{response}")
        return response
