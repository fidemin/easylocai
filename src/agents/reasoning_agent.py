import json
import logging

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from ollama import AsyncClient
from pydantic import BaseModel

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.utlis.prompt import pretty_prompt_text

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
        task = input_.task

        prompt = self._prompt_template.render(task=task)

        logger.debug(pretty_prompt_text("Reasoning Prompt", prompt))

        response = await self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            options={"temperature": 0.5},
        )
        result_dict = json.loads(response["message"]["content"])
        return ReasoningAgentOutput(**result_dict)
