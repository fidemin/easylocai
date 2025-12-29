import logging
from abc import ABC
from typing import Generic, TypeVar, Any, Type

from jinja2 import Template, Environment, FileSystemLoader, StrictUndefined
from ollama import AsyncClient
from pydantic import BaseModel

from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)

InModel = TypeVar("InModel", bound=BaseModel)
OutModel = TypeVar("OutModel", bound=BaseModel)


class LLMCall(ABC, Generic[InModel, OutModel]):
    _client: AsyncClient
    _model: str
    _options: dict[str, Any]
    _output_model: Type[OutModel]

    _system_prompt_template: Template
    _user_prompt_template: Template

    def __init__(
        self,
        client: AsyncClient,
        model: str,
        options: dict[str, Any],
        system_prompt_path: str,
        user_prompt_path: str,
        output_model: Type[OutModel],
    ):
        self._client = client
        self._model = model
        self._options = options

        env = Environment(loader=FileSystemLoader(""), undefined=StrictUndefined)
        self._system_prompt_template = env.get_template(system_prompt_path)
        self._user_prompt_template = env.get_template(user_prompt_path)
        self._output_model = output_model

    async def call(self, input_: InModel) -> OutModel:
        system_prompt = self._system_prompt_template.render()
        logger.debug(
            pretty_prompt_text(
                f"{self.__class__.__name__} System Prompt", system_prompt
            )
        )
        user_prompt = self._user_prompt_template.render(**input_.model_dump())
        logger.debug(
            pretty_prompt_text(f"{self.__class__.__name__} User Prompt", user_prompt)
        )

        llm_call_response = await self._client.chat(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt_template.render(),
                },
                {
                    "role": "user",
                    "content": self._user_prompt_template.render(**input_.model_dump()),
                },
            ],
            options=self._options,
        )

        content = llm_call_response["message"]["content"]
        response = self._output_model.model_validate_json(content)

        logger.debug(f"{self.__class__.__name__} Response:\n{response}")
        return response
