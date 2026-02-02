import logging
from abc import ABC
from typing import Generic, TypeVar, Any, Type, Union, AsyncIterator

from jinja2 import Template, Environment, FileSystemLoader, StrictUndefined
from ollama import AsyncClient, ChatResponse
from pydantic import BaseModel, RootModel

from easylocai.utlis.prompt import pretty_prompt_text
from easylocai.utlis.resource_util import installed_resources_dir

logger = logging.getLogger(__name__)

resources_dir = installed_resources_dir()

InModel = TypeVar("InModel", bound=BaseModel)
OutModel = TypeVar("OutModel", bound=BaseModel)


class LLMCallV2(ABC, Generic[InModel, OutModel]):
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
        system_prompt_path: str | None,
        user_prompt_path: str | None,
        output_model: Type[OutModel],
    ):
        self._client = client
        self._model = model
        self._options = options

        env = Environment(
            loader=FileSystemLoader(str(resources_dir)),
            undefined=StrictUndefined,
        )
        if system_prompt_path is not None:
            self._system_prompt_template = env.get_template(system_prompt_path)
        if user_prompt_path is not None:
            self._user_prompt_template = env.get_template(user_prompt_path)
        self._output_model = output_model

    @property
    def llm_call_response(
        self,
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse], None]:
        return self._current_llm_call_response

    async def call(self, input_: InModel, *, think=None) -> OutModel:
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

        output_model_format = None

        if issubclass(self._output_model, BaseModel):
            output_model_format = self._output_model.model_json_schema()
        else:
            output_model_format = None

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
            think=think,
            format=self._output_model.model_json_schema(),
        )

        self._current_llm_call_response = llm_call_response

        content = llm_call_response["message"]["content"]

        if issubclass(self._output_model, RootModel):
            # RootModel[str] 같은 경우: content가 그냥 텍스트여도 validate 가능
            response = self._output_model.model_validate(content)
        else:
            # BaseModel object output: JSON 형태일 때 validate
            response = self._output_model.model_validate_json(content)

        logger.debug(f"{self.__class__.__name__} Response:\n{response}")
        return response
