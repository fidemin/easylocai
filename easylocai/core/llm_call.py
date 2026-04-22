import logging
from abc import ABC
from typing import Generic, TypeVar, Any, Type, Union, AsyncIterator

from jinja2 import Template, Environment, FileSystemLoader, StrictUndefined
from ollama import AsyncClient, ChatResponse
from pydantic import BaseModel, RootModel, ValidationError

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

    async def call(self, input_: InModel, *, think=None, max_retries: int = 3) -> OutModel:
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

        # If output model is a RootModel (e.g., RootModel[str]), no format is provided to LLM. (pure text)
        # Otherwise, provide JSON schema to LLM for structured output.
        # Caution: If RootModel is sent to LLM with format, wrong result is returned.
        if issubclass(self._output_model, RootModel):
            output_model_format = None
        else:
            output_model_format = self._output_model.model_json_schema()

        messages = [
            {
                "role": "system",
                "content": self._system_prompt_template.render(),
            },
            {
                "role": "user",
                "content": self._user_prompt_template.render(**input_.model_dump()),
            },
        ]

        last_error: Exception | None = None
        for attempt in range(max_retries):
            llm_call_response = await self._client.chat(
                model=self._model,
                messages=messages,
                options=self._options,
                think=think,
                format=output_model_format,
            )

            self._current_llm_call_response = llm_call_response
            content = llm_call_response["message"]["content"]

            if not content:
                logger.warning(
                    f"{self.__class__.__name__} received empty response "
                    f"(attempt {attempt + 1}/{max_retries}), retrying..."
                )
                last_error = ValueError("LLM returned empty response")
                continue

            try:
                if issubclass(self._output_model, RootModel):
                    # RootModel[str] case: validate plain text
                    response = self._output_model.model_validate(content)
                else:
                    # BaseModel object output: JSON validate
                    response = self._output_model.model_validate_json(content)
                logger.debug(f"{self.__class__.__name__} Response:\n{response}")
                return response
            except ValidationError as e:
                thinking = llm_call_response["message"].get("thinking")
                logger.error(
                    f"{self.__class__.__name__} failed to parse response "
                    f"(attempt {attempt + 1}/{max_retries}): {content}"
                    + (f"\nThinking: {thinking}" if thinking else "")
                )
                last_error = e

        raise last_error
