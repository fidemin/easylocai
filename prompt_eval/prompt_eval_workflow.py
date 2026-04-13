import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Type

from jinja2 import FileSystemLoader, Environment, Template
from ollama import AsyncClient
from pydantic import BaseModel


@dataclass
class EvalResultItem:
    id: str | None
    response: str
    expected: Any
    thinking: str | None


@dataclass
class EvalMetadata:
    config: str | None
    system_prompt: str | None
    user_prompt: str | None
    input_file: str
    output_model: str | None
    model: str
    date: str
    total: int


@dataclass
class EvalOutput:
    metadata: EvalMetadata
    results: list[EvalResultItem]

    def to_text(self) -> str:
        col_width = 60

        def truncate(val, width):
            val = str(val) if val is not None else ""
            val = val.replace("\n", " ")
            return val[:width] + "..." if len(val) > width else val

        headers = ["ID", "Response", "Expected", "Thinking"]
        header_row = " | ".join(f"{h:<{col_width}}" for h in headers)
        separator = "-+-".join("-" * col_width for _ in headers)
        lines = [
            f"## Eval Results — {self.metadata.config or 'unknown'} — {self.metadata.date}\n",
            f"| {header_row} |",
            f"|-{separator}-|",
        ]
        for r in self.results:
            row = " | ".join([
                f"{truncate(r.id, col_width):<{col_width}}",
                f"{truncate(r.response, col_width):<{col_width}}",
                f"{truncate(r.expected, col_width):<{col_width}}",
                f"{truncate(r.thinking, col_width):<{col_width}}",
            ])
            lines.append(f"| {row} |")

        lines += [
            "",
            "## Summary",
            *(
                [f"Config:        {self.metadata.config}"]
                if self.metadata.config else []
            ),
            f"System prompt: {self.metadata.system_prompt or '-'}",
            f"User prompt:   {self.metadata.user_prompt or '-'}",
            f"Input file:    {self.metadata.input_file}",
            f"Output model:  {self.metadata.output_model or '-'}",
            f"Model:         {self.metadata.model}",
            f"Total:         {self.metadata.total} cases",
        ]
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "metadata": {
                "config": self.metadata.config,
                "system_prompt": self.metadata.system_prompt,
                "user_prompt": self.metadata.user_prompt,
                "input_file": self.metadata.input_file,
                "output_model": self.metadata.output_model,
                "model": self.metadata.model,
                "date": self.metadata.date,
            },
            "results": [
                {
                    "id": r.id,
                    "response": r.response,
                    "expected": r.expected,
                    "thinking": r.thinking,
                }
                for r in self.results
            ],
            "summary": {"total": self.metadata.total},
        }


class PromptEvalWorkflow:
    def __init__(
        self,
        *,
        prompt_path_info: dict,
        input_file_path: str,
        model_info: dict,
        user_input_schema: Type[BaseModel] | None = None,
        output_model: Type[BaseModel] | None = None,
    ):
        self._prompt_path_info = prompt_path_info
        self._input_file_path = input_file_path
        self._model_info = model_info
        self._user_input_schema = user_input_schema
        self._output_model = output_model

    async def run(self, config_path: str | None = None, output_file: str | None = None, output_format: str = "text"):
        results = await self.run_and_collect()
        output = self._build_output(results, config_path)
        text = output.to_text()
        print(text)
        if output_file:
            content = json.dumps(output.to_json(), indent=2, ensure_ascii=False) if output_format == "json" else text
            with open(output_file, "w") as f:
                f.write(content)
            print(f"Results saved to: {output_file}")

    async def run_and_collect(self) -> list[dict]:
        """
        Run all test cases and return structured results.

        Each result dict contains:
          - id: test case id (may be None)
          - messages: rendered chat messages [{"role": ..., "content": ...}]
          - response: raw LLM response string
          - thinking: model thinking/reasoning string (may be None)
          - expected: expected output (may be None)
          - scoring_criteria: scoring guidance (may be None)
        """
        with open(self._input_file_path, "r") as file:
            input_dict_list = json.loads(file.read())

        system_prompt_path = self._prompt_path_info.get("system")
        system_prompt_template = self._get_prompt_template(system_prompt_path)

        user_prompt_path = self._prompt_path_info.get("user", None)
        user_prompt_template = self._get_prompt_template(user_prompt_path)

        chat_input_list: list[dict] = []

        for input_dict in input_dict_list:
            chat_messages: list[dict] = []
            id_ = input_dict.get("id", None)

            for message in input_dict["messages"]:
                role = message["role"]
                argument = message["arguments"]
                if role == "system":
                    if system_prompt_template is None:
                        raise ValueError(
                            "System prompt template is not provided for system role."
                        )
                    chat_messages.append(
                        {
                            "role": "system",
                            "content": system_prompt_template.render(**argument),
                        }
                    )
                elif role == "user":
                    if user_prompt_template is None:
                        raise ValueError(
                            "User prompt template is not provided for user role."
                        )
                    if self._user_input_schema is not None:
                        argument_schema = self._user_input_schema(**argument)
                        user_prompt = user_prompt_template.render(
                            **argument_schema.model_dump()
                        )
                    else:
                        user_prompt = user_prompt_template.render(**argument)
                    chat_messages.append({"role": "user", "content": user_prompt})
                elif role == "assistant":
                    assistant_response = argument.get("response", "")
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": f"{assistant_response}",
                        }
                    )
            chat_input_list.append(
                {
                    "id": id_,
                    "messages": chat_messages,
                    "expected": input_dict.get("expected"),
                    "scoring_criteria": input_dict.get("scoring_criteria"),
                }
            )

        ollama_client = AsyncClient(host=self._model_info["host"])
        results: list[dict] = []

        for chat_input in chat_input_list:
            if self._output_model:
                response = await ollama_client.chat(
                    model=self._model_info["model"],
                    messages=chat_input["messages"],
                    options=self._model_info.get("options"),
                    format=self._output_model.model_json_schema(),
                )
            else:
                response = await ollama_client.chat(
                    model=self._model_info["model"],
                    messages=chat_input["messages"],
                    options=self._model_info.get("options"),
                )
            results.append(
                {
                    "id": chat_input["id"],
                    "messages": chat_input["messages"],
                    "response": response["message"]["content"],
                    "thinking": response["message"]["thinking"],
                    "expected": chat_input.get("expected"),
                    "scoring_criteria": chat_input.get("scoring_criteria"),
                }
            )

        return results

    def _build_output(self, results: list[dict], config_path: str | None) -> EvalOutput:
        output_model_str: str | None = (
            self._output_model.__module__ + "." + self._output_model.__name__
            if self._output_model else None
        )
        metadata = EvalMetadata(
            config=config_path,
            system_prompt=self._prompt_path_info.get("system"),
            user_prompt=self._prompt_path_info.get("user"),
            input_file=self._input_file_path,
            output_model=output_model_str,
            model=self._model_info.get("model", "-"),
            date=date.today().isoformat(),
            total=len(results),
        )
        items = [
            EvalResultItem(
                id=r.get("id"),
                response=r.get("response", ""),
                expected=r.get("expected"),
                thinking=r.get("thinking"),
            )
            for r in results
        ]
        return EvalOutput(metadata=metadata, results=items)

    def _get_prompt_template(self, prompt_path: str | None) -> Template | None:
        if prompt_path is None:
            return None
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(prompt_path)
        return prompt_template
