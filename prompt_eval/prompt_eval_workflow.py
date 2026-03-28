import json
from typing import Type

from jinja2 import FileSystemLoader, Environment, Template
from ollama import AsyncClient
from pydantic import BaseModel


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

    async def run(self):
        results = await self.run_and_collect()
        for result in results:
            self._print(
                {"id": result["id"], "messages": result["messages"]},
                result["response"],
                result["thinking"],
            )

    async def run_and_collect(self) -> list[dict]:
        """
        Run all test cases and return structured results.

        Each result dict contains:
          - id: test case id (may be None)
          - messages: rendered chat messages [{"role": ..., "content": ...}]
          - response: raw LLM response string
          - thinking: model thinking/reasoning string (may be None)
        """
        with open(self._input_file_path, "r") as file:
            input_dict_list = json.loads(file.read())

        system_prompt_path = self._prompt_path_info.get("system")
        system_prompt_template = self._get_prompt_template(system_prompt_path)

        user_prompt_path = self._prompt_path_info.get("user", None)
        user_prompt_template = self._get_prompt_template(user_prompt_path)

        chat_input_list = []

        for input_dict in input_dict_list:
            chat_messages = []
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
        results = []

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

    def _get_prompt_template(self, prompt_path: str | None) -> Template | None:
        if prompt_path is None:
            return None
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(prompt_path)
        return prompt_template

    def _print(self, chat_input, response, thinking):
        id_ = chat_input["id"]
        print(f"----- Prompt Eval Result (ID: {id_}) -----")
        print("<Chat Input>")
        for message in chat_input["messages"]:
            role = message["role"]
            content = message["content"]
            print(f"[{role.upper()}]\n{content}")
        print("</Chat Input>")
        print("<Response>")
        print(response)
        print("</Response>")
        print("<Thinking>")
        print(thinking)
        print("</Thinking>")
        print("-------------------------")
        print("")
