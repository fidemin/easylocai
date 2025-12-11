import json

from jinja2 import FileSystemLoader, Environment, Template
from ollama import AsyncClient


class PromptEvalWorkflow:
    def __init__(
        self, *, prompt_path_info: dict, input_file_path: str, model_info: dict
    ):
        self._prompt_path_info = prompt_path_info
        self._input_file_path = input_file_path
        self._model_info = model_info

    async def run(self):
        """
        input_dict structure

        input_dict = {
          "messages": [
            {
              "role": "system",
              "argument": {}
            },
            {
              "role": "user",
              "argument": {
                "user_query": "Explain the theory of relativity."
              }
            }
          ]
        }
        """
        input_dict_list = None
        with open(self._input_file_path, "r") as file:
            input_ = file.read()
            input_dict_list = json.loads(input_)

        system_prompt_path = self._prompt_path_info.get("system")
        system_prompt_template = self._get_prompt_template(system_prompt_path)

        user_prompt_path = self._prompt_path_info.get("user", None)
        user_prompt_template = self._get_prompt_template(user_prompt_path)

        chat_input_list = []

        for input_dict in input_dict_list:
            chat_input = []

            for message in input_dict["messages"]:
                role = message["role"]
                argument = message["arguments"]
                if role == "system":
                    if system_prompt_template is None:
                        raise ValueError(
                            "System prompt template is not provided for system role."
                        )

                    system_prompt_template.render(**argument)
                    chat_input.append(
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

                    user_prompt = user_prompt_template.render(**argument)
                    chat_input.append({"role": "user", "content": user_prompt})
                elif role == "assistant":
                    assistant_response = argument.get("response", "")
                    chat_input.append(
                        {
                            "role": "assistant",
                            "content": f"{assistant_response}",
                        }
                    )
            chat_input_list.append(chat_input)

        ollama_client = AsyncClient(host=self._model_info["host"])

        for chat_input in chat_input_list:
            response = await ollama_client.chat(
                model=self._model_info["model"],
                messages=chat_input,
                options=self._model_info.get("options"),
            )
            self._print(chat_input, response["message"]["content"])

    def _get_prompt_template(self, prompt_path: str | None) -> Template | None:
        if prompt_path is None:
            return None
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(prompt_path)
        return prompt_template

    def _print(self, chat_input, response):
        print("<Chat Input>")
        for message in chat_input:
            role = message["role"]
            content = message["content"]
            print(f"[{role.upper()}]\n{content}")
        print("</Chat Input>")
        print("<Response>")
        print(response)
        print("</Response>")
        print("-------------------------")
        print("")
