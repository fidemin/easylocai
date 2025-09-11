from jinja2 import Environment, FileSystemLoader
from ollama import Client

from src.core.agent import Agent
from src.core.server import ServerManager
from src.utlis.prompt import print_prompt


class PlanningAgent(Agent):
    _prompt_path = "resources/prompts/planning_prompt.txt"

    def __init__(
        self,
        *,
        client: Client,
        model: str,
    ):
        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._prompt_path)
        self._prompt_template = prompt_template
        self._model = model

    def _chat(self, query: str):
        prompt = self._prompt_template.render()
        print_prompt("Planning prompt", prompt)
        response = self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )
        return response["message"]["content"]


class DetailPlanningAgent(Agent):
    _system_prompt_path = "resources/prompts/detail_planning_system_prompt.txt"

    def __init__(
        self,
        *,
        client: Client,
        collection,
        model: str,
        server_manager: ServerManager,
    ):
        self._ollama_client = client
        self._collection = collection
        self._server_manager = server_manager

        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._system_prompt_path)
        self._prompt_template = prompt_template
        self._model = model

    def _chat(self, query: str | dict):
        user_goal = query["user_goal"]
        plans = query["plans"]

        possible_tools = []
        tool_results = self._collection.query(query_texts=plans, n_results=2)
        for tool_ids in tool_results["ids"]:
            for id_ in tool_ids:
                server_name, tool_name = id_.split(":")
                tool = self._server_manager.get_server(server_name).get_tool(tool_name)

                possible_tools.append(
                    {
                        "tool_description": tool.description,
                        "tool_input_schema": tool.input_schema,
                    }
                )

        system_prompt = self._prompt_template.render(possible_tools=possible_tools)
        print_prompt("detail planning prompt", system_prompt)

        user_inputs = [
            f"<goal>{user_goal}</goal>",
            "<plans>",
            *[plan for plan in plans],
            "</plans>",
        ]

        user_query = "\n".join(user_inputs)

        response = self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )
        return response


class AnswerAgent(Agent):
    _system_prompt_path = "resources/prompts/answer_system_prompt.txt"

    def __init__(
        self,
        *,
        client: Client,
        model: str,
    ):
        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._system_prompt_path)
        self._prompt_template = prompt_template
        self._model = model

    def _chat(self, query: str | dict):
        user_query = query["user_query"]
        tool_results = query["tool_results"]
        system_prompt = self._prompt_template.render(
            user_query=user_query,
            tool_results=tool_results,
        )
        print_prompt("answer prompt", system_prompt)
        response = self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
            ],
        )
        return response
