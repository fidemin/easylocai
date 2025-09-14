from chromadb.types import Collection
from jinja2 import Environment, FileSystemLoader

from src.core.agent import Agent
from src.core.server import ServerManager
from src.utlis.prompt import print_prompt


class TaskToolAgent(Agent):
    _prompt_path = "resources/prompts/task_tool_prompt.txt"

    def __init__(
        self,
        *,
        client,
        model: str,
        tool_collection: Collection,
        server_manager: ServerManager,
    ):
        self._client = client
        env = Environment(loader=FileSystemLoader(""))
        prompt_template = env.get_template(self._prompt_path)
        self._prompt_template = prompt_template
        self._model = model
        self._tool_collection = tool_collection
        self._server_manager = server_manager

    def _chat(self, query: str | dict):
        original_user_query = query["original_user_query"]
        plan = query["plan"]

        previous_task_results: list[dict] = query["previous_task_results"]
        task_result_str_list = []
        for previous_task_result in previous_task_results:
            task = previous_task_result["task"]
            result = previous_task_result["result"]
            task_result_str = f"<task>{task}</task>\n<result>{result}</result>\n"
            task_result_str_list.append(task_result_str)

        task_results_context = "\n".join(task_result_str_list)

        tool_search_result = self._tool_collection.query(
            query_texts=[plan],
            n_results=5,
        )

        metadatas = tool_search_result["metadatas"][0]
        possible_tools = []

        for metadata in metadatas:
            server_name = metadata["server_name"]
            tool_name = metadata["tool_name"]
            tool = self._server_manager.get_server(server_name).get_tool(tool_name)
            possible_tools.append(
                {
                    "server_name": server_name,
                    "tool_name": tool.name,
                    "tool_description": tool.description,
                    "tool_input_schema": tool.input_schema,
                }
            )

        prompt = self._prompt_template.render(
            original_user_query=original_user_query,
            plan=plan,
            possible_tools=possible_tools,
            task_results_context=task_results_context,
        )

        print_prompt("Task Tool Prompt", prompt)

        response = self._client.chat(
            model=self._model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        return response["message"]["content"]
