import json
import logging

from chromadb.types import Collection
from jinja2 import Environment, FileSystemLoader
from ollama import AsyncClient

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.core.server import ServerManager
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class PlanAgent(Agent):
    _plan_system_prompt_path = "resources/prompts/v2/plan_system_prompt.jinja2"
    _plan_user_prompt_path = "resources/prompts/v2/plan_user_prompt.jinja2"
    _replan_system_prompt_path = "resources/prompts/v2/replan_system_prompt.jinja2"
    _replan_user_prompt_path = "resources/prompts/v2/replan_user_prompt.jinja2"

    def __init__(
        self,
        *,
        client: AsyncClient,
        tool_collection: Collection,
        server_manager: ServerManager,
    ):
        self._ollama_client = client
        env = Environment(loader=FileSystemLoader(""))

        self._tool_collection = tool_collection
        self._server_manager = server_manager

        plan_system_prompt_template = env.get_template(self._plan_system_prompt_path)
        self._plan_system_prompt_template = plan_system_prompt_template
        plan_user_prompt_template = env.get_template(self._plan_user_prompt_path)
        self._plan_user_prompt_template = plan_user_prompt_template

        replan_system_prompt_template = env.get_template(
            self._replan_system_prompt_path
        )
        self._replan_system_prompt_template = replan_system_prompt_template
        replan_user_prompt_template = env.get_template(self._replan_user_prompt_path)
        self._replan_user_prompt_template = replan_user_prompt_template

        self._model = DEFAULT_LLM_MODEL

    async def run(self, init=True, **query) -> str | dict:
        user_query = query["user_query"]
        task_results = query.get("task_results", [])

        if init:
            system_prompt = self._plan_system_prompt_template.render()
            logger.debug(pretty_prompt_text("Plan System Prompt", system_prompt))

            user_prompt = self._plan_user_prompt_template.render(user_query=user_query)
            logger.debug(pretty_prompt_text("Plan User Prompt", user_prompt))

            response = await self._ollama_client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.2},
            )
            previous_plan = json.loads(response["message"]["content"])["tasks"]
        else:
            previous_plan = query["previous_plan"]

        logger.debug(f"Previous Plan: {previous_plan}")

        tool_candidates = []

        if previous_plan:
            tool_search_result = self._tool_collection.query(
                query_texts=previous_plan,
                n_results=5,
            )

            tool_candidates = []

            for metadatas in tool_search_result["metadatas"]:
                for metadata in metadatas:
                    server_name = metadata["server_name"]
                    tool_name = metadata["tool_name"]
                    tool = self._server_manager.get_server(server_name).get_tool(
                        tool_name
                    )
                    tool_candidates.append(
                        {
                            "tool_name": tool.name,
                            "tool_description": tool.description,
                        }
                    )

        system_prompt = self._replan_system_prompt_template.render()
        logger.debug(pretty_prompt_text("Replan System Prompt", system_prompt))

        user_prompt = self._replan_user_prompt_template.render(
            original_user_query=user_query,
            previous_plan=previous_plan,
            tool_candidates=tool_candidates,
            task_results=task_results,
        )
        logger.debug(pretty_prompt_text("Replan User Prompt", user_prompt))

        response = await self._ollama_client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.2},
        )

        revised_plan = json.loads(response["message"]["content"])
        return revised_plan
