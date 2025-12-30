import json
import logging
from typing import Optional

from chromadb.types import Collection
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from ollama import AsyncClient
from pydantic import BaseModel

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.core.server import ServerManager
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class ReplanAgentInput(BaseModel):
    user_query: str
    user_context: str | None
    task_results: list[dict] = []
    previous_plan: list[str] = []


class ReplanAgentOutput(BaseModel):
    tasks: list[dict]
    response: Optional[str]


class ReplanAgent(Agent[ReplanAgentInput, ReplanAgentOutput]):
    _plan_system_prompt_path = "resources/prompts/v2/planner_system_prompt.jinja2"
    _plan_user_prompt_path = "resources/prompts/v2/planner_user_prompt.jinja2"
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
        env = Environment(loader=FileSystemLoader(""), undefined=StrictUndefined)

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

    async def _run(self, input_: ReplanAgentInput) -> ReplanAgentOutput:
        previous_plan = input_.previous_plan
        tool_candidates = []
        if previous_plan:
            tool_search_result = self._tool_collection.query(
                query_texts=previous_plan,
                n_results=10,
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
            user_context=input_.user_context,
            original_user_query=input_.user_query,
            previous_plan=previous_plan,
            tool_candidates=tool_candidates,
            task_results=input_.task_results,
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

        revised_plan_dict = json.loads(response["message"]["content"])
        revised_plan = ReplanAgentOutput(**revised_plan_dict)

        return revised_plan
