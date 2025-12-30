import json
import logging
from typing import Any, Optional

from chromadb.types import Collection
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from ollama import AsyncClient
from pydantic import BaseModel

from src.core.agent import Agent
from src.core.contants import DEFAULT_LLM_MODEL
from src.core.server import ServerManager
from src.llm_calls.query_normalizer import (
    QueryNormalizer,
    QueryNormalizerInput,
    Conversation,
    QueryNormalizerOutput,
)
from src.utlis.prompt import pretty_prompt_text

logger = logging.getLogger(__name__)


class ReplanAgentInput(BaseModel):
    user_query: str
    user_contexts: list[dict] = []
    task_results: list[dict] = []
    previous_plan: list[str] = []
    init: bool = True


class ReplanAgentOutput(BaseModel):
    context: str
    tasks: list[dict]
    response: Optional[str]


class ReplanAgent(Agent[ReplanAgentInput, ReplanAgentOutput]):
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
        # TODO: separate initial plan and replan into different agents
        original_user_query = input_.user_query
        previous_conversations = input_.user_contexts
        task_results = input_.task_results
        init = input_.init

        normalizer_input = QueryNormalizerInput(
            user_query=original_user_query,
            previous_conversations=[
                Conversation(
                    user_query=conversation_dict["user_query"],
                    answer=conversation_dict["answer"],
                )
                for conversation_dict in previous_conversations
            ],
        )

        query_normalizer: QueryNormalizer = QueryNormalizer(client=self._ollama_client)
        normalizer_output: QueryNormalizerOutput = await query_normalizer.call(
            normalizer_input
        )
        user_query = normalizer_output.user_query
        user_context = normalizer_output.context

        if init:
            previous_plan = await self._initial_plan(user_query, user_context)
        else:
            previous_plan = input_.previous_plan

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
            user_context=user_context,
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

        revised_plan_dict = json.loads(response["message"]["content"])
        revised_plan = ReplanAgentOutput(**revised_plan_dict, context=user_context)

        return revised_plan

    async def _initial_plan(self, user_query: str, user_context: str) -> Any:
        system_prompt = self._plan_system_prompt_template.render()
        logger.debug(pretty_prompt_text("Plan System Prompt", system_prompt))

        user_prompt = self._plan_user_prompt_template.render(
            user_query=user_query,
            user_context=user_context,
        )
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
        return previous_plan
