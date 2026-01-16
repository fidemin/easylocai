import json
import logging
from contextlib import AsyncExitStack

import chromadb
from ollama import AsyncClient
from rich import get_console

from easylocai.agents.plan_agent import PlanAgent, PlanAgentInput, PlanAgentOutput
from easylocai.agents.replan_agent import (
    ReplanAgent,
    ReplanAgentInput,
    ReplanAgentOutput,
)
from easylocai.agents.single_task_agent import (
    SingleTaskAgent,
    SingleTaskAgentInput,
    SingleTaskAgentOutput,
)
from easylocai.config import user_config_path
from easylocai.core.tool_manager import ToolManager
from easylocai.schemas.common import UserConversation
from easylocai.utlis.console_util import multiline_input, render_chat, ConsoleSpinner

logger = logging.getLogger(__name__)


async def run_agent_flow():
    console = get_console()
    ollama_client = AsyncClient(host="http://localhost:11434")

    chromadb_client = chromadb.Client()

    config_path = user_config_path()

    with open(config_path) as f:
        config_dict = json.load(f)

    tool_manager = ToolManager(chromadb_client, mpc_servers=config_dict["mcpServers"])

    plan_agent = PlanAgent(
        client=ollama_client,
        tool_manager=tool_manager,
    )

    replan_agent = ReplanAgent(
        client=ollama_client,
        tool_manager=tool_manager,
    )
    single_task_agent = SingleTaskAgent(
        client=ollama_client,
        tool_manager=tool_manager,
    )

    messages = []
    user_conversations: list[UserConversation] = []

    stack = AsyncExitStack()
    async with stack:
        await tool_manager.initialize(stack)

        while True:
            render_chat(console, messages)
            user_input = await multiline_input("> ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            messages.append({"role": "user", "content": user_input})
            render_chat(console, messages)

            plan_agent_input = PlanAgentInput(
                user_query=user_input,
                user_conversations=user_conversations,
            )

            with ConsoleSpinner(console) as spinner:
                spinner.set_prefix("Planning...")
                plan_agent_output: PlanAgentOutput = await plan_agent.run(
                    plan_agent_input
                )
                logger.debug(f"Plan Agent Response:\n{plan_agent_output}")

                if plan_agent_output.response is not None:
                    messages.append(
                        {"role": "assistant", "content": plan_agent_output.response}
                    )
                    render_chat(console, messages)
                    user_conversations.append(
                        UserConversation(
                            user_query=user_input,
                            assistant_answer=plan_agent_output.response,
                        )
                    )
                    continue

                tasks = plan_agent_output.tasks
                user_context = plan_agent_output.context
                previous_task_results = []

                answer = None
                while True:
                    next_task = tasks[0]
                    spinner.set_prefix(next_task["description"])

                    task_agent_input = SingleTaskAgentInput(
                        original_tasks=tasks,
                        original_user_query=user_input,
                        task=next_task,
                        previous_task_results=previous_task_results,
                        user_context=user_context,
                    )

                    task_agent_response: SingleTaskAgentOutput = (
                        await single_task_agent.run(task_agent_input)
                    )

                    previous_task_results.append(
                        {
                            "task": task_agent_response.task,
                            "result": task_agent_response.result,
                        }
                    )

                    spinner.set_prefix("Check for completion...")
                    replan_agent_input = ReplanAgentInput(
                        user_query=user_input,
                        previous_plan=[task["description"] for task in tasks],
                        task_results=previous_task_results,
                        user_context=user_context,
                    )
                    replan_agent_output: ReplanAgentOutput = await replan_agent.run(
                        replan_agent_input
                    )
                    logger.debug(f"Plan Agent Response:\n{plan_agent_output}")

                    response = replan_agent_output.response

                    if response is not None:
                        answer = response
                        break

                    tasks = replan_agent_output.tasks

                messages.append({"role": "assistant", "content": answer})
                render_chat(console, messages)
                user_conversations.append(
                    UserConversation(user_query=user_input, assistant_answer=answer)
                )
