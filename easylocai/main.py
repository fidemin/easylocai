import json
import logging
from contextlib import AsyncExitStack

import chromadb
from ollama import AsyncClient
from rich import get_console

from easylocai.agents.plan_agent import (
    PlanAgentInput,
    PlanAgentBeta,
    PlanAgentOutputBeta,
)
from easylocai.agents.replan_agent import (
    ReplanAgentV2,
    ReplanAgentV2Input,
    ReplanAgentV2Output,
)
from easylocai.agents.single_task_agent_v2 import (
    SingleTaskAgentV2,
    SingleTaskAgentV2Input,
    SingleTaskAgentV2Output,
)
from easylocai.config import user_config_path
from easylocai.core.tool_manager import ToolManager
from easylocai.schemas.common import UserConversation
from easylocai.utlis.console_util import multiline_input, render_chat, ConsoleSpinner

logger = logging.getLogger(__name__)


async def run_agent_flow(flag: str | None = None):
    console = get_console()
    ollama_client = AsyncClient(host="http://localhost:11434")

    chromadb_client = chromadb.Client()

    config_path = user_config_path()

    with open(config_path) as f:
        config_dict = json.load(f)

    tool_manager = ToolManager(chromadb_client, mpc_servers=config_dict["mcpServers"])
    plan_agent_beta = PlanAgentBeta(client=ollama_client)
    replan_agent_v2 = ReplanAgentV2(client=ollama_client)
    single_task_agent_v2 = SingleTaskAgentV2(
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
                spinner.set_prefix("Thinking...")

                plan_agent_output: PlanAgentOutputBeta = await plan_agent_beta.run(
                    plan_agent_input
                )
                logger.debug(f"Plan Agent Response:\n{plan_agent_output}")

                tasks = plan_agent_output.tasks
                user_context = plan_agent_output.context
                previous_task_results = []

                answer = None
                while True:
                    next_task = tasks[0]

                    spinner.set_prefix(next_task)

                    task_agent_input_v2 = SingleTaskAgentV2Input(
                        original_user_query=user_input,
                        task=next_task,
                        previous_task_results=previous_task_results,
                        user_context=user_context,
                    )

                    task_agent_response_v2: SingleTaskAgentV2Output = (
                        await single_task_agent_v2.run(task_agent_input_v2)
                    )

                    previous_task_results.append(
                        {
                            "task": task_agent_response_v2.task,
                            "result": task_agent_response_v2.result,
                        }
                    )

                    spinner.set_prefix("Check for completion...")
                    replan_agent_input_v2 = ReplanAgentV2Input(
                        user_query=user_input,
                        previous_plan=tasks,
                        task_results=previous_task_results,
                        user_context=user_context,
                    )

                    replan_agent_output: ReplanAgentV2Output = (
                        await replan_agent_v2.run(replan_agent_input_v2)
                    )
                    logger.debug(f"ReplanAgentV2 Response:\n{replan_agent_output}")

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
