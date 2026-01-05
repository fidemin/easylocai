import argparse
import asyncio
import logging
import sys
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
from easylocai.config import user_config_path, ensure_user_config
from easylocai.core.server import ServerManager
from easylocai.schemas.common import UserConversation
from easylocai.utlis.console_util import multiline_input, render_chat, ConsoleSpinner
from easylocai.utlis.loggers.default_dict import default_logging_config

logging.config.dictConfig(default_logging_config)

logger = logging.getLogger(__name__)


AI_MODEL = "gpt-oss:20b"


async def initialize_tools(server_manager: ServerManager, tool_collection):
    # initialize chromadb tool
    ids = []
    metadatas = []
    documents = []
    for server in server_manager.list_servers():
        for tool in await server.list_tools():
            id_ = f"{server.name}:{tool.name}"
            metadata = {
                "server_name": server.name,
                "tool_name": tool.name,
            }
            ids.append(id_)
            documents.append(tool.description)
            metadatas.append(metadata)

    if len(ids) == 0:
        return
    tool_collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )


async def main():
    console = get_console()
    ollama_client = AsyncClient(host="http://localhost:11434")

    chromadb_client = chromadb.Client()
    tool_collection = chromadb_client.get_or_create_collection("tools")

    config_path = user_config_path()
    server_manager = ServerManager.from_json_config_file(config_path)

    plan_agent = PlanAgent(
        client=ollama_client,
        tool_collection=tool_collection,
        server_manager=server_manager,
    )

    replan_agent = ReplanAgent(
        client=ollama_client,
        tool_collection=tool_collection,
        server_manager=server_manager,
    )
    single_task_agent = SingleTaskAgent(
        client=ollama_client,
        tool_collection=tool_collection,
        server_manager=server_manager,
    )

    messages = []
    user_conversations: list[UserConversation] = []

    stack = AsyncExitStack()
    async with stack:
        await server_manager.initialize_servers(stack)
        await initialize_tools(server_manager, tool_collection)

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
                user_conversations.append(
                    UserConversation(user_query=user_input, assistant_answer=answer)
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="easylocai",
        description="Easy Local AI CLI",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ADD easylocai init command
    # easylocai init
    init_parser = subparsers.add_parser(
        "init",
        help="Create config at ~/.config/easylocai/config.json",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config",
    )
    return parser


def run() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init":
        path = ensure_user_config(overwrite=args.force)
        print(f"Config initialized at: {path}")
        return 0

    ensure_user_config(overwrite=False)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(run())
