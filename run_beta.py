import asyncio
import logging
from contextlib import AsyncExitStack

import chromadb
from ollama import AsyncClient
from rich import get_console

from src.agents.plan_agent import PlanAgent
from src.agents.single_task_agent import SingleTaskAgent
from src.core.server import ServerManager
from src.utlis.console_util import multiline_input, render_chat, ConsoleSpinner
from src.utlis.loggers.default_dict import default_logging_config

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

    tool_collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )


async def main():
    console = get_console()
    ollama_client = AsyncClient(host="http://localhost:11434")

    chromadb_client = chromadb.Client()
    user_context_collection = chromadb_client.get_or_create_collection("user_context")
    tool_collection = chromadb_client.get_or_create_collection("tools")

    server_manager = ServerManager.from_json_config_file("mcp_server_config.json")

    plan_agent = PlanAgent(
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

            # TODO: apply user context
            related_user_context_list = user_context_collection.query(
                query_texts=[user_input],
                n_results=5,
            )["documents"][0]

            plan_query = {
                "user_query": user_input,
                "user_contexts": [],
            }

            with ConsoleSpinner(console) as spinner:
                spinner.set_prefix("Planning...")
                plan_agent_response = await plan_agent.run(**plan_query)
                logger.debug(f"Plan Agent Response:\n{plan_agent_response}")

                if plan_agent_response["response"] is not None:
                    answer = plan_agent_response["response"]
                    messages.append({"role": "assistant", "content": answer})
                    continue

                tasks = plan_agent_response["tasks"]
                previous_task_results = []

                while True:
                    task = tasks[0]
                    spinner.set_prefix(task["description"])
                    task_agent_query = {
                        "original_tasks": tasks,
                        "original_user_query": user_input,
                        "task": task,
                        "previous_task_list": previous_task_results,
                    }
                    task_agent_response = await single_task_agent.run(
                        **task_agent_query
                    )

                    previous_task_results.append(task_agent_response)

                    spinner.set_prefix("Check for completion...")
                    plan_agent_response = await plan_agent.run(
                        init=False,
                        previous_plan=[task["description"] for task in tasks],
                        task_results=previous_task_results,
                        **plan_query,
                    )
                    logger.debug(f"Plan Agent Response:\n{plan_agent_response}")

                    response = plan_agent_response["response"]

                    if response is not None:
                        answer = response
                        break

                    tasks = plan_agent_response["tasks"]

                messages.append({"role": "assistant", "content": answer})
                plan_query["user_contexts"].append(
                    {
                        "query": user_input,
                        "answer": answer,
                    }
                )


if __name__ == "__main__":
    asyncio.run(main())
