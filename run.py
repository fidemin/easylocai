import asyncio
import logging
from contextlib import AsyncExitStack
from datetime import datetime

import chromadb
from ollama import AsyncClient

from src.core.server import ServerManager
from src.plannings.agent import NextPlanAgent, AnswerAgent
from src.tools.agent import ToolAgent
from src.utlis.loggers.default_dict import default_logging_config

logging.config.dictConfig(default_logging_config)

logger = logging.getLogger(__name__)


AI_MODEL = "gpt-oss:20b"


async def initialize_tools(stack, server_manager, tool_collection):
    for server in server_manager.list_servers():
        await server.initialize(stack)

        tools = await server.list_tools()
        logger.info(
            f"Server Name: {server.name}, Available tools: {[tool.name for tool in tools]}"
        )
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
    ollama_client = AsyncClient(host="http://localhost:11434")

    chromadb_client = chromadb.Client()
    user_context_collection = chromadb_client.get_or_create_collection("user_context")
    tool_collection = chromadb_client.get_or_create_collection("tools")

    server_manager = ServerManager.from_json_config_file("mcp_server_config.json")

    next_plan_agent = NextPlanAgent(
        client=ollama_client,
        model=AI_MODEL,
    )

    tool_agent = ToolAgent(
        client=ollama_client,
        model=AI_MODEL,
        tool_collection=tool_collection,
        server_manager=server_manager,
    )

    answer_agent = AnswerAgent(
        client=ollama_client,
        model=AI_MODEL,
    )

    stack = AsyncExitStack()
    async with stack:
        await initialize_tools(stack, server_manager, tool_collection)

        while True:
            user_input = input("\nUser >> ")

            if user_input.strip().lower() in {"exit", "quit"}:
                break

            related_user_context_list = user_context_collection.query(
                query_texts=[user_input],
                n_results=5,
            )["documents"][0]

            next_plan_query = {
                "original_user_query": user_input,
                "previous_task_results": [],
                "user_context_list": related_user_context_list,
            }

            while True:
                next_plan_data = await next_plan_agent.run(next_plan_query)
                logger.debug(f"Next Plan Response:\n{next_plan_data}")

                if not next_plan_data["continue"]:
                    answer_agent_input = {
                        "user_query": user_input,
                        "task_results": next_plan_query["previous_task_results"],
                        "user_context_list": related_user_context_list,
                    }
                    response = await answer_agent.run(answer_agent_input)
                    print("Assistant >>\n" + response)

                    created_at = datetime.now().isoformat()
                    user_context = "\n".join(
                        [
                            f"User Query: {user_input}",
                            f"Assitant Response: {response}",
                            f"Created At: {created_at}",
                        ]
                    )
                    user_context_collection.add(
                        documents=[user_context],
                        metadatas=[{"created_at": created_at}],
                        ids=[f"user_context_{user_context_collection.count()}"],
                    )
                    break

                next_plan = next_plan_data["next_plan"].strip()

                task_result = await tool_agent.run(
                    {
                        "original_user_query": user_input,
                        "plan": next_plan,
                        "previous_task_results": next_plan_query[
                            "previous_task_results"
                        ],
                    }
                )
                next_plan_query["previous_task_results"].append(task_result)


if __name__ == "__main__":
    asyncio.run(main())
