import asyncio
import json
from contextlib import AsyncExitStack

import chromadb
from ollama import Client

from src.core.server import ServerManager
from src.plannings.agent import NextPlanAgent


async def initialize_tools(stack, server_manager, tool_collection):
    for server in server_manager.list_servers():
        await server.initialize(stack)

        tools = await server.list_tools()
        print(
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
    ollama_client = Client(host="http://localhost:11434")

    chromadb_client = chromadb.Client()
    tool_collection = chromadb_client.get_or_create_collection("tools")

    next_plan_agent = NextPlanAgent(
        client=ollama_client,
        model="gpt-oss:20b",
    )

    server_manager = ServerManager.from_json_config_file("mcp_server_config.json")

    stack = AsyncExitStack()
    async with stack:
        await initialize_tools(stack, server_manager, tool_collection)

        while True:
            user_input = input("\nQuery: ")

            if user_input.strip().lower() in {"exit", "quit"}:
                break

            next_plan_query = {
                "original_user_query": user_input,
                "previous_task_results": [],
            }

            while True:
                response = next_plan_agent.chat(next_plan_query)
                print(response)

                data = json.loads(response)
                if data["direct_answer"]:
                    print(data["direct_answer"])
                    break

                if data["continue"] is False:
                    # TODO: Make answer result based on task results
                    break

                next_plan = data["next_plan"].strip()
                tool_search_result = tool_collection.query(
                    query_texts=[next_plan],
                    n_results=5,
                )

                metadatas = tool_search_result["metadatas"][0]
                possible_tools = []

                for metadata in metadatas:
                    server_name = metadata["server_name"]
                    tool_name = metadata["tool_name"]
                    tool = server_manager.get_server(server_name).get_tool(tool_name)
                    possible_tools.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool.name,
                            "tool_description": tool.description,
                            "tool_input_schema": tool.input_schema,
                        }
                    )

                temp_next_task = input("\nNext Task: ")
                temp_next_result = input("\nNext Result: ")

                next_plan_query["previous_task_results"].append(
                    {
                        "task": temp_next_task,
                        "result": temp_next_result,
                    }
                )


if __name__ == "__main__":
    asyncio.run(main())
