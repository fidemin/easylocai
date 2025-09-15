import asyncio
import logging
from contextlib import AsyncExitStack

import chromadb
from jinja2 import Environment, FileSystemLoader
from ollama import Client

from src.core.server import ServerManager
from src.plannings.agent import NextPlanAgent
from src.tools.agent import TaskToolAgent
from src.utlis.loggers.default_dict import default_logging_config
from src.utlis.prompt import pretty_prompt_text

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


def filter_tool_result(
    ollama_client,
    user_query: str,
    task: str,
    tooL_result: str,
):
    env = Environment(loader=FileSystemLoader(""))
    template = env.get_template("resources/prompts/tool_result_prompt.txt")
    prompt = template.render(
        {"user_query": user_query, "task": task, "tooL_result": tooL_result}
    )

    logger.debug(pretty_prompt_text("Tool Filter Prompt", prompt))

    tooL_result_str = (
        "TOOL RESULT:\n" + tooL_result
        if isinstance(tooL_result, str)
        else str(tooL_result)
    )

    response = ollama_client.chat(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": tooL_result_str,
            },
        ],
    )
    return response["message"]["content"]


async def main():
    ollama_client = Client(host="http://localhost:11434")

    chromadb_client = chromadb.Client()
    tool_collection = chromadb_client.get_or_create_collection("tools")

    next_plan_agent = NextPlanAgent(
        client=ollama_client,
        model=AI_MODEL,
    )

    server_manager = ServerManager.from_json_config_file("mcp_server_config.json")

    stack = AsyncExitStack()
    async with stack:
        await initialize_tools(stack, server_manager, tool_collection)

        while True:
            user_input = input("\nUser >> ")

            if user_input.strip().lower() in {"exit", "quit"}:
                break

            next_plan_query = {
                "original_user_query": user_input,
                "previous_task_results": [],
            }

            while True:
                next_plan_data = next_plan_agent.chat(next_plan_query)
                logger.debug(f"Next Plan Response:\n{next_plan_data}")

                if next_plan_data["answer"]:
                    print("Assistant >> " + next_plan_data["answer"])
                    break

                next_plan = next_plan_data["next_plan"].strip()

                task_tool_agent = TaskToolAgent(
                    client=ollama_client,
                    model=AI_MODEL,
                    tool_collection=tool_collection,
                    server_manager=server_manager,
                )

                task_tool_data = task_tool_agent.chat(
                    {
                        "original_user_query": user_input,
                        "plan": next_plan,
                        "previous_task_results": next_plan_query[
                            "previous_task_results"
                        ],
                    }
                )
                logger.debug(f"Task Tool Response:\n{task_tool_data}")

                if task_tool_data["use_llm"] is True:
                    task = task_tool_data["task"]
                    response = ollama_client.chat(
                        model=AI_MODEL,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are task execution AI assistant. Answer the user's query as best as you can.",
                            },
                            {
                                "role": "user",
                                "content": f"{task}",
                            },
                        ],
                    )
                    result = response["message"]["content"]
                    logger.debug(f"LLM Result: {result}")
                    next_plan_query["previous_task_results"].append(
                        {
                            "task": task,
                            "result": result,
                        }
                    )
                    continue

                if task_tool_data["use_tool"] is False:
                    print("Assistant >> No tool to use, stop executing.")
                    break

                tool_result = await server_manager.call_tool(
                    task_tool_data["server_name"],
                    task_tool_data["tool_name"],
                    task_tool_data.get("tool_args"),
                )

                logger.debug(f"original tool result:\n{tool_result}")
                filtered_tool_result = filter_tool_result(
                    ollama_client,
                    user_input,
                    task_tool_data["task"],
                    tool_result,
                )
                logger.debug(f"filtered tool result:\n{filtered_tool_result}")

                next_plan_query["previous_task_results"].append(
                    {
                        "task": task_tool_data["task"],
                        "result": filtered_tool_result,
                    }
                )


if __name__ == "__main__":
    asyncio.run(main())
