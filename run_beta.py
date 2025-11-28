import asyncio
import logging
from contextlib import AsyncExitStack

import chromadb
from ollama import AsyncClient
from rich import get_console

from src.agents.plan_agent import PlanAgent
from src.agents.replan_agent import ReplanAgent
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

    plan_agent = PlanAgent(client=ollama_client)
    replan_agent = ReplanAgent(client=ollama_client)
    single_task_agent = SingleTaskAgent(client=ollama_client)

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

            related_user_context_list = user_context_collection.query(
                query_texts=[user_input],
                n_results=5,
            )["documents"][0]

            plan_query = {
                "user_query": user_input,
            }

            with ConsoleSpinner(console) as spinner:
                spinner.set_prefix("Creating initial plan...")
                plan_agent_response = await plan_agent.run(**plan_query)
                logger.debug(f"Plan Agent Response:\n{plan_agent_response}")
                steps = plan_agent_response["steps"]
                previous_task_results = []
                answer = None

                while True:
                    task = steps[0]

                    iteration_results = []

                    while True:
                        spinner.set_prefix(f"Executing task: {task}...")

                        tool_search_result = tool_collection.query(
                            query_texts=[task],
                            n_results=5,
                        )

                        metadatas = tool_search_result["metadatas"][0]
                        tool_candidates = []

                        for metadata in metadatas:
                            server_name = metadata["server_name"]
                            tool_name = metadata["tool_name"]
                            tool = server_manager.get_server(server_name).get_tool(
                                tool_name
                            )
                            tool_candidates.append(
                                {
                                    "server_name": server_name,
                                    "tool_name": tool.name,
                                    "tool_description": tool.description,
                                    "tool_input_schema": tool.input_schema,
                                }
                            )

                        single_task_agent_query = {
                            "original_tasks": steps,
                            "original_user_query": user_input,
                            "task": task,
                            "previous_task_list": previous_task_results,
                            "iteration_results": iteration_results,
                            "tool_candidates": tool_candidates,
                        }

                        single_task_agent_response = await single_task_agent.run(
                            **single_task_agent_query
                        )
                        logger.debug(
                            f"Single Task Agent Response:\n{single_task_agent_response}"
                        )
                        finished = single_task_agent_response["finished"]

                        if finished:
                            break

                        if single_task_agent_response["use_llm"] is True:
                            # TODO: implement LLM reasoning here
                            break

                        tool_result = await server_manager.call_tool(
                            single_task_agent_response["server_name"],
                            single_task_agent_response["tool_name"],
                            single_task_agent_response.get("tool_args"),
                        )

                        logger.debug(f"original tool result:\n{tool_result}")
                        iteration_results.append(
                            {
                                "executed_task": single_task_agent_response[
                                    "executed_task"
                                ],
                                "result": tool_result,
                            }
                        )

                    # TODO: task / step terminology unification
                    previous_task_results.append(
                        {
                            "task": task,
                            "step": task,
                            "result": iteration_results,
                        }
                    )

                    replan_agent_query = {
                        "original_user_query": user_input,
                        "original_plan": steps,
                        "step_results": previous_task_results,
                    }

                    replan_agent_response = await replan_agent.run(**replan_agent_query)
                    logger.debug(f"Replan Agent Response:\n{replan_agent_response}")

                    response = replan_agent_response["response"]
                    if response is not None:
                        answer = response
                        break

                messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    asyncio.run(main())
