import asyncio
import json
import logging
from contextlib import AsyncExitStack

import chromadb
from jinja2 import FileSystemLoader, Environment
from ollama import AsyncClient

from src.core.server import ServerManager
from src.plannings.agent import PlanningAgent, DetailPlanningAgent, AnswerAgent
from src.utlis.prompt import print_prompt

logger = logging.getLogger(__name__)

AI_MODEL = "gpt-oss:20b"


def convert_to_tool_txt_data(server_name, tool_info_dict):
    tool_description_lines = []
    server_name = f"- server_name: {server_name}"
    tool_description_lines.append(server_name)

    for tool_name, tool_info in tool_info_dict.items():
        tool_name_text = f"    - tool_name: {tool_name}"
        tool_description_lines.append(tool_name_text)
        tool_info_text = f"    - Tool Info: {tool_info}"
        tool_description_lines.append(tool_info_text)

    tool_description = "\n".join(tool_description_lines)
    return tool_description


def convert_to_tool_data(server_name, tool_name, tool_info):
    tool_description_lines = []
    tool_desc_text = f"- description: {tool_info["description"]}"
    tool_description_lines.append(tool_desc_text)

    tool_description = "\n".join(tool_description_lines)
    return tool_description


async def create_mcp_tool_request_body(
    client,
    user_query,
    filtered_user_context_list,
    possible_tools,
    task_results,
):
    filtered_user_context = "\n".join(
        [
            filtered_user_context["document"]
            for filtered_user_context in filtered_user_context_list
        ]
    )

    task_results_context = "\n".join(task_results)

    env = Environment(loader=FileSystemLoader(""))
    template = env.get_template("resources/prompts/tool_assistant.txt")
    prompt = template.render(
        {
            "possible_tools": possible_tools,
            "user_query": user_query,
            "filtered_user_context": filtered_user_context,
            "task_results_context": task_results_context,
        }
    )

    print_prompt("Tool Prompt", prompt)

    response = await client.chat(
        model=AI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    content = response["message"]["content"]

    print(f"Tool Assistant: {content}")

    # Extract JSON from response
    try:
        parsed = json.loads(content.strip())
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"LLM did not return valid JSON: {content} with error: {str(e)}")
        print("❌ LLM did not return valid JSON:", content)
        return None


async def filter_tool_result(
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

    print_prompt("Tool Filter Prompt", prompt)

    tooL_result_str = (
        "TOOL RESULT:\n" + tooL_result
        if isinstance(tooL_result, str)
        else str(tooL_result)
    )

    response = await ollama_client.chat(
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
    chromadb_client = chromadb.Client()
    tool_collection = chromadb_client.get_or_create_collection("tools")
    user_context_collection = chromadb_client.get_or_create_collection("user_context")

    server_manager = ServerManager.from_json_config_file("mcp_server_config.json")

    stack = AsyncExitStack()
    async with stack:
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

        ollama_client = AsyncClient(host="http://localhost:11434")

        planning_agent = PlanningAgent(
            client=ollama_client,
            model=AI_MODEL,
        )

        detail_planning_agent = DetailPlanningAgent(
            client=ollama_client,
            collection=tool_collection,
            model=AI_MODEL,
            server_manager=server_manager,
        )

        answer_agent = AnswerAgent(
            client=ollama_client,
            model=AI_MODEL,
        )

        query_id = 1
        while True:
            user_input = input("\nQuery: ")

            if user_input.strip().lower() in {"exit", "quit"}:
                break

            json_text = await planning_agent.run(user_input)
            print(json_text)
            data = json.loads(json_text)
            if not data.get("plans"):
                print(data["answer"])
                continue

            detail_plan_query = {
                "user_goal": user_input,
                "plans": data.get("plans"),
            }

            planning_contents_str = await detail_planning_agent.run(detail_plan_query)
            print(planning_contents_str)
            planning_contents = json.loads(planning_contents_str)
            answer = planning_contents["answer"]
            print(answer)

            task_results = []

            for task in planning_contents["tasks"]:
                tool_query = task["action"]

                tool_search_result = tool_collection.query(
                    query_texts=[tool_query],
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

                user_context_result = user_context_collection.query(
                    query_texts=[tool_query],
                    n_results=10,
                )

                threshold = 1.5  # cosign distance [0, 2] for [same, opposite]
                filtered_results = []
                for i, distance in enumerate(user_context_result["distances"][0]):
                    if distance < threshold:
                        filtered_results.append(
                            {
                                "id": user_context_result["ids"][0][i],
                                "document": user_context_result["documents"][0][i],
                            }
                        )

                filtered_results.sort(key=lambda x: int(x["id"]))

                tool_call = await create_mcp_tool_request_body(
                    ollama_client,
                    tool_query,
                    filtered_results,
                    possible_tools,
                    task_results,
                )

                if tool_call is None:
                    print("invalid tool call. please try again")
                    continue

                if not tool_call["use_tool"]:
                    print("No possible tools. please try again")
                    continue

                chosen_server_name = tool_call["server_name"]
                chosen_tool_name = tool_call["tool_name"]

                server = server_manager.get_server(chosen_server_name)
                result = await server.call_tool(
                    chosen_tool_name, tool_call.get("tool_args")
                )

                tool_search_result = result.content[0].text

                print("original tool result:\n", tool_search_result)
                filtered_tool_result = await filter_tool_result(
                    ollama_client, user_input, task["action"], tool_search_result
                )
                print("filtered tool result:\n", filtered_tool_result)
                task_results.append(filtered_tool_result)

                user_context_document = (
                    f"- Question: {tool_query}\n- Answer: {filtered_tool_result}"
                )
                user_context_collection.add(
                    documents=[user_context_document],
                    ids=[str(query_id)],
                )
                query_id += 1

            chat_input = {
                "user_query": user_input,
                "tool_results": task_results,
            }
            response = await answer_agent.run(chat_input)
            print(response)


if __name__ == "__main__":
    asyncio.run(main())
