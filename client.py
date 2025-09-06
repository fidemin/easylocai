import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack

import chromadb
from jinja2 import FileSystemLoader, Environment
from mcp import StdioServerParameters, stdio_client, ClientSession
from ollama import Client

from src.planning.agent import PlanningAgent, DetailPlanningAgent
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
    # tool_name_text = f"- name: {tool_name}"
    # tool_description_lines.append(tool_name_text)
    tool_desc_text = f"- description: {tool_info["description"]}"
    tool_description_lines.append(tool_desc_text)
    # tool_input_schema_text = f"- input schema: {tool_info["input_schema"]}"
    # tool_description_lines.append(tool_input_schema_text)

    tool_description = "\n".join(tool_description_lines)
    return tool_description


def choose_mcp_tool(
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

    response = client.chat(
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


async def main():
    chromadb_client = chromadb.Client()
    tool_collection = chromadb_client.get_or_create_collection("tools")
    user_context_collection = chromadb_client.get_or_create_collection("user_context")

    with open("mcp_server_config.json") as f:
        servers = json.load(f)["mcpServers"]

        server_params_dict = {}
        for server_name, config in servers.items():
            if "env" in config:
                env = {}
                for k, v in config["env"].items():
                    env[k] = os.path.expandvars(v)
            server_params = StdioServerParameters(
                **config,
            )

            server_params_dict[server_name] = server_params

    stack = AsyncExitStack()
    async with stack:
        session_dict = {}
        server_name_tool_info_dict = {}
        for server_name, server_params in server_params_dict.items():
            stdio, write = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()
            session_dict[server_name] = session

            tools_response = await session.list_tools()
            print(
                f"Server Name: {server_name}, Available tools: {[tool.name for tool in tools_response.tools]}"
            )

            tool_info = {
                tool.name: {
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in tools_response.tools
            }
            server_name_tool_info_dict[server_name] = tool_info

        # initialize chromadb tool
        ids = []
        documents = []
        for server_name, tool_info_dict in server_name_tool_info_dict.items():
            for tool_name, tool_info in tool_info_dict.items():
                id_ = f"{server_name}:{tool_name}"
                ids.append(id_)

                document = convert_to_tool_data(server_name, tool_name, tool_info)
                documents.append(document)

        tool_collection.add(
            documents=documents,
            ids=ids,
        )

        ollama_client = Client(host="http://localhost:11434")

        planning_agent = PlanningAgent(
            client=ollama_client,
            model=AI_MODEL,
        )

        detail_planning_agent = DetailPlanningAgent(
            client=ollama_client,
            collection=tool_collection,
            model=AI_MODEL,
            server_name_tool_info_dict=server_name_tool_info_dict,
        )

        query_id = 1
        while True:
            user_input = input("\nQuery: ")

            if user_input.strip().lower() in {"exit", "quit"}:
                break

            json_text = planning_agent.chat(user_input)
            print(json_text)
            data = json.loads(json_text)
            if not data.get("planned"):
                print(data["answer"])
                continue

            planned = data.get("planned")
            detail_plan_query = user_input + "\n" + planned

            response = detail_planning_agent.chat(detail_plan_query)

            planning_contents_str = response["message"]["content"]
            print(planning_contents_str)
            planning_contents = json.loads(planning_contents_str)
            answer = planning_contents["answer"]
            print(answer)

            task_result_memory = {}

            for task in planning_contents["tasks"]:
                inputs_from_task = task["inputs_from_task"]
                task_id = task["id"]
                task_results = []
                for pre_task_id in inputs_from_task:
                    task_result = task_result_memory[pre_task_id]
                    task_results.append(task_result)

                tool_query = task["action"]

                tool_result = tool_collection.query(
                    query_texts=[tool_query],
                    n_results=5,
                )

                ids = tool_result["ids"][0]
                possible_tools = []
                for id_ in ids:
                    server_name, tool_name = id_.split(":")
                    tool_info = server_name_tool_info_dict[server_name][tool_name]
                    tool_description = tool_info["description"]
                    tool_input_schema = tool_info["input_schema"]

                    possible_tools.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "tool_description": tool_description,
                            "tool_input_schema": tool_input_schema,
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

                tool_call = choose_mcp_tool(
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

                if chosen_server_name not in list(server_name_tool_info_dict.keys()):
                    print(f"Invalid server call: {tool_call}")
                    continue

                chosen_tool_name = tool_call["tool_name"]

                if (
                    chosen_tool_name
                    not in server_name_tool_info_dict[chosen_server_name].keys()
                ):
                    print(f"Invalid tool call: {tool_call}")

                print(f"Calling tool call: {tool_call}")

                result = await session_dict[chosen_server_name].call_tool(
                    chosen_tool_name, tool_call.get("tool_args")
                )
                tool_result = result.content[0].text
                print("tool result:\n", tool_result)
                task_result_memory[task_id] = tool_result

                user_context_document = (
                    f"- Question: {tool_query}\n- Answer: {tool_result}"
                )
                user_context_collection.add(
                    documents=[user_context_document],
                    ids=[str(query_id)],
                )
                query_id += 1


if __name__ == "__main__":
    asyncio.run(main())
