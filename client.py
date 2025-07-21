import asyncio
import json
import logging
from contextlib import AsyncExitStack

import chromadb
from mcp import StdioServerParameters, stdio_client, ClientSession
from ollama import Client

logger = logging.getLogger(__name__)


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
    server_name = f"- server_name: {server_name}"
    tool_description_lines.append(server_name)
    tool_name_text = f"    - tool name: {tool_name}"
    tool_description_lines.append(tool_name_text)
    tool_info_text = f"    - tool info: {tool_info}"
    tool_description_lines.append(tool_info_text)

    tool_description = "\n".join(tool_description_lines)
    return tool_description


def choose_mcp_tool(
    client,
    user_query,
    user_context_list,
    mcp_tools_dict,
    server_name,
    tool_name,
    document,
):
    # tool_description_lines = []
    # for server_name, tool_info_dict in mcp_tools_dict.items():
    #     this_tool_description = convert_to_tool_txt_data(server_name, tool_info_dict)
    #     tool_description_lines.append(this_tool_description)
    #
    # tool_description = "\n".join(tool_description_lines)

    user_contexts = "\n".join(
        [
            f"- UserInput: {user_context["user_input"]}\n-Answer: {user_context["llm_answer"]}"
            for user_context in user_context_list
        ]
    )

    prompt = f"""
    You are a tool-choosing assistant. A user wants to interact with a file system.
    You have access to the following MCP tools:

    Selected server_name: {server_name}
    Selected tool_name: {tool_name}
    Tool Description:
    {document}

    User query:
    "{user_query}"
    
    Context:
    {user_contexts}

    Return a JSON object with:
    - tool_name: one of the tool_name above based on description
    - tool_args: any parameters needed (or empty if none) based on input_schema
    - server_name: one of the server_name above based on tool_name you choose

    Respond ONLY with the JSON object. (Not ``` included)
    """
    print(prompt)

    response = client.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
    )

    content = response["message"]["content"]

    print(f"content: {content}")

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
    with open("mcp_server_config.json") as f:
        servers = json.load(f)["mcpServers"]

        server_params_dict = {}
        for server_name, config in servers.items():
            server_params = StdioServerParameters(
                command=config["command"], args=config["args"], env=None
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

        user_context_list = []

        while True:
            user_input = input("\nQuery: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            tool_result = tool_collection.query(
                query_texts=[user_input],
                n_results=1,
            )

            id_ = tool_result["ids"][0][0]
            server_name, tool_name = id_.split(":")
            document = tool_result["documents"][0][0]
            tool_call = choose_mcp_tool(
                ollama_client,
                user_input,
                user_context_list,
                server_name_tool_info_dict,
                server_name,
                tool_name,
                document,
            )

            if tool_call is None:
                print("invalid tool call. please try again")

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
            print(result.content[0].text)

            user_context_list.append(
                {"user_input": user_input, "llm_answer": result.content[0].text}
            )


if __name__ == "__main__":
    asyncio.run(main())
