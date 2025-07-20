import asyncio
import json
from contextlib import AsyncExitStack

from mcp import StdioServerParameters, stdio_client, ClientSession
from ollama import Client


def choose_mcp_tool(client, user_query, user_context_list, mcp_tools_dict):
    tool_description_lines = []
    for server_name, tool_info_dict in mcp_tools_dict.items():
        server_name = f"- Server Name: {server_name}"
        tool_description_lines.append(server_name)

        for tool_name, tool_info in tool_info_dict.items():
            tool_name_text = f"    - Tool Name: {tool_name}"
            tool_description_lines.append(tool_name_text)
            tool_info_text = f"    - Tool Info: {tool_info}"
            tool_description_lines.append(tool_info_text)

    tool_description = "\n".join(tool_description_lines)

    user_contexts = "\n".join(
        [
            f"- UserInput: {user_context["user_input"]}\n-Answer: {user_context["llm_answer"]}"
            for user_context in user_context_list
        ]
    )

    prompt = f"""
    You are a tool-choosing assistant. A user wants to interact with a file system.
    You have access to the following MCP tools:

    Tool Description:
    {tool_description}

    User query:
    "{user_query}"
    
    Context:
    {user_contexts}

    Return a JSON object with:
    - server_name: one of the server name above
    - tool: one of the tool names above
    - args: any parameters needed (or empty if none)

    Respond ONLY with the JSON object.
    """
    print(prompt)

    response = client.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
    )

    content = response["message"]["content"]

    # Extract JSON from response
    try:
        parsed = json.loads(content)
        return parsed
    except json.JSONDecodeError:
        print("❌ LLM did not return valid JSON:", content)
        return {}


async def main():
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
        tool_info_dict = {}
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
            tool_info_dict[server_name] = tool_info

        ollama_client = Client(host="http://localhost:11434")

        user_context_list = []

        while True:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            tool_call = choose_mcp_tool(
                ollama_client, user_input, user_context_list, tool_info_dict
            )

            choosen_server_name = tool_call["server_name"]

            if choosen_server_name not in list(tool_info_dict.keys()):
                print(f"Invalid server call: {tool_call}")
                continue

            choosen_tool_name = tool_call["tool"]

            if choosen_tool_name not in tool_info_dict[choosen_server_name].keys():
                print(f"Invalid tool call: {tool_call}")

            print(f"Calling tool call: {tool_call}")

            result = await session_dict[choosen_server_name].call_tool(
                tool_call["tool"], tool_call["args"]
            )
            print(result.content[0].text)

            user_context_list.append(
                {"user_input": user_input, "llm_answer": result.content[0].text}
            )


if __name__ == "__main__":
    asyncio.run(main())
