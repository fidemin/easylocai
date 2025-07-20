import asyncio
import json
from contextlib import AsyncExitStack

from mcp import StdioServerParameters, stdio_client, ClientSession
from ollama import Client


def get_requested_extensions(ollama_client, user_input: str) -> set[str]:
    system_prompt = (
        "Extract file extensions like 'md', 'txt', 'json', etc. from user input. "
        "Only output a space-separated list of file extensions. No explanation. "
        "If none found, return empty string."
    )

    response = ollama_client.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    content = response["message"]["content"]
    return set(content.strip().split())


async def get_allowed_directories(session):
    allowed_response = await session.call_tool("list_allowed_directories")
    allowed_text = allowed_response.content[0].text

    directories = [line.strip() for line in allowed_text.split("\n") if line.strip()][
        1:
    ]

    return directories


async def list_files(session, extensions: set[str]) -> list[str]:
    directories = await get_allowed_directories(session)
    matches = []

    for directory in directories:
        response = await session.call_tool("list_directory", {"path": directory})
        lines = response.content[0].text.splitlines()

        for line in lines:
            if line.startswith("[FILE] "):
                filename = line.split(" ", 1)[1]
                ext = filename.rsplit(".", 1)[-1]
                if ext in extensions:
                    matches.append(f"{directory}/{filename}")
    return matches


MCP_TOOLS = {
    # "list_allowed_directories": "Lists directories the user can browse.",
    # "list_directory": "Lists files and directories at a given path. Requires: {'path': <str>}",
    # "read_file": "Reads contents of a file. Requires: {'path': <str>}",
}


def choose_mcp_tool(client, user_query, user_context_list):
    tool_description = "\n".join(
        f"- {name}: {info}" for name, info in MCP_TOOLS.items()
    )

    user_contexts = "\n".join(
        [
            f"- UserInput: {user_context["user_input"]}\n-Answer: {user_context["llm_answer"]}"
            for user_context in user_context_list
        ]
    )

    prompt = f"""
    You are a tool-choosing assistant. A user wants to interact with a file system.
    You have access to the following MCP tools:

    {tool_description}

    User query:
    "{user_query}"
    
    Context:
    {user_contexts}

    Return a JSON object with:
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
        config = servers["filesystem"]
        config1 = servers["custom"]

    server_params = StdioServerParameters(
        command=config["command"], args=config["args"], env=None
    )

    server_params2 = StdioServerParameters(
        command=config1["command"], args=config1["args"], env=None
    )

    stack = AsyncExitStack()
    async with stack:
        stdio, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        stdio2, write2 = await stack.enter_async_context(stdio_client(server_params2))
        session2 = await stack.enter_async_context(ClientSession(stdio2, write2))
        await session2.initialize()

        tools_response = await session.list_tools()
        print(f"Available tools: {[tool.name for tool in tools_response.tools]}")

        tool_response2 = await session2.list_tools()
        print(f"Available tools1: {[tool.name for tool in tool_response2.tools]}")

        for tool in tools_response.tools:
            MCP_TOOLS[tool.name] = {
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }

        for tool in tool_response2.tools:
            print(tool.name, tool.description, tool.inputSchema)

        ollama_client = Client(host="http://localhost:11434")

        user_context_list = []

        while True:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            tool_call = choose_mcp_tool(ollama_client, user_input, user_context_list)
            if tool_call["tool"] not in list(MCP_TOOLS.keys()):
                print(f"Invalid tool call: {tool_call}")
                continue
            else:
                print(f"Calling tool call: {tool_call}")

            result = await session.call_tool(tool_call["tool"], tool_call["args"])
            print(result.content[0].text)

            user_context_list.append(
                {"user_input": user_input, "llm_answer": result.content[0].text}
            )


if __name__ == "__main__":
    asyncio.run(main())
