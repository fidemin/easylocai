import asyncio
import json
import os
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


async def main():
    with open("mcp_server_config.json") as f:
        config = json.load(f)["mcpServers"]["filesystem"]

    server_params = StdioServerParameters(
        command=config["command"], args=config["args"], env=None
    )

    stack = AsyncExitStack()
    async with stack:
        stdio, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        tools_response = await session.list_tools()
        tool_names = [tool.name for tool in tools_response.tools]

        print(f"tools: {tool_names}")

        ollama_client = Client(host="http://localhost:11434")

        while True:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                break

            extensions = get_requested_extensions(ollama_client, user_input)
            if not extensions:
                print("🤖 No valid file types found in input.")
                continue

            matching_files = await list_files(session, extensions)
            if matching_files:
                print("📄 Matching files:")
                for f in matching_files:
                    print("  -", f)
            else:
                print("❌ No matching files found.")


if __name__ == "__main__":
    asyncio.run(main())
