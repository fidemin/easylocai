import asyncio
import json
import os
from contextlib import AsyncExitStack

from mcp import StdioServerParameters, stdio_client, ClientSession


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

        allowed_response = await session.call_tool("list_allowed_directories")
        allowed_text = allowed_response.content[0].text

        directories = [
            line.strip() for line in allowed_text.split("\n") if line.strip()
        ][1:]
        if not directories:
            directories = ["."]

        print(f"allowed directories: {', '.join(directories)}")

        text_file_extensions = ("txt", "md", "py", "json", "csv", "log")

        text_file_names = []
        for directory in directories:
            print(f"list files in {directory}")

            dir_response = await session.call_tool(
                "list_directory", {"path": directory}
            )
            dir_text = dir_response.content[0].text
            print(dir_text)

            file_txt_list = dir_text.split("\n")

            for file_txt in file_txt_list:
                # file_txt:
                # [DIR] .git
                # [FILE] README.md
                text_filename = file_txt.split(" ", 1)[1]
                extension = text_filename.split(".")[-1]

                if extension in text_file_extensions:
                    text_file_names.append(text_filename)

            print(f"text file names: {text_file_names}")

        for text_filename in text_file_names:
            file_path = os.path.join(directory, text_filename)
            file_response = await session.call_tool("read_file", {"path": file_path})
            content = file_response.content[0].text
            lines = content.split("\n")

            print(f"\n> {text_filename}:")
            print("\n".join(lines[: min(len(lines), 5)]))


if __name__ == "__main__":
    asyncio.run(main())
