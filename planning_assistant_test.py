import asyncio

from jinja2 import Environment, FileSystemLoader
from ollama import Client


async def main():
    ollama_client = Client(host="http://localhost:11434")

    while True:
        user_input = input("\nQuery: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        env = Environment(loader=FileSystemLoader(""))
        template = env.get_template("resources/prompts/planning_assistant.txt")
        prompt = template.render()

        response = ollama_client.chat(
            model="gpt-oss:20b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ],
        )

        content = response["message"]["content"]

        print(f"content: {content}")


if __name__ == "__main__":
    asyncio.run(main())
