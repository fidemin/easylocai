import asyncio

from ollama import AsyncClient

from src.agents.plan_agent import PlanAgent


async def main():
    ollama_client = AsyncClient(host="http://localhost:11434")
    agent = PlanAgent(client=ollama_client, model="gpt-oss:20b")
    response = await agent.run(user_query="Where is Seoul?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
