import asyncio
import json

from ollama import Client

from src.plannings.agent import NextPlanAgent


async def main():
    ollama_client = Client(host="http://localhost:11434")
    next_plan_agent = NextPlanAgent(
        client=ollama_client,
        model="gpt-oss:20b",
    )

    while True:
        user_input = input("\nQuery: ")

        if user_input.strip().lower() in {"exit", "quit"}:
            break

        next_plan_query = {
            "original_user_query": user_input,
            "previous_task_results": [],
        }

        while True:
            response = next_plan_agent.chat(next_plan_query)
            print(response)

            data = json.loads(response)
            if data["direct_answer"]:
                print(data["direct_answer"])
                break

            if data["continue"] is False:
                # TODO: Make answer result based on task results
                break

            next_plan = data["next_plan"].strip()

            temp_next_task = input("\nNext Task: ")
            temp_next_result = input("\nNext Result: ")

            next_plan_query["previous_task_results"].append(
                {
                    "task": temp_next_task,
                    "result": temp_next_result,
                }
            )


if __name__ == "__main__":
    asyncio.run(main())
