import asyncio
import json

from ollama import AsyncClient

from src.agents.evals.plan_eval_agent import PlanEvalAgent
from src.agents.plan_agent import PlanAgent


def print_evals(
    user_query,
    steps,
    eval,
):
    print("-------------------------")
    print(f"User Query: {user_query}")
    print(f"Steps: {steps}")
    print(f"eval: {eval}")
    print()


async def process(plan_agent, plan_eval_agent, input_dict):
    response = await plan_agent.run(**input_dict)

    eval_result = await plan_eval_agent.run(
        user_query=input_dict["user_query"], steps=response["steps"]
    )
    print_evals(input_dict["user_query"], response["steps"], eval_result)


async def main():
    input_dicts = []
    with open("./resources/prompt_eval/plan_agent_inputs_reasoning.jsonl") as f:
        for row in f:
            row = row.strip()
            input_dicts.append(json.loads(row))

    ollama_client = AsyncClient(host="http://localhost:11434")
    plan_agent = PlanAgent(client=ollama_client)
    plan_eval_agent = PlanEvalAgent(client=ollama_client)

    tasks = []
    for input_dict in input_dicts:
        task = asyncio.create_task(process(plan_agent, plan_eval_agent, input_dict))
        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
