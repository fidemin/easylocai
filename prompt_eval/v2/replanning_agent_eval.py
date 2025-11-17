import asyncio
import json

from ollama import AsyncClient

from src.agents.replan_agent import ReplanAgent


def print_evals(
    original_user_query,
    original_plan,
    step_results,
    response,
):
    print("-------------------------")
    print(f"User Query: {original_user_query}")
    print(f"Steps: {original_plan}")
    print(f"Step Results: {step_results}")
    print(f"response: {response}")
    print()


async def process(agent, plan_eval_agent, input_dict):
    response = await agent.run(**input_dict)
    print_evals(
        input_dict["original_user_query"],
        input_dict["original_plan"],
        input_dict["step_results"],
        response,
    )


async def main():
    with open("./resources/prompt_eval/replan_agent_inputs.json") as f:
        input_dicts = json.load(f)

    ollama_client = AsyncClient(host="http://localhost:11434")
    replan_agent = ReplanAgent(client=ollama_client)
    # plan_eval_agent = PlanEvalAgent(client=ollama_client)

    tasks = []
    for input_dict in input_dicts:
        task = asyncio.create_task(process(replan_agent, None, input_dict))
        tasks.append(task)

    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
