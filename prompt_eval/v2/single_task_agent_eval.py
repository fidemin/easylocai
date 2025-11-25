import asyncio
import json
import logging.config

from ollama import AsyncClient

from src.agents.single_task_agent import SingleTaskAgent
from src.utlis.loggers.default_dict import default_logging_config

logging.config.dictConfig(default_logging_config)


def print_evals(
    input_dict,
    response,
    eval,
):
    print("-------------------------")
    print(f"Input Dict:\n{input_dict}")
    print(f"Response:\n{response}")
    print(f"Eval:\n{eval}")
    print()


async def process(agent, replan_eval_agent, input_dict):
    response = await agent.run(**input_dict)
    if replan_eval_agent is not None:
        eval = await replan_eval_agent.run(**input_dict, **response)
    else:
        eval = None
    print_evals(
        input_dict,
        response,
        eval,
    )


async def main():
    with open("./resources/prompt_eval/single_task_agent_inputs.json") as f:
        input_dicts = json.load(f)

    ollama_client = AsyncClient(host="http://localhost:11434")
    replan_agent = SingleTaskAgent(client=ollama_client)
    # replan_eval_agent = ReplanEvalAgent(client=ollama_client)

    tasks = []
    for input_dict in input_dicts:
        task = asyncio.create_task(process(replan_agent, None, input_dict))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed for input_dict:")
            print(input_dicts[i])
            print("Exception:", repr(result))


if __name__ == "__main__":
    asyncio.run(main())
