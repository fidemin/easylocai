import asyncio
import json

from ollama import AsyncClient

from prompt_eval.v2.eval_workflow import EvalWorkflow
from src.agents.evals.replan_eval_agent import ReplanEvalAgent
from src.agents.replan_agent import ReplanAgent


# def print_evals(
#     original_user_query,
#     original_plan,
#     step_results,
#     response,
#     eval,
# ):
#     print("-------------------------")
#     print(f"User Query: {original_user_query}")
#     print(f"Steps: {original_plan}")
#     print(f"Step Results: {step_results}")
#     print(f"Response: {response}")
#     print(f"Eval: {eval}")
#     print()
#
#
# async def process(agent, replan_eval_agent, input_dict):
#     response = await agent.run(**input_dict)
#     eval = await replan_eval_agent.run(**input_dict, **response)
#     print_evals(
#         input_dict["original_user_query"],
#         input_dict["original_plan"],
#         input_dict["step_results"],
#         response,
#         eval,
#     )
#
#
# async def main():
#     with open("./resources/prompt_eval/replan_agent_inputs.json") as f:
#         input_dicts = json.load(f)
#
#     ollama_client = AsyncClient(host="http://localhost:11434")
#     replan_agent = ReplanAgent(client=ollama_client)
#     replan_eval_agent = ReplanEvalAgent(client=ollama_client)
#
#     tasks = []
#     for input_dict in input_dicts:
#         task = asyncio.create_task(process(replan_agent, replan_eval_agent, input_dict))
#         tasks.append(task)
#
#     results = await asyncio.gather(*tasks, return_exceptions=True)
#
#     for i, result in enumerate(results):
#         if isinstance(result, Exception):
#             print(f"Task {i} failed for input_dict:")
#             print(input_dicts[i])
#             print("Exception:", repr(result))


if __name__ == "__main__":
    ollama_client = AsyncClient(host="http://localhost:11434")
    plan_agent = ReplanAgent(client=ollama_client)
    plan_eval_agent = ReplanEvalAgent(client=ollama_client)
    workflow = EvalWorkflow(
        agent=plan_agent,
        eval_agent=plan_eval_agent,
        input_path="./resources/prompt_eval/replan_agent_inputs.json",
    )
    asyncio.run(workflow.run())
