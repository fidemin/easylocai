import asyncio

from ollama import AsyncClient

from prompt_eval.v2.eval_workflow import EvalWorkflow
from src.agents.evals.replan_eval_agent import ReplanEvalAgent
from src.agents.replan_agent import ReplanAgent

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
