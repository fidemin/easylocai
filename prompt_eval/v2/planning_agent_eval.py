import asyncio

from ollama import AsyncClient

from prompt_eval.v2.eval_workflow import EvalWorkflow
from src.agents.evals.plan_eval_agent import PlanEvalAgent
from src.agents.plan_agent import PlanAgent

if __name__ == "__main__":
    ollama_client = AsyncClient(host="http://localhost:11434")
    plan_agent = PlanAgent(client=ollama_client)
    plan_eval_agent = PlanEvalAgent(client=ollama_client)
    workflow = EvalWorkflow(
        agent=plan_agent,
        eval_agent=plan_eval_agent,
        input_path="./resources/prompt_eval/plan_agent_inputs.json",
    )
    asyncio.run(workflow.run())
