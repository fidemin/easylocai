import asyncio

from ollama import AsyncClient

from prompt_eval.v2.eval_workflow import EvalWorkflow
from src.agents.reasoning_agent import ReasoningAgent

if __name__ == "__main__":
    ollama_client = AsyncClient(host="http://localhost:11434")
    agent = ReasoningAgent(client=ollama_client)
    eval_agent = None
    workflow = EvalWorkflow(
        agent=agent,
        eval_agent=eval_agent,
        input_path="./resources/prompt_eval/reasoning_prompt_inputs.json",
    )
    asyncio.run(workflow.run())
