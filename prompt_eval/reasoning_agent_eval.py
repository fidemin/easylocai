import asyncio

from ollama import AsyncClient

from easylocai.agents.reasoning_agent import ReasoningAgent
from prompt_eval.eval_workflow import EvalWorkflow

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
