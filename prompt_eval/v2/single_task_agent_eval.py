import asyncio
import logging.config

from ollama import AsyncClient

from prompt_eval.v2.eval_workflow import EvalWorkflow
from src.agents.single_task_agent import SingleTaskAgent
from src.utlis.loggers.default_dict import default_logging_config

logging.config.dictConfig(default_logging_config)


if __name__ == "__main__":
    ollama_client = AsyncClient(host="http://localhost:11434")
    plan_agent = SingleTaskAgent(client=ollama_client)
    plan_eval_agent = None
    workflow = EvalWorkflow(
        agent=plan_agent,
        eval_agent=plan_eval_agent,
        input_path="./resources/prompt_eval/single_task_agent_inputs.json",
    )
    asyncio.run(workflow.run())
