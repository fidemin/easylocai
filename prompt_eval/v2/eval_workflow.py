import asyncio
import json
import logging.config

from src.core.agent import Agent
from src.utlis.loggers.default_dict import default_logging_config

logging.config.dictConfig(default_logging_config)


class EvalWorkflow:
    def __init__(
        self,
        *,
        agent: Agent,
        input_path: str,
        eval_agent=None,
    ):
        self._agent = agent
        self._eval_agent = eval_agent
        self._input_path = input_path

    async def run(self):
        with open(self._input_path) as f:
            input_dicts = json.load(f)

        tasks = []
        for input_dict in input_dicts:
            task = asyncio.create_task(self._process(input_dict))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task {i} failed for input_dict:")
                print(input_dicts[i])
                print("Exception:", repr(result))

    async def _process(self, input_dict):
        response = await self._agent.run(**input_dict)
        if self._eval_agent is not None:
            eval = await self._eval_agent.run(**input_dict, **response)
        else:
            eval = None
        self._print_evals(
            input_dict,
            response,
            eval,
        )

    def _print_evals(
        self,
        input_dict,
        response,
        eval,
    ):
        print("-------------------------")
        print(f"Input Dict:")
        print(json.dumps(input_dict, indent=2, ensure_ascii=False))
        print(f"Response:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print(f"Eval:\n{eval}")
        print()
