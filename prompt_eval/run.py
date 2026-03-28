"""
Run prompt eval for a given config file.

Usage:
    python -m prompt_eval.run <config_file>

Example:
    python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json
"""
import asyncio
import importlib
import json
import sys

from prompt_eval.prompt_eval_workflow import PromptEvalWorkflow

_DEFAULT_MODEL_INFO = {
    "host": "http://localhost:11434",
    "model": "gpt-oss:20b",
    "options": {"temperature": 0.2},
}


def _load_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


async def run(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    output_model = _load_class(config["output_model"]) if config.get("output_model") else None
    model_info = {**_DEFAULT_MODEL_INFO, **config.get("model_info", {})}

    workflow = PromptEvalWorkflow(
        input_file_path=config["input_file"],
        prompt_path_info=config["prompt_info"],
        model_info=model_info,
        output_model=output_model,
    )
    await workflow.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m prompt_eval.run <config_file>")
        print("Example: python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json")
        sys.exit(1)

    asyncio.run(run(sys.argv[1]))
