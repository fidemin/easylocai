"""
Run prompt eval for a given config file.

Usage:
    python -m prompt_eval.run <config_file> [--output <path>] [--format text|json]

Example:
    python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json
    python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json --output results.md
    python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json --output results.json --format json
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


async def run(config_path: str, output_file: str | None = None, output_format: str = "text"):
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
    await workflow.run(config_path=config_path, output_file=output_file, output_format=output_format)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m prompt_eval.run <config_file> [--output <path>] [--format text|json]")
        print("Example: python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json")
        print("Example: python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json --output results.md")
        print("Example: python -m prompt_eval.run resources/prompt_eval/configs/plan_prompt_v2_config.json --output results.json --format json")
        sys.exit(1)

    _output_file = None
    _output_format = "text"

    args = sys.argv[1:]
    config_arg = args[0]

    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            _output_file = args[idx + 1]

    if "--format" in args:
        idx = args.index("--format")
        if idx + 1 < len(args):
            fmt = args[idx + 1]
            if fmt not in ("text", "json"):
                print(f"Error: --format must be 'text' or 'json', got '{fmt}'")
                sys.exit(1)
            _output_format = fmt

    asyncio.run(run(config_arg, output_file=_output_file, output_format=_output_format))
