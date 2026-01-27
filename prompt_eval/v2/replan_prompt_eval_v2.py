import asyncio

from easylocai.llm_calls.replanner import ReplannerV2Output
from prompt_eval.prompt_eval_workflow import PromptEvalWorkflow

if __name__ == "__main__":
    input_file_path = "resources/prompt_eval/replan_prompt_inputs.json"
    prompt_info = {
        "system": "resources/prompts/v2/replanner_system_prompt.jinja2",
        "user": "resources/prompts/v2/replanner_user_prompt.jinja2",
    }
    model_info = {
        "host": "http://localhost:11434",
        "model": "gpt-oss:20b",
        "options": {"temperature": 0.2},
    }
    workflow = PromptEvalWorkflow(
        input_file_path=input_file_path,
        prompt_path_info=prompt_info,
        model_info=model_info,
        output_model=ReplannerV2Output,
    )
    asyncio.run(workflow.run())
