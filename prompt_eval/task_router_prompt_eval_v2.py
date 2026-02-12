import asyncio

from easylocai.llm_calls.task_router import TaskRouterOutputV2
from prompt_eval.prompt_eval_workflow import PromptEvalWorkflow

if __name__ == "__main__":
    input_file_path = "resources/prompt_eval/task_router_prompt_inputs_v2.json"
    prompt_info = {
        "system": "resources/prompts/task_router_system_prompt_v2.jinja2",
        "user": "resources/prompts/task_router_user_prompt_v2.jinja2",
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
        output_model=TaskRouterOutputV2,
    )
    asyncio.run(workflow.run())
