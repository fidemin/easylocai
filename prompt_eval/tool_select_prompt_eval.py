import asyncio

from easylocai.llm_calls.tool_selector import ToolSelectorOutput
from prompt_eval.prompt_eval_workflow import PromptEvalWorkflow

if __name__ == "__main__":
    input_file_path = "resources/prompt_eval/tool_selector_prompt_inputs.json"
    prompt_info = {
        "system": "resources/prompts/tool_selector_system_prompt.jinja2",
        "user": "resources/prompts/tool_selector_user_prompt.jinja2",
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
        output_model=ToolSelectorOutput,
    )
    asyncio.run(workflow.run())
