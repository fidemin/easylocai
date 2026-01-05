from pydantic import BaseModel

from easylocai.core.llm_call import LLMCall


class ReasoningInput(BaseModel):
    task: str
    user_context: str | None


class ReasoningOutput(BaseModel):
    reasoning: str
    final: str
    confidence: int


class Reasoning(LLMCall[ReasoningInput, ReasoningOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/v2/reasoning_system_prompt.jinja2"
        user_prompt_path = "prompts/v2/reasoning_user_prompt.jinja2"
        options = {
            "temperature": 0.5,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=ReasoningOutput,
            options=options,
        )
