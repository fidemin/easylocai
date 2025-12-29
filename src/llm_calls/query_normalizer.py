from pydantic import BaseModel

from src.core.llm_call import LLMCall


class Conversation(BaseModel):
    user_query: str
    answer: str


class QueryNormalizerInput(BaseModel):
    user_query: str
    previous_conversations: list[Conversation] = []


class QueryNormalizerOutput(BaseModel):
    user_query: str
    context: str | None = None


class QueryNormalizer(LLMCall[QueryNormalizerInput, QueryNormalizerOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = (
            "resources/prompts/v2/query_normalizer_system_prompt.jinja2"
        )
        user_prompt_path = "resources/prompts/v2/query_normalizer_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=QueryNormalizerOutput,
            options=options,
        )
