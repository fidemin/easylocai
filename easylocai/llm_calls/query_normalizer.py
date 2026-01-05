from pydantic import BaseModel

from easylocai.core.llm_call import LLMCall
from easylocai.schemas.common import UserConversation


class QueryNormalizerInput(BaseModel):
    user_query: str
    previous_conversations: list[UserConversation] = []


class QueryNormalizerOutput(BaseModel):
    user_query: str
    user_context: str | None


class QueryNormalizer(LLMCall[QueryNormalizerInput, QueryNormalizerOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/v2/query_normalizer_system_prompt.jinja2"
        user_prompt_path = "prompts/v2/query_normalizer_user_prompt.jinja2"
        options = {
            "temperature": 0.1,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=QueryNormalizerOutput,
            options=options,
        )
