from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2
from easylocai.schemas.common import UserConversation


class QueryNormalizerInput(BaseModel):
    user_query: str
    previous_conversations: list[UserConversation] = []


class QueryNormalizerOutput(BaseModel):
    user_query: str
    user_context: str | None


class QueryNormalizerInputV2(BaseModel):
    user_query: str = Field(
        title="User Query", description="The user's original query."
    )
    previous_conversations: list[UserConversation] = Field(
        default=[],
        title="Previous Conversations",
        description="A list of previous user conversations to provide context.",
    )


class QueryNormalizerOutputV2(BaseModel):
    reformed_query: str = Field(
        title="Reformed Query",
        description="The reformed version of the user query.",
    )
    query_context: str | None = Field(
        title="Query Context",
        description="Relevant context extracted from user query and previous conversations.",
    )


class QueryNormalizerV2(LLMCallV2[QueryNormalizerInputV2, QueryNormalizerOutputV2]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/v2/query_normalizer_system_prompt_v2.jinja2"
        user_prompt_path = "prompts/v2/query_normalizer_user_prompt_v2.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=QueryNormalizerOutputV2,
            options=options,
        )
