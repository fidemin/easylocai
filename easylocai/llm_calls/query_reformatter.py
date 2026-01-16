from pydantic import BaseModel, Field

from easylocai.core.llm_call import LLMCallV2
from easylocai.schemas.common import UserConversation


class QueryReformatterInput(BaseModel):
    user_query: str = Field(
        title="User Query", description="The user's original query."
    )
    previous_conversations: list[UserConversation] = Field(
        default=[],
        title="Previous Conversations",
        description="A list of previous user conversations to provide context.",
    )


class QueryReformatterOutput(BaseModel):
    reformed_query: str = Field(
        title="Reformed Query",
        description="The reformed version of the user query.",
    )
    query_context: str | None = Field(
        title="Query Context",
        description="Relevant context extracted from user query and previous conversations.",
    )


class QueryReformatter(LLMCallV2[QueryReformatterInput, QueryReformatterOutput]):
    def __init__(self, *, client):
        model = "gpt-oss:20b"
        system_prompt_path = "prompts/v2/query_reformatter_system_prompt.jinja2"
        user_prompt_path = "prompts/v2/query_reformatter_user_prompt.jinja2"
        options = {
            "temperature": 0.2,
        }

        super().__init__(
            client=client,
            model=model,
            system_prompt_path=system_prompt_path,
            user_prompt_path=user_prompt_path,
            output_model=QueryReformatterOutput,
            options=options,
        )
