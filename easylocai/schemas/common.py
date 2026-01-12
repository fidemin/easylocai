from pydantic import BaseModel, Field


class UserConversation(BaseModel):
    user_query: str = Field(
        title="User Query", description="The user's query in conversation"
    )
    assistant_answer: str = Field(
        title="Assistant Answer", description="The assistant's answer in conversation"
    )
