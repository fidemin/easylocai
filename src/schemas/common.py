from pydantic import BaseModel


class UserConversation(BaseModel):
    user_query: str
    assistant_answer: str
