# app/agents/web_search_graph/schemas.py
from langchain_core.pydantic_v1 import BaseModel, Field

class ClassificationSchema(BaseModel):
    should_search: bool = Field(..., description="Whether or not to search the web based on the user's latest message.")
