# app/agents/web_search_graph/state.py
from typing import List, Optional, TypedDict
from langchain_core.messages import BaseMessage # type: ignore
from app.schemas.common import SearchResult # Assuming SearchResult is in app.schemas.common

class WebSearchState(TypedDict):
    messages: List[BaseMessage] # Input: chat history from the main graph
    query: Optional[str]        # Output of queryGenerator, input to search_node
    webSearchResults: Optional[List[SearchResult]] # Output of search_node, to be passed back to main graph
    shouldSearch: Optional[bool] # Output of classifyMessage, used for conditional routing within this graph
