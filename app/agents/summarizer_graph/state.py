# app/agents/summarizer_graph/state.py
from typing import List, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage # type: ignore

class SummarizerGraphState(TypedDict):
    messages_in: List[BaseMessage] # Input messages to be summarized
    summarized_message_out: Optional[HumanMessage] # Output of the summarization
    # thread_id is not part of this state as the graph itself is stateless regarding threads;
    # thread context would be managed by how it's called if it needed to update thread-specific metadata.
