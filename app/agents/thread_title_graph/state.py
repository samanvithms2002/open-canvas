# app/agents/thread_title_graph/state.py
from typing import List, Optional, TypedDict
from langchain_core.messages import BaseMessage # type: ignore
from app.schemas.common import ArtifactV3

class TitleGenerationState(TypedDict):
    messages: List[BaseMessage]    # Conversation history for title generation
    artifact: Optional[ArtifactV3] # Artifact context, if any
    # The 'open_canvas_thread_id' will be passed via config.configurable for this graph,
    # similar to how 'open_canvas_assistant_id' is handled for the reflection graph.
    # It's used by the node to know which thread's metadata to update (placeholder for now).
