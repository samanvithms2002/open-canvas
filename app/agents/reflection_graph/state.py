# app/agents/reflection_graph/state.py
from typing import List, Optional, TypedDict
from langchain_core.messages import BaseMessage # type: ignore
from app.schemas.common import ArtifactV3

class ReflectionGraphState(TypedDict):
    messages: List[BaseMessage]       # The conversation history
    artifact: Optional[ArtifactV3]    # The current artifact, if any
    # The 'open_canvas_assistant_id' will be passed via config.configurable,
    # so it doesn't need to be an explicit field in the graph's state definition itself,
    # unless intermediate nodes within this graph need to pass it to each other via state.
    # For a single-node graph like this, config is sufficient.
