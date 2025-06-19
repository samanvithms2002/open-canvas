# app/schemas/agent.py
from typing import Any, List, Dict, Optional
from pydantic import BaseModel

class Configurable(BaseModel):
    supabase_session: Optional[Any] = None
    supabase_user_id: Optional[str] = None
    thread_id: Optional[str] = None
    # Add other potential configurable fields based on LangGraph examples

class AgentInvokeRequest(BaseModel):
    input: Any
    config: Optional[Dict[str, Any]] = None # To hold {'configurable': Configurable.dict()}

class AgentInvokeResponse(BaseModel):
    output: Any
    run_id: str
    # Potentially other fields like intermediate steps if streamed

class AgentStatus(BaseModel):
    run_id: str
    status: str # e.g., "running", "completed", "failed"
    output: Optional[Any] = None
    # Add other relevant status fields

class Message(BaseModel):
    type: str # "human", "ai", "system", "tool"
    content: Any
    # Add other message fields like 'id', 'timestamp'

class AgentHistory(BaseModel):
    workspace: str
    history: List[Message]

class ToolDefinition(BaseModel):
    name: str
    description: str
    # Parameters schema if available

class AgentTools(BaseModel):
    tools: List[ToolDefinition]
