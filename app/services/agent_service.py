# app/services/agent_service.py
import os
import json # For robust metadata handling
from typing import Dict, Any, Optional, AsyncIterator, List
from langgraph.checkpoint.sqlite import SqliteSaver
from app.agents.open_canvas import open_canvas_graph_app, OpenCanvasState # Import the compiled app and state type
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from app.schemas.agent import AgentInvokeRequest # For type hinting if directly used
from app.config.settings import settings # Import the settings object

# --- Checkpointer Setup ---
SQLITE_PATH = settings.LANGGRAPH_SQLITE_PATH
print(f"Agent service: Using SQLite checkpointer at {SQLITE_PATH}")

try:
    memory_saver = SqliteSaver.from_conn_string(SQLITE_PATH)
    open_canvas_app_with_checkpoint = open_canvas_graph_app.with_checkpointer(memory_saver)
    print(f"Agent service: OpenCanvas graph recompiled with SQLite checkpointer.")
except Exception as e:
    print(f"Error initializing SQLite checkpointer or recompiling graph: {e}")
    print("Agent service: Falling back to graph without checkpointer. State will be ephemeral.")
    open_canvas_app_with_checkpoint = open_canvas_graph_app


class AgentService:
    def __init__(self, app_with_checkpoint = open_canvas_app_with_checkpoint):
        self.graph_app = app_with_checkpoint

    async def invoke_agent_stream(
        self,
        graph_input_payload: Dict[str, Any], # Contains 'messages' and other direct graph inputs from API layer
        assistant_id: str,
        thread_id: str,
        request_config_extras: Optional[Dict[str, Any]] = None # From payload.config.configurable in the original request
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Invokes the agent and streams events.
        graph_input_payload: Data that directly feeds into the graph's OpenCanvasState input (e.g., {"messages": ...}).
                             It may also temporarily contain 'modelConfig' and 'systemPrompt' from the API layer.
        assistant_id, thread_id: Core IDs for configuring the LangGraph run.
        request_config_extras: Additional items from the original request's
                               payload.config.configurable (e.g., supabase_user_id, customModelName).
        """

        final_configurable: Dict[str, Any] = {
            "assistant_id": assistant_id,
            "thread_id": thread_id,
        }

        # Pop modelConfig and systemPrompt from graph_input_payload if they exist,
        # as they are part of the run configuration, not direct graph state inputs.
        if "modelConfig" in graph_input_payload:
            final_configurable["modelConfig"] = graph_input_payload.pop("modelConfig")
        if "systemPrompt" in graph_input_payload:
            final_configurable["systemPrompt"] = graph_input_payload.pop("systemPrompt")

        # Merge items from request_config_extras.
        # This is where supabase_user_id, supabase_session, and potentially customModelName
        # (if not part of modelConfig dict) would live.
        if request_config_extras:
            # If customModelName is in request_config_extras, ensure it's preferred or merged into modelConfig
            if "customModelName" in request_config_extras:
                final_configurable["customModelName"] = request_config_extras.get("customModelName")

            for key, value in request_config_extras.items():
                # Allow overriding assistant_id/thread_id if they were defaults and request_config_extras has specifics.
                if key == "assistant_id" and final_configurable.get(key) == "default_assistant" and value:
                    final_configurable[key] = value
                elif key == "thread_id" and value: # Client-provided thread_id to resume
                    final_configurable[key] = value
                # Add other keys like supabase_user_id, supabase_session, etc.
                # Avoid overwriting already processed keys like modelConfig from top-level of graph_input_payload,
                # unless a more specific handling strategy is needed.
                elif key not in final_configurable:
                    final_configurable[key] = value

        run_config = {"configurable": final_configurable}

        # graph_input_payload should now only contain actual graph inputs like 'messages', 'customQuickActionId'.
        print(f"AgentService invoking graph with final graph_input_payload: {graph_input_payload}, and run_config: {run_config}")
        async for event in self.graph_app.astream_events(graph_input_payload, config=run_config, version="v2"):
            yield event

    async def get_conversation_history(self, assistant_id: str, thread_id: str) -> List[BaseMessage]:
        config: Dict[str, Any] = {"configurable": {"assistant_id": assistant_id, "thread_id": thread_id}}
        state_snapshot = await self.graph_app.get_state(config)
        if state_snapshot:
            return state_snapshot.values.get("messages", [])
        return []

    async def get_full_state(self, assistant_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
        config: Dict[str, Any] = {"configurable": {"assistant_id": assistant_id, "thread_id": thread_id}}
        state_history = await self.graph_app.get_state_history(config, limit=1)
        if state_history:
            return state_history[0].values
        return None

    async def get_run_status(self, assistant_id: str, thread_id: str, run_id: str) -> Dict[str, Any]:
        latest_state_dict = await self.get_full_state(assistant_id, thread_id)
        if latest_state_dict:
            status = "completed"
            last_message = None
            if latest_state_dict.get("messages"):
                messages_list = latest_state_dict["messages"]
                if isinstance(messages_list, list) and messages_list:
                    last_message_content = messages_list[-1].content if hasattr(messages_list[-1], 'content') else str(messages_list[-1])
                    last_message = {"type": messages_list[-1].type if hasattr(messages_list[-1], 'type') else "unknown", "content": str(last_message_content)[:100] + "..."}

            current_artifact_state = latest_state_dict.get("artifact")
            artifact_index = None
            if isinstance(current_artifact_state, dict):
                artifact_index = current_artifact_state.get("currentIndex")
            elif hasattr(current_artifact_state, 'currentIndex'):
                artifact_index = current_artifact_state.currentIndex # type: ignore

            return {
                "run_id": run_id,
                "thread_id": thread_id,
                "status": status,
                "latest_artifact_index": artifact_index,
                "last_message_summary": last_message,
            }
        return {"run_id": run_id, "thread_id": thread_id, "status": "not_found"}

# Example usage (commented out)
# ... (main async test function from previous version) ...
