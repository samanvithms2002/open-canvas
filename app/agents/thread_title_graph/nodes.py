# app/agents/thread_title_graph/nodes.py
import os
import json # For parsing tool args if stringified
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage # type: ignore
from langchain_openai import ChatOpenAI

from .state import TitleGenerationState
from .schemas import GenerateTitleToolSchema
from .prompts import TITLE_SYSTEM_PROMPT, TITLE_USER_PROMPT

from app.utils.langchain_helpers import get_artifact_content
from app.schemas.common import ArtifactType # For checking artifact type
from app.utils.text_processing import get_string_from_content # For formatting messages

async def generate_title_internal(state: TitleGenerationState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Internal Title Generation Node (`generate_title_internal`)")
    if config is None: config = {}
    # graph_config is the full RunnableConfig for this sub-graph

    configurable_params = config.get("configurable", {})
    thread_id = configurable_params.get("open_canvas_thread_id") # Passed from main graph's config
    if not thread_id:
        # This is a critical configuration error for the title generation graph.
        print("Error: `open_canvas_thread_id` not found in configurable for title generation node. Cannot proceed.")
        return {} # Stop processing

    tool_def = {
        "name": "generate_title",
        "description": "Generate a concise and descriptive title for the conversation.",
        "input_schema": GenerateTitleToolSchema, # Use Pydantic model directly
    }

    # Using gpt-4o-mini as specified. Requires OPENAI_API_KEY environment variable.
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0, # Temperature 0 for consistent title generation
    ).bind_tools([tool_def], tool_choice="generate_title") # Force tool call

    current_artifact_model = get_artifact_content(state.get("artifact"))
    artifact_context_str = "No artifact was generated during this conversation."
    if current_artifact_model:
        content = ""
        # Ensure attribute access is safe
        if current_artifact_model.type == ArtifactType.CODE and hasattr(current_artifact_model, 'code'):
            content = current_artifact_model.code # type: ignore
        elif hasattr(current_artifact_model, 'fullMarkdown'): # TEXT
            content = current_artifact_model.fullMarkdown # type: ignore

        # Truncate artifact content for the prompt to keep it concise
        max_artifact_context_len = 500
        truncated_content = content[:max_artifact_context_len] + "..." if len(content) > max_artifact_context_len else content
        artifact_context_str = f"An artifact was generated during this conversation (first {max_artifact_context_len} chars):\n\n{truncated_content}"

    # Format conversation history for the prompt
    # The TS version takes up to 5 messages. Let's replicate that for brevity.
    messages_for_prompt = state.get("messages", [])[:5]
    conversation_history_str = "\n\n".join(
        [f"<{msg.type}>\n{get_string_from_content(msg.content)}\n</{msg.type}>" for msg in messages_for_prompt]
    )
    if not conversation_history_str:
        conversation_history_str = "The conversation is empty."

    formatted_user_prompt = TITLE_USER_PROMPT.format(
        conversation=conversation_history_str,
        artifact_context=artifact_context_str
    )

    llm_result = await model.ainvoke([
        SystemMessage(content=TITLE_SYSTEM_PROMPT),
        HumanMessage(content=formatted_user_prompt)
    ], {"run_name": f"generate_title_llm_call_tid_{thread_id}"})

    tool_args_dict = None
    if llm_result.tool_calls and llm_result.tool_calls[0].get("name") == "generate_title":
        raw_args = ll_result.tool_calls[0].get("args")
        if isinstance(raw_args, dict):
            tool_args_dict = raw_args
        elif isinstance(raw_args, str):
            try: tool_args_dict = json.loads(raw_args)
            except json.JSONDecodeError:
                print(f"Error: Could not parse stringified tool call arguments for title: {raw_args}")

    if not tool_args_dict:
        print(f"Title generation tool call 'generate_title' failed or returned no valid args. LLM Response: {llm_result.content}")
        return {}

    try:
        generated_title_data = GenerateTitleToolSchema(**tool_args_dict)
    except Exception as e: # Pydantic ValidationError
        print(f"Error parsing 'generate_title' tool call arguments: {e}. Args: {tool_args_dict}")
        return {}

    # --- Placeholder for langGraphClient.threads.update ---
    # This is where you'd interact with your thread management system.
    # If using LangGraph Cloud (langgraph_sdk), it would be something like:
    # from langgraph_sdk import get_client
    # client = get_client() # Assuming client is configured
    # await client.update_thread(thread_id, metadata={"title": generated_title_data.title})
    #
    # If using a custom DB or another service, the update logic would go here.
    print(f"Success: Title for thread ID '{thread_id}' would be updated to: '{generated_title_data.title}'. (Actual thread metadata update is a placeholder).")
    # --- End Placeholder ---

    # This node's primary job is to update external thread metadata.
    # It does not modify its own graph's state to be passed to other nodes within TitleGenerationState.
    return {}

# Ensure all necessary types are available in this file's scope
from app.schemas.common import ArtifactType
from app.utils.text_processing import get_string_from_content
