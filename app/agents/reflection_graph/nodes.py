# app/agents/reflection_graph/nodes.py
import os
import json # For safely parsing tool call args if they are strings
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage # type: ignore
from langchain_anthropic import ChatAnthropic

from .state import ReflectionGraphState
from .schemas import GenerateReflectionsToolSchema, ReflectionsData
from .prompts import REFLECT_SYSTEM_PROMPT, REFLECT_USER_PROMPT

from app.utils.langchain_helpers import get_artifact_content
from app.schemas.common import ArtifactType
from app.utils.text_processing import get_string_from_content


# Placeholder for formatReflections from app/utils.ts
# This needs to be robustly ported if the exact XML-like structure is critical.
# For now, a simplified version that produces a string representation.
def format_reflections_placeholder(reflections: Optional[ReflectionsData], only_content: bool = False) -> str:
    if not reflections or (not reflections.styleRules and not reflections.content):
        return "No reflections found." # Or "You have not generated any reflections yet." as in TS

    output_parts = []
    if reflections.styleRules:
        style_str = "Style Guidelines:\n- " + "\n- ".join(reflections.styleRules)
        output_parts.append(style_str)

    if reflections.content:
        content_str = "User Facts/Memories:\n- " + "\n- ".join(reflections.content)
        if only_content: # If only_content is true, return only this part
            return content_str
        output_parts.append(content_str)

    return "\n\n".join(output_parts)


async def reflect_node_internal(state: ReflectionGraphState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Internal Reflection Node (`reflect_node_internal`)")
    if config is None: config = {}
    # graph_config here is the full RunnableConfig passed to this specific graph's invocation

    configurable_params = config.get("configurable", {})
    # The main graph's 'assistant_id' is passed as 'open_canvas_assistant_id' in the config for this sub-graph
    assistant_id = configurable_params.get("open_canvas_assistant_id")
    if not assistant_id:
        # This is a critical configuration error for the reflection graph.
        # Depending on desired behavior, could raise error or log and return empty to not break main flow.
        print("Error: `open_canvas_assistant_id` not found in configurable for reflection node. Cannot proceed.")
        return {} # Stop processing for this node

    # --- Placeholder for Store Interaction: Get existing reflections ---
    # In a real setup:
    # checkpointer = config.get("checkpointer") # LangGraph checkpointer instance
    # if checkpointer:
    #   thread_ts = checkpointer.get_checkpoint_ts(assistant_id, "reflection_thread") # Example: using a specific thread for reflections
    #   if thread_ts:
    #      existing_reflections_state = await checkpointer.aget(thread_ts)
    #      parsed_reflections = ReflectionsData(**existing_reflections_state.get("reflections_data_key", {}))
    # else: # Fallback or error if no checkpointer
    #    parsed_reflections = None

    print(f"Warning: Using placeholder for fetching existing reflections for assistant ID: {assistant_id}. Assuming no prior reflections.")
    existing_reflections: Optional[ReflectionsData] = None
    memories_as_string_for_prompt = format_reflections_placeholder(existing_reflections)
    # --- End Placeholder ---

    tool_def = {
        "name": "generate_reflections",
        "description": "Generate and update the complete list of style guidelines and user content reflections based on the provided conversation and artifact context.",
        "input_schema": GenerateReflectionsToolSchema, # Use Pydantic model directly for input_schema
    }

    # Using ChatAnthropic as specified. Requires ANTHROPIC_API_KEY environment variable.
    # Temperature 0 for consistent reflection generation.
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20240620", # Or other suitable Claude model
        temperature=0,
    ).bind_tools([tool_def], tool_choice="generate_reflections") # Force tool call

    current_artifact_model = get_artifact_content(state.get("artifact"))
    artifact_content_str = "User has not generated an artifact yet." # Default
    if current_artifact_model:
        if current_artifact_model.type == ArtifactType.CODE and hasattr(current_artifact_model, 'code'):
            artifact_content_str = current_artifact_model.code # type: ignore
        elif hasattr(current_artifact_model, 'fullMarkdown'): # TEXT
            artifact_content_str = current_artifact_model.fullMarkdown # type: ignore

    formatted_system_prompt = REFLECT_SYSTEM_PROMPT.format(
        artifact=artifact_content_str,
        reflections=memories_as_string_for_prompt
    )

    # Format conversation history
    conversation_history_str = "\n\n".join(
        [f"<{msg.type}>\n{get_string_from_content(msg.content)}\n</{msg.type}>" for msg in state.get("messages", [])]
    )
    formatted_user_prompt = REFLECT_USER_PROMPT.format(conversation=conversation_history_str)

    llm_result = await model.ainvoke([
        SystemMessage(content=formatted_system_prompt),
        HumanMessage(content=formatted_user_prompt)
    ], {"run_name": f"reflect_node_llm_call_aid_{assistant_id}"})

    tool_args_dict = None
    if llm_result.tool_calls and llm_result.tool_calls[0].get("name") == "generate_reflections":
        raw_args = llm_result.tool_calls[0].get("args")
        if isinstance(raw_args, dict):
            tool_args_dict = raw_args
        elif isinstance(raw_args, str): # Some models might stringify JSON tool args
            try: tool_args_dict = json.loads(raw_args)
            except json.JSONDecodeError:
                print(f"Error: Could not parse stringified tool call arguments: {raw_args}")

    if not tool_args_dict:
        print(f"Reflection tool call 'generate_reflections' failed or returned no valid args. LLM Response: {llm_result.content}")
        return {}

    try:
        new_reflections_from_llm = GenerateReflectionsToolSchema(**tool_args_dict)
    except Exception as e: # Pydantic ValidationError
        print(f"Error parsing 'generate_reflections' tool call arguments: {e}. Args: {tool_args_dict}")
        return {}

    # Store the validated reflections data
    updated_reflections_data_to_store = ReflectionsData(
        styleRules=new_reflections_from_llm.styleRules,
        content=new_reflections_from_llm.content
    )

    # --- Placeholder for Store Interaction: Put new reflections ---
    # In a real setup:
    # if checkpointer:
    #    await checkpointer.aput(assistant_id, "reflection_thread", {"reflections_data_key": updated_reflections_data_to_store.dict()})
    # else:
    #    print("Error: No checkpointer configured to save reflections.")

    print(f"Success: Reflections for assistant ID {assistant_id} would be updated in store to: {updated_reflections_data_to_store.dict()}. (Actual store interaction is placeholder).")
    # --- End Placeholder ---

    # This node's primary job is to update the external store (reflections).
    # It does not modify its own graph's state to be passed to other nodes within ReflectionGraphState.
    return {}
from app.schemas.common import ArtifactType # ensure imported for nodes.py
from app.utils.text_processing import get_string_from_content # ensure imported for nodes.py
