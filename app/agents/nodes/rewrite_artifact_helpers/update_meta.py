# app/agents/nodes/rewrite_artifact_helpers/update_meta.py
from typing import Dict, Any, Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage # type: ignore

from app.agents.state import OpenCanvasState
from app.utils.langchain_helpers import (
    get_model_from_config,
    is_using_o1_mini_model,
    # get_formatted_reflections, # This is a placeholder in open_canvas.py, will use that
    get_artifact_content,
    format_artifact_content, # For formatting artifact for the prompt
    get_string_from_message_content # To extract text from last human message
)
from app.agents.prompts import GET_TITLE_TYPE_REWRITE_ARTIFACT
from .schemas import OptionallyUpdateArtifactMetaSchema
from app.schemas.common import ProgrammingLanguageOptions # For fallback

# Placeholder for get_formatted_reflections, assuming it's defined in open_canvas or similar scope
# For direct use here, it would need to be imported or passed, or we use a local placeholder.
# Using a local placeholder for now to match the structure in open_canvas.py
async def get_formatted_reflections_placeholder(config: Dict[str, Any]) -> str:
    print("Warning: Using placeholder for get_formatted_reflections in update_meta.py.")
    return "No specific user reflections available at this time (placeholder)."


async def optionally_update_artifact_meta(
    state: OpenCanvasState,
    config: Dict[str, Any] # This is the full RunnableConfig
) -> OptionallyUpdateArtifactMetaSchema:
    # Extract the 'configurable' part for helpers
    graph_config = config.get("configurable", {})

    # For structured_output, model needs to support functions/tools or specific structured output prompting
    # get_model_from_config should be returning a compatible model if isToolCalling=True
    # Temperature is set to 0 for this call in TS.
    llm = await get_model_from_config(graph_config, extra={"isToolCalling": True, "temperature": 0.0})

    # Using bind_tools with Pydantic model for structured output
    structured_llm = llm.bind_tools(
        tools=[{
            "name": "optionallyUpdateArtifactMeta", # Tool name
            "description": "Determines new type, title, and language for an artifact based on user request.", # Tool description
            "parameters": OptionallyUpdateArtifactMetaSchema.schema()
        }],
        tool_choice="optionallyUpdateArtifactMeta" # Force this tool
    )

    memories_as_string = await get_formatted_reflections_placeholder(graph_config)

    current_artifact_model = get_artifact_content(state.get("artifact"))
    if not current_artifact_model:
        # This case should ideally be prevented by graph logic (e.g., not calling rewrite if no artifact)
        # However, if it happens, return current state's meta or default.
        print("Error: No artifact found in state for optionallyUpdateArtifactMeta. Returning default meta.")
        return OptionallyUpdateArtifactMetaSchema(
            type=ArtifactType.TEXT, # Default to text
            title="Untitled",
            language=ProgrammingLanguageOptions.OTHER
        )

    # Find last human message for the prompt
    recent_human_message_content_str = "No recent human input found." # Fallback
    for msg in reversed(state.get("_messages", [])): # Check _messages which is conversation history
        if msg.type == "human": # or isinstance(msg, HumanMessage):
            recent_human_message_content_str = get_string_from_message_content(msg.content)
            break

    prompt_str = GET_TITLE_TYPE_REWRITE_ARTIFACT.format(
        recent_human_message_content=recent_human_message_content_str,
        artifact=format_artifact_content(current_artifact_model, shorten_content=True, max_length=2000), # Keep context reasonable
        reflections=memories_as_string
    )

    is_o1 = is_using_o1_mini_model(graph_config)
    prompt_role = "user" if is_o1 else "system" # Main prompt role

    # The TS version only sends the main prompt, not the recent human message separately for this particular call.
    # The human message content is embedded IN the prompt.
    messages_for_llm: List[BaseMessage] = []
    if prompt_role == "system":
        messages_for_llm.append(SystemMessage(content=prompt_str))
    else:
        messages_for_llm.append(HumanMessage(content=prompt_str))

    llm_response = await structured_llm.ainvoke(messages_for_llm, {"run_name": "optionally_update_artifact_meta_llm"})

    response_payload_dict = None
    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        # We forced 'optionallyUpdateArtifactMeta' tool.
        tool_call = ll_response.tool_calls[0]
        if tool_call.get("name") == "optionallyUpdateArtifactMeta":
            response_payload_dict = tool_call.get("args")

    if response_payload_dict:
        try:
            return OptionallyUpdateArtifactMetaSchema(**response_payload_dict)
        except Exception as e: # Pydantic ValidationError
            print(f"Error parsing LLM response for artifact meta: {e}. Payload: {response_payload_dict}")
    else:
        print(f"Warning: Structured output for artifact meta was not as expected or missing. LLM Response: {llm_response}")

    # Fallback: If LLM fails to provide structured output or it's unparsable,
    # return metadata derived from the current artifact. This means no change to title, type, or language.
    current_type = current_artifact_model.type
    current_title = current_artifact_model.title
    current_lang = current_artifact_model.language if hasattr(current_artifact_model, 'language') else ProgrammingLanguageOptions.OTHER

    # If type is TEXT, language should be OTHER.
    if current_type == ArtifactType.TEXT:
        current_lang = ProgrammingLanguageOptions.OTHER
    # If type is CODE and language is None (e.g. from old data or if not set), default to OTHER.
    elif current_type == ArtifactType.CODE and current_lang is None:
        current_lang = ProgrammingLanguageOptions.OTHER

    return OptionallyUpdateArtifactMetaSchema(
        type=current_type,
        title=current_title, # No change to title on failure
        language=current_lang
    )

from app.schemas.common import ArtifactType # For fallback logic
from app.utils.langchain_helpers import ProgrammingLanguageOptions # For fallback
