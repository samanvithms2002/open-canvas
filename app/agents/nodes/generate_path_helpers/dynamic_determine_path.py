# app/agents/nodes/generate_path_helpers/dynamic_determine_path.py
import json # For parsing model output if it's a string
from typing import List, Optional, Dict, Any, Literal as PyLiteral
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage # type: ignore
from langchain_core.pydantic_v1 import BaseModel, Field

from app.agents.prompts import (
    ROUTE_QUERY_PROMPT,
    ROUTE_QUERY_OPTIONS_HAS_ARTIFACTS,
    ROUTE_QUERY_OPTIONS_NO_ARTIFACTS,
    CURRENT_ARTIFACT_PROMPT,
    NO_ARTIFACT_PROMPT
)
from app.utils.langchain_helpers import (
    get_model_from_config,
    # create_context_document_messages, # Not directly used here, context is in state.messages
    get_artifact_content,
    format_artifact_content_with_template,
    get_string_from_message_content, # Changed from text_processing import
)
from app.agents.state import OpenCanvasState
# from app.utils.text_processing import get_string_from_content # Moved to langchain_helpers to avoid potential circularity

# from langsmith import traceable # If using langsmith


# @traceable(name="dynamic_determine_path") # If using langsmith
async def dynamic_determine_path_func(
    state: OpenCanvasState,
    # new_messages are messages that might have been transformed by document processing (e.g., PDF text extraction)
    # and are intended to be part of the input to this routing LLM call.
    # If no such transformations happened, new_messages might be empty or None.
    newly_processed_messages: Optional[List[BaseMessage]],
    config: Dict[str, Any], # LangGraphRunnableConfig's 'configurable' field
) -> Dict[str, str]: # Returns a dictionary like {"route": "routeName"}

    current_artifact_model = get_artifact_content(state.get("artifact"))

    artifact_options_prompt_text = ROUTE_QUERY_OPTIONS_HAS_ARTIFACTS if current_artifact_model else ROUTE_QUERY_OPTIONS_NO_ARTIFACTS

    # Get last 3 messages from the main conversation history in state
    # These are distinct from newly_processed_messages which might be hidden from UI
    conversation_history = state.get("messages", [])
    recent_messages_str = "\n\n".join(
        [f"{msg.type}: {get_string_from_message_content(msg.content)}" for msg in conversation_history[-3:]]
    )

    current_artifact_for_prompt_str = ""
    if current_artifact_model:
        # format_artifact_content_with_template expects the template string first
        current_artifact_for_prompt_str = format_artifact_content_with_template(
            CURRENT_ARTIFACT_PROMPT, # This is the template string "{artifact}"
            current_artifact_model,
            shorten_content=True, # Keep it concise for routing prompt
            max_length=1000
        )
    else:
        current_artifact_for_prompt_str = NO_ARTIFACT_PROMPT

    final_route_query_prompt_str = ROUTE_QUERY_PROMPT.format(
        artifactOptions=artifact_options_prompt_text,
        recentMessages=recent_messages_str,
        currentArtifactPrompt=current_artifact_for_prompt_str
    )

    # Define Pydantic schema for the tool call dynamically based on artifact presence
    # This determines the valid 'route' enum values the LLM can choose.
    if current_artifact_model:
        class DynamicRouteQuerySchema(BaseModel):
            route: PyLiteral["replyToGeneralInput", "rewriteArtifact"] = Field(..., description="The route to take based on the user's query.")
        # artifact_route_name_for_schema = "rewriteArtifact" # Not directly used, but good for clarity
    else:
        class DynamicRouteQuerySchema(BaseModel):
            route: PyLiteral["replyToGeneralInput", "generateArtifact"] = Field(..., description="The route to take based on the user's query.")
        # artifact_route_name_for_schema = "generateArtifact"

    tool_definition = {
        "name": "route_query",
        "description": "Select the appropriate route based on the user's query and current context.",
        "parameters": DynamicRouteQuerySchema.schema()
    }

    # Use a copy of the config to potentially modify model selection for this specific call
    llm_call_config = config.copy()
    # Example: if you wanted to use a specific model for routing:
    # llm_call_config["customModelName"] = "gpt-3.5-turbo"
    # if "modelConfig" not in llm_call_config or not llm_call_config["modelConfig"]:
    #    llm_call_config["modelConfig"] = {"provider": "openai"}
    # elif "provider" not in llm_call_config["modelConfig"]:
    #    llm_call_config["modelConfig"]["provider"] = "openai"


    model = await get_model_from_config(llm_call_config, extra={"isToolCalling": True})
    model_with_tool = model.bind_tools(tools=[tool_definition], tool_choice="route_query")

    # Construct messages for the LLM routing call
    llm_messages_for_routing: List[BaseMessage] = []

    # If there were messages processed from documents (e.g. PDF text, URL content),
    # they should be included as context for the routing decision.
    if newly_processed_messages:
        llm_messages_for_routing.extend(newly_processed_messages)

    # Add the main routing prompt itself
    llm_messages_for_routing.append(HumanMessage(content=final_route_query_prompt_str))

    try:
        llm_response = await model_with_tool.ainvoke(llm_messages_for_routing)

        args = {}
        if isinstance(llm_response, AIMessage) and llm_response.tool_calls:
            if llm_response.tool_calls:
                raw_args = llm_response.tool_calls[0].get("args")
                if isinstance(raw_args, str): # Should be dict if Pydantic schema used by Langchain
                    try: args = json.loads(raw_args)
                    except json.JSONDecodeError: print(f"Error parsing tool call args string: {raw_args}")
                elif isinstance(raw_args, dict): args = raw_args
        else:
            print(f"Warning: Routing LLM did not return expected tool call. Response: {llm_response}")

        chosen_route = args.get("route")

        # Validate chosen_route against the dynamically set schema options
        valid_routes = list(DynamicRouteQuerySchema.schema()['properties']['route']['enum'])
        if chosen_route and chosen_route in valid_routes:
            return {"next_node": chosen_route} # Ensure key is 'next_node' for graph state
        else:
            print(f"Warning: LLM returned invalid route '{chosen_route}' or no route. Args: {args}. Valid: {valid_routes}")
            # Fallback logic:
            fallback_route = "rewriteArtifact" if current_artifact_model else "replyToGeneralInput"
            print(f"Defaulting to fallback route: {fallback_route}")
            return {"next_node": fallback_route}

    except Exception as e:
        import traceback
        print(f"Error in dynamic_determine_path_func LLM call: {e}")
        traceback.print_exc()
        # Fallback logic in case of any error during LLM call
        fallback_route = "rewriteArtifact" if current_artifact_model else "replyToGeneralInput"
        print(f"Defaulting to fallback route due to error: {fallback_route}")
        return {"next_node": fallback_route}
