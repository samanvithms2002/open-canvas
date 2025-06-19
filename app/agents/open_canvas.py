# app/agents/open_canvas.py
import inspect # For placeholder_node name
import copy # For deepcopy in clean_state_node

from langgraph.graph import StateGraph, END, START
# from langgraph.checkpoint.sqlite import SqliteSaver # Example checkpointer
from .state import OpenCanvasState
from typing import Dict, Any, Literal, Optional, List, Union

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage # type: ignore

from app.utils.text_processing import get_string_from_content, extract_urls
from app.agents.nodes.generate_path_helpers.documents import (
    convert_context_document_to_human_message,
    fix_misformatted_context_doc_message,
    RemoveMessage
)
from app.agents.nodes.generate_path_helpers.include_url_contents import include_url_contents_func
from app.agents.nodes.generate_path_helpers.dynamic_determine_path import dynamic_determine_path_func
    from app.utils.langchain_helpers import ( # Added for reply_to_general_input
        get_model_from_config,
        get_artifact_content,
        format_artifact_content_with_template,
        create_context_document_messages,
        is_using_o1_mini_model,
        # format_reflections, # Needs porting if reflections store is used
    )
    from app.agents.prompts import CURRENT_ARTIFACT_PROMPT, NO_ARTIFACT_PROMPT # Added
    # Imports for generate_artifact (already present from previous step, ensure they are correctly placed)
    from app.agents.nodes.generate_artifact_helpers.schemas import ArtifactToolSchema
    from app.agents.nodes.generate_artifact_helpers.utils import format_new_artifact_prompt, create_artifact_content
    from app.schemas.common import ArtifactV3, ArtifactType, ArtifactMarkdownV3, ArtifactCodeV3, ProgrammingLanguageOptions

    # Imports for rewrite_artifact
    from app.agents.nodes.rewrite_artifact_helpers.schemas import OptionallyUpdateArtifactMetaSchema
    from app.agents.nodes.rewrite_artifact_helpers.update_meta import optionally_update_artifact_meta
    from app.agents.nodes.rewrite_artifact_helpers.utils import (
        validate_state as validate_rewrite_state,
        build_prompt as build_rewrite_prompt,
        create_new_artifact_content as create_new_rewrite_artifact_content # Aliased
    )
    from app.utils.text_processing import is_thinking_model, extract_thinking_and_response_tokens # Added
    from app.utils.langchain_helpers import get_model_config # ensure get_model_config is available
    import uuid # Added for thinking message ID


# Placeholder for DEFAULT_INPUTS equivalent in Python
DEFAULT_INPUTS_PYTHON = {
    "messages": [],
    "_messages": [],
    "highlightedCode": None,
    "highlightedText": None,
    "artifact": None,
    "next_node": None,
    "language": None,
    "artifactLength": None,
    "regenerateWithEmojis": None,
    "readingLevel": None,
    "addComments": None,
    "addLogs": None,
    "portLanguage": None,
    "fixBugs": None,
    "customQuickActionId": None,
    "webSearchEnabled": None,
    "webSearchResults": None,
}

async def generate_path(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: generate_path (Full Implementation Attempt)")
    if config is None:
        config = {}

    # Ensure 'configurable' key exists in config, as helpers might expect it.
    # The graph's `compile(checkpointer=memory, interrupt_before=["generate_path"])`
    # call will pass the full RunnableConfig, which has 'configurable'.
    # If calling this node directly in a test, ensure config is shaped like RunnableConfig.
    graph_config = config.get("configurable", {})


    # Messages to be added to the state's 'messages' key (user-facing or UI-hidden context)
    processed_messages_for_state: List[BaseMessage] = []
    # Internal messages list, potentially modified by URL scraping, for LLM calls.
    current_internal_messages: List[BaseMessage] = list(state.get("_messages", []))

    # 1. Document Handling (convertContextDocumentToHumanMessage)
    # Operates on current_internal_messages (which is state._messages)
    doc_message = await convert_context_document_to_human_message(current_internal_messages, graph_config)
    if doc_message:
        processed_messages_for_state.append(doc_message)

    # 2. Document Handling (fixMisFormattedContextDocMessage)
    ids_to_remove_from_internal_messages = set()
    messages_to_add_to_internal_messages = [] # For current_internal_messages
    messages_to_add_to_processed_for_state = [] # For processed_messages_for_state

    temp_current_internal_messages_for_fixing = list(current_internal_messages) # Iterate over a copy

    for i, msg_to_check in enumerate(temp_current_internal_messages_for_fixing):
        if isinstance(msg_to_check, HumanMessage) and msg_to_check.id: # Ensure message has an ID for removal
            fixed_msgs_result = await fix_misformatted_context_doc_message(msg_to_check, graph_config)
            if fixed_msgs_result:
                for f_msg in fixed_msgs_result:
                    if isinstance(f_msg, RemoveMessage) and f_msg.id_to_remove:
                        ids_to_remove_from_internal_messages.add(f_msg.id_to_remove)
                    elif isinstance(f_msg, HumanMessage):
                        messages_to_add_to_internal_messages.append(f_msg)
                        # If a message was fixed, the fixed version should also be in processed_messages_for_state
                        # if it's a UI-hidden context message.
                        if f_msg.additional_kwargs.get("oc_hide_from_ui"):
                             messages_to_add_to_processed_for_state.append(f_msg)

    if ids_to_remove_from_internal_messages:
        current_internal_messages = [m for m in current_internal_messages if m.id not in ids_to_remove_from_internal_messages]
        current_internal_messages.extend(messages_to_add_to_internal_messages)

        # Also update processed_messages_for_state if any of its messages were among those removed/fixed
        processed_messages_for_state = [m for m in processed_messages_for_state if m.id not in ids_to_remove_from_internal_messages]
        processed_messages_for_state.extend(messages_to_add_to_processed_for_state)


    # 3. Direct Routing based on State fields
    # Output of direct routing should be: {"next_node": "...", "messages": msgs_for_state_update_or_none}
    # Where messages is for the main 'messages' state key. _messages is not updated by direct routes.
    direct_route_messages_update = processed_messages_for_state or None # Use None if empty

    if state.get("highlightedCode"):
        return {"next_node": "updateArtifact", "messages": direct_route_messages_update}
    if state.get("highlightedText"):
        return {"next_node": "updateHighlightedText", "messages": direct_route_messages_update}
    if state.get("language") or state.get("artifactLength") or state.get("regenerateWithEmojis") or state.get("readingLevel"):
        return {"next_node": "rewriteArtifactTheme", "messages": direct_route_messages_update}
    if state.get("addComments") or state.get("addLogs") or state.get("portLanguage") or state.get("fixBugs"):
        return {"next_node": "rewriteCodeArtifactTheme", "messages": direct_route_messages_update}
    if state.get("customQuickActionId"):
        return {"next_node": "customAction", "messages": direct_route_messages_update}
    # webSearchEnabled is a trigger for the webSearch node.
    # If it's true, it means the user's query might need web search BEFORE normal routing.
    if state.get("webSearchEnabled"):
        # The webSearch node itself will handle search and then route to generateArtifact/rewriteArtifact.
        # It will also set webSearchEnabled to False.
        # processed_messages_for_state are passed along.
        return {"next_node": "webSearch", "messages": direct_route_messages_update}

    # 4. URL Extraction & Content Inclusion (operates on current_internal_messages)
    if current_internal_messages:
        last_internal_message = current_internal_messages[-1]
        if isinstance(last_internal_message, HumanMessage):
            last_message_content_str = get_string_from_content(last_internal_message.content)
            urls_in_last_message = extract_urls(last_message_content_str)
            if urls_in_last_message:
                # include_url_contents_func expects the 'configurable' part of the config
                updated_message_from_url_inclusion = await include_url_contents_func(last_internal_message, urls_in_last_message, graph_config)
                if updated_message_from_url_inclusion:
                    current_internal_messages = current_internal_messages[:-1] + [updated_message_from_url_inclusion]

    # 5. Dynamic Path Determination
    # Pass the original state, but dynamic_determine_path_func will use its own copy
    # and primarily relies on the _messages it's given (current_internal_messages here).
    # processed_messages_for_state are the "newly_processed_messages" for the routing LLM context.
    routing_result = await dynamic_determine_path_func(
        state, # Original state for other context (like artifact presence)
        processed_messages_for_state, # UI-hidden docs for routing LLM context
        graph_config # Pass 'configurable' part of config
    )

    route_name = routing_result.get("next_node", "replyToGeneralInput")


    # 6. Construct final return dictionary
    # 'messages' key for user-facing state.messages (UI hidden docs)
    # '_messages' key for internal LLM context state._messages (actual conversation + UI hidden docs)
    final_messages_for_main_state = processed_messages_for_state or None

    # Combine current_internal_messages (which might have been updated by URL inclusion)
    # with processed_messages_for_state (UI-hidden context docs) for the next LLM call's context.
    final_internal_messages_for_llm_context = current_internal_messages + (processed_messages_for_state if processed_messages_for_state else [])

    # Filter out any None values just in case, though lists should be empty not contain None
    final_internal_messages_for_llm_context = [m for m in final_internal_messages_for_llm_context if m is not None]


    return {
        "next_node": route_name,
        "messages": final_messages_for_main_state,
        "_messages": final_internal_messages_for_llm_context
    }

# --- Other Node Placeholders (Signatures updated to include config) ---
async def reply_to_general_input(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: reply_to_general_input")
    # graph_config = config.get("configurable", {}) if config else {}
    # llm = await get_model_from_config(graph_config)
    # response = await llm.ainvoke(state.get("_messages", []))
    # return {"messages": [response]} # This should update 'messages' for UI, and also '_messages' for history

    # Simplified placeholder return for now:
    # new_ai_message = AIMessage(content="This is a general reply from reply_to_general_input.")
    # return {
    #     "messages": [new_ai_message], # Adds to user-facing history
    #     "_messages": state.get("_messages", []) + [new_ai_message] # Adds to internal history
    # }
    print("Executing Node: reply_to_general_input")
    if config is None: config = {}
    # graph_config should be the 'configurable' part of the RunnableConfig
    graph_config = config.get("configurable", {})


    llm = await get_model_from_config(graph_config)

    # Placeholder for reflections - replace with actual store logic later
    # reflections_data = await get_reflections_from_store(graph_config) # Needs implementation
    # memories_as_string = format_reflections(reflections_data) if reflections_data else "No reflections available (placeholder)."
    memories_as_string = "No reflections available (placeholder)." # TS uses ensureStoreInConfig

    current_artifact_model = get_artifact_content(state.get("artifact"))
    current_artifact_prompt_str = ""
    if current_artifact_model:
        current_artifact_prompt_str = format_artifact_content_with_template(
            CURRENT_ARTIFACT_PROMPT, current_artifact_model, shorten_content=True, max_length=3000 # Keep it reasonably sized
        )
    else:
        current_artifact_prompt_str = NO_ARTIFACT_PROMPT

    # Define the prompt template here or import from prompts.py
    REPLY_TO_GENERAL_INPUT_PROMPT_TEMPLATE = """You are an AI assistant tasked with responding to the users question.
The user has generated artifacts in the past. Use the following artifacts as context when responding to the users question.
You also have the following reflections on style guidelines and general memories/facts about the user to use when generating your response.
<reflections>
{reflections}
</reflections>
{currentArtifactPrompt}"""

    formatted_prompt = REPLY_TO_GENERAL_INPUT_PROMPT_TEMPLATE.format(
        reflections=memories_as_string,
        currentArtifactPrompt=current_artifact_prompt_str
    )

    # Context documents from config (e.g. uploaded files if any were attached to the config for this run)
    # create_context_document_messages takes the 'configurable' part of the config
    context_document_llm_parts = await create_context_document_messages(graph_config)

    context_docs_messages_for_llm = []
    if context_document_llm_parts: # Ensure it's not an empty list
        # Wrap parts in a single HumanMessage, marked as hidden from UI
        context_docs_messages_for_llm.append(HumanMessage(
            content=context_document_llm_parts,
            additional_kwargs={"oc_hide_from_ui": True}
            ))

    # Determine role for the main system/user prompt based on model
    prompt_is_user_role = is_using_o1_mini_model(graph_config)

    llm_messages: List[BaseMessage] = []
    if prompt_is_user_role:
        llm_messages.append(HumanMessage(content=formatted_prompt))
    else:
        llm_messages.append(SystemMessage(content=formatted_prompt))

    llm_messages.extend(context_docs_messages_for_llm) # Add formatted context document message if any

    # Add chat history (_messages from state)
    # Filter out any None messages from state._messages just in case
    history_messages = [msg for msg in state.get("_messages", []) if msg is not None]
    llm_messages.extend(history_messages)

    response = await llm.ainvoke(llm_messages)

    # The TS code returns [response] for both 'messages' and '_messages'.
    # This implies the AI's response becomes the sole message in these lists for the next step.
    # This is correct if the graph's `OpenCanvasState` TypedDict uses a custom reducer for `_messages`
    # (like `self.new_messages + messages`) as seen in some LangGraph examples or the original TS.
    # If default reducer (overwrite) is used, then this is fine.
    # If additive reducer is used for _messages, then `state.get("_messages", []) + [response]` would be for _messages.
    # For 'messages' (UI), returning just [response] is usually what's desired for a new turn.
    return {
        "messages": [response],
        "_messages": [response] # Mirrored TS return. Assumes specific reducer for _messages if history is to be preserved AND appended.
                               # If OpenCanvasState._messages uses default TypedDict update (overwrite), this is the only message for next node.
                               # If it's like `operator.add` or a custom list extend, then it appends.
                               # The TS GraphState reducer for _messages is: `(left, right) => (right ?? []).concat(left ?? [])`
                               # which means new messages (right) are prepended. So [response] + state._messages would be closer.
                               # Or, if the meaning is "this is THE new list of messages", then [response] is correct.
                               # Given the name "_messages" often implies the whole history for the LLM, prepending/appending seems more likely.
                               # Let's adjust to append for _messages, as it's more common for history.
                               # messages: [response] -> for UI display of current turn
                               # _messages: state.get("_messages", []) + [response] -> for next LLM call context
                               # This seems more standard for many LangGraph examples.
                               # However, the TS code's OpenCanvasGraphState explicitly defines a reducer for _messages:
                               # `(left: BaseMessage[] | undefined, right: BaseMessage[] | undefined): BaseMessage[] => (right ?? []).concat(left ?? []);`
                               # This means `right` (the new message list from node output) is prepended to `left` (existing messages in state).
                               # So, if node returns `_messages: [response]`, then state becomes `[response] + existing_messages`.
                               # This is unusual. More typical is `existing_messages + [response]`.
                               # Sticking to the direct port of `_messages: [response]` for now as per prompt.
    }


# Placeholder for get_formatted_reflections - replace with actual store logic later
async def get_formatted_reflections(config: Dict[str, Any]) -> str:
    # graph_config is the 'configurable' part of RunnableConfig
    # Actual logic would involve config.store.get(...) or similar checkpoint interaction
    print("Warning: Using placeholder for get_formatted_reflections.")
    # Example: store = config.get("store") if config else None
    # if store: reflections = await store.mget("reflections_key") ...
    return "No specific user reflections available at this time (placeholder)."

# Placeholder/inline for optionally_get_system_prompt_from_config
def optionally_get_system_prompt_from_config(graph_config: Dict[str, Any]) -> Optional[str]:
    # graph_config is the 'configurable' part of RunnableConfig
    return graph_config.get("systemPrompt")


async def generate_artifact(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: generate_artifact")
    if config is None: config = {}
    graph_config = config.get("configurable", {}) # Extract 'configurable' part

    # Get model and its name, explicitly enabling tool calling
    # Temperature is set to 0.5 for this node in TS, pass as extra
    model_details = get_model_config(graph_config, extra={"isToolCalling": True}) # get_model_config needs 'configurable'
    model_name = model_details.get("model_name", "unknown_model")

    llm = await get_model_from_config(graph_config, extra={"temperature": 0.5, "isToolCalling": True})

    # Define the tool schema for artifact generation
    # from app.agents.nodes.generate_artifact_helpers.schemas import ArtifactToolSchema (already imported if at top)
    # from app.agents.nodes.generate_artifact_helpers.utils import format_new_artifact_prompt, create_artifact_content (already imported)
    # from app.schemas.common import ArtifactV3, ArtifactType (already imported)

    tool_payload = {
        "name": "generate_artifact", # Must match the tool call name expected by the LLM
        "description": "Generates an artifact which can be text or code based on the user's query and chat history.",
        "parameters": ArtifactToolSchema.schema()
    }

    llm_with_artifact_tool = llm.bind_tools(
        tools=[tool_payload],
        tool_choice="generate_artifact" # Force the LLM to use this tool
    )

    memories_as_string = await get_formatted_reflections(graph_config) # Placeholder
    formatted_prompt_str = format_new_artifact_prompt(memories_as_string, model_name)

    user_system_prompt = optionally_get_system_prompt_from_config(graph_config) # Pass 'configurable'
    # Combine system prompts if user_system_prompt exists
    full_system_prompt = f"{user_system_prompt}\n\n{formatted_prompt_str}" if user_system_prompt else formatted_prompt_str

    # Context documents (e.g. from file uploads)
    context_document_llm_parts = await create_context_document_messages(graph_config) # Pass 'configurable'
    context_docs_messages_for_llm = []
    if context_document_llm_parts:
        context_docs_messages_for_llm.append(HumanMessage(content=context_document_llm_parts, additional_kwargs={"oc_hide_from_ui": True}))

    # Determine role for the main system/user prompt based on model name
    prompt_is_user_role = is_using_o1_mini_model(graph_config) # Pass 'configurable'

    llm_messages_for_invoke: List[BaseMessage] = []
    if prompt_is_user_role:
        llm_messages_for_invoke.append(HumanMessage(content=full_system_prompt))
    else:
        llm_messages_for_invoke.append(SystemMessage(content=full_system_prompt))

    llm_messages_for_invoke.extend(context_docs_messages_for_llm)

    # Add chat history (_messages from state)
    history_messages = [msg for msg in state.get("_messages", []) if msg is not None]
    llm_messages_for_invoke.extend(history_messages)

    # Invoke the LLM with the prepared messages and tool choice
    # The run_name is for LangSmith tracing, can be omitted if not using LangSmith
    llm_response = await llm_with_artifact_tool.ainvoke(llm_messages_for_invoke, {"run_name": "generate_artifact_llm_call"})

    args_dict = None
    # Check for tool calls in the response
    if hasattr(llm_response, "tool_calls") and llm_response.tool_calls:
        # Assuming the first tool call is the one we want
        # In TS, it checks `if (response.tool_calls?.[0]?.name === "generateArtifact")`
        # Here, we forced tool_choice, so it should be "generate_artifact"
        first_tool_call = llm_response.tool_calls[0]
        if first_tool_call.get("name") == "generate_artifact":
            args_dict = first_tool_call.get("args")
        else:
            print(f"Warning: Expected tool 'generate_artifact' but got '{first_tool_call.get('name')}'")

    if not args_dict:
        print(f"Error: generate_artifact tool call did not return valid arguments. Response: {llm_response.content}")
        # Create a fallback text artifact indicating failure
        error_artifact_content = ArtifactMarkdownV3(
            index=0, type=ArtifactType.TEXT, title="Error Generating Artifact",
            fullMarkdown=f"I was unable to generate the artifact as requested. The model response was: {llm_response.content}"
        )
        error_artifact = ArtifactV3(currentIndex=0, contents=[error_artifact_content])
        # Also provide an AI message for the chat
        error_ai_message = AIMessage(content="I encountered an issue generating the artifact. Please try again or rephrase your request.")
        return {
            "artifact": error_artifact,
            "messages": [error_ai_message], # Update user-facing messages
            "_messages": state.get("_messages", []) + [error_ai_message] # Update internal history
        }

    try:
        # Validate and parse the arguments using the Pydantic schema
        parsed_args = ArtifactToolSchema(**args_dict)
    except Exception as e: # Catches Pydantic ValidationError
        print(f"Error parsing tool arguments for generate_artifact: {e}. Args: {args_dict}")
        error_artifact_content = ArtifactMarkdownV3(
            index=0, type=ArtifactType.TEXT, title="Error Parsing Artifact Data",
            fullMarkdown=f"There was an issue with the data format for the artifact: {e}. Received arguments: {args_dict}"
        )
        error_artifact = ArtifactV3(currentIndex=0, contents=[error_artifact_content])
        error_ai_message = AIMessage(content=f"I had trouble formatting the generated artifact due to: {e}.")
        return {
            "artifact": error_artifact,
            "messages": [error_ai_message],
            "_messages": state.get("_messages", []) + [error_ai_message]
        }

    # Create the actual artifact content object (ArtifactCodeV3 or ArtifactMarkdownV3)
    new_artifact_content = create_artifact_content(parsed_args)

    # Embed this content within an ArtifactV3 structure
    new_artifact_v3 = ArtifactV3(
        currentIndex=0, # New artifact, so current index is 0 (pointing to the first and only content)
        contents=[new_artifact_content]
    )

    # The TS node for generateArtifact ONLY returns the new artifact.
    # It does not directly add any AIMessage to the chat history.
    # Any user-facing message about the artifact generation would typically come from a subsequent node
    # or if the LLM itself (if not using tools) generated text like "Okay, here is your artifact: ...".
    # Since we are forcing a tool call, the LLM's response *is* the tool call, not a chat message.
    # If a chat message is desired *in addition* to the artifact, it needs to be constructed here.
    # For now, strictly adhering to the TS return signature for this node:
    return {"artifact": new_artifact_v3}


async def rewrite_artifact(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: rewrite_artifact")
    if config is None: config = {}
    # graph_config is the 'configurable' part of RunnableConfig
    graph_config = config.get("configurable", {})

    model_details = get_model_config(graph_config)
    model_name = model_details.get("model_name", "unknown_model")

    llm = await get_model_from_config(graph_config)

    memories_as_string = await get_formatted_reflections(graph_config)

    try:
        current_artifact_content_model, recent_human_message = validate_rewrite_state(state)
    except ValueError as e:
        print(f"Validation error in rewrite_artifact: {e}")
        error_message = AIMessage(content=f"Error: Could not rewrite artifact - {e}")
        return {
            "messages": [error_message],
            "_messages": state.get("_messages", []) + [error_message]
        }

    artifact_meta_update_schema: OptionallyUpdateArtifactMetaSchema = await optionally_update_artifact_meta(state, config) # Pass full config

    determined_artifact_type = artifact_meta_update_schema.type
    is_new_type = determined_artifact_type != current_artifact_content_model.type

    current_artifact_text_content = ""
    if is_artifact_code_content(current_artifact_content_model) and isinstance(current_artifact_content_model, ArtifactCodeV3):
        current_artifact_text_content = current_artifact_content_model.code
    elif isinstance(current_artifact_content_model, ArtifactMarkdownV3):
        current_artifact_text_content = current_artifact_content_model.fullMarkdown

    formatted_prompt_str = build_rewrite_prompt({
        "artifact_content_str": current_artifact_text_content,
        "memories_as_string": memories_as_string,
        "is_new_type": is_new_type,
        "artifact_meta_tool_call": artifact_meta_update_schema,
    })

    user_system_prompt = optionally_get_system_prompt_from_config(graph_config)
    full_system_prompt = f"{user_system_prompt}\n{formatted_prompt_str}" if user_system_prompt else formatted_prompt_str

    context_document_llm_parts = await create_context_document_messages(graph_config)
    context_docs_messages_for_llm = []
    if context_document_llm_parts:
        context_docs_messages_for_llm.append(HumanMessage(content=context_document_llm_parts, additional_kwargs={"oc_hide_from_ui": True}))

    prompt_is_user_role = is_using_o1_mini_model(graph_config)

    llm_messages_for_invoke: List[BaseMessage] = []
    if prompt_is_user_role:
        llm_messages_for_invoke.append(HumanMessage(content=full_system_prompt))
    else:
        llm_messages_for_invoke.append(SystemMessage(content=full_system_prompt))

    llm_messages_for_invoke.extend(context_docs_messages_for_llm)
    llm_messages_for_invoke.append(recent_human_message)

    response_ai_message = await llm.ainvoke(llm_messages_for_invoke, {"run_name": "rewrite_artifact_llm_call"})

    actual_artifact_text_response = response_ai_message.content
    if not isinstance(actual_artifact_text_response, str):
        actual_artifact_text_response = str(actual_artifact_text_response)

    thinking_message_for_state: Optional[AIMessage] = None
    if is_thinking_model(model_name):
        extracted_parts = extract_thinking_and_response_tokens(actual_artifact_text_response)
        if extracted_parts["thinking"]:
            thinking_message_for_state = AIMessage(
                id=f"thinking-{uuid.uuid4()}",
                content=extracted_parts["thinking"],
                additional_kwargs={"oc_hide_from_ui": True}
            )
        actual_artifact_text_response = extracted_parts["response"]

    new_artifact_content_item = create_new_rewrite_artifact_content(CreateNewArtifactContentArgs( # type: ignore
        artifact_type=determined_artifact_type,
        state=state,
        current_artifact_content_model=current_artifact_content_model,
        artifact_meta_tool_call=artifact_meta_update_schema,
        new_content_text=actual_artifact_text_response,
    ))

    existing_artifact_v3: Optional[ArtifactV3] = state.get("artifact")
    if not existing_artifact_v3:
         error_message = AIMessage(content="Critical error: Original artifact missing during rewrite.")
         return {"messages": [error_message], "_messages": state.get("_messages", []) + [error_message]}

    updated_artifact_contents = list(existing_artifact_v3.contents) + [new_artifact_content_item]

    final_artifact_v3 = ArtifactV3(
        currentIndex=len(updated_artifact_contents) - 1,
        contents=updated_artifact_contents
    )

    update_dict: Dict[str, Any] = {"artifact": final_artifact_v3}
    if thinking_message_for_state:
        update_dict["messages"] = [thinking_message_for_state]
        update_dict["_messages"] = [thinking_message_for_state]

    return update_dict


async def rewrite_artifact_theme(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: rewrite_artifact_theme")
    if config is None: config = {}
    # graph_config should be the 'configurable' part for helpers
    graph_config = config.get("configurable", {})


    model_details = get_model_config(graph_config)
    model_name = model_details.get("model_name", "unknown_model")
    llm = await get_model_from_config(graph_config)

    memories_as_string = await get_formatted_reflections(graph_config) # Placeholder

    current_artifact_content_model = get_artifact_content(state.get("artifact"))
    if not current_artifact_content_model:
        # This should ideally be prevented by routing logic if no artifact exists.
        error_message = AIMessage(content="Cannot rewrite artifact theme: No artifact found in state.")
        return {"messages": [error_message], "_messages": state.get("_messages", []) + [error_message]} # type: ignore

    if current_artifact_content_model.type != ArtifactType.TEXT:
        error_message = AIMessage(content=f"Cannot rewrite artifact theme: Artifact is not text-based (type: {current_artifact_content_model.type}).")
        return {"messages": [error_message], "_messages": state.get("_messages", []) + [error_message]} # type: ignore

    current_markdown_artifact: ArtifactMarkdownV3 = current_artifact_content_model # type: ignore

    formatted_prompt = ""
    # State fields that trigger this node, e.g., state.language, state.readingLevel, etc.
    state_lang: Optional[LanguageOptions] = state.get("language")
    state_reading_level: Optional[ReadingLevelOptions] = state.get("readingLevel")
    state_artifact_length: Optional[ArtifactLengthOptions] = state.get("artifactLength")
    state_regen_emojis: Optional[bool] = state.get("regenerateWithEmojis")

    # Import prompts if not already at top level (ensure they are available)
    from app.agents.prompts import (
        CHANGE_ARTIFACT_LANGUAGE_PROMPT,
        CHANGE_ARTIFACT_READING_LEVEL_PROMPT,
        CHANGE_ARTIFACT_TO_PIRATE_PROMPT,
        CHANGE_ARTIFACT_LENGTH_PROMPT,
        ADD_EMOJIS_TO_ARTIFACT_PROMPT
    )

    if state_lang:
        formatted_prompt = CHANGE_ARTIFACT_LANGUAGE_PROMPT.format(
            newLanguage=state_lang.value,
            artifactContent=current_markdown_artifact.fullMarkdown,
            reflections=memories_as_string
        )
    elif state_reading_level and state_reading_level == ReadingLevelOptions.PIRATE: # Pirate needs to be checked before general reading level
        formatted_prompt = CHANGE_ARTIFACT_TO_PIRATE_PROMPT.format(
            artifactContent=current_markdown_artifact.fullMarkdown,
            reflections=memories_as_string
        )
    elif state_reading_level:
        level_description = {
            ReadingLevelOptions.CHILD: "an elementary school student", # More natural phrasing
            ReadingLevelOptions.TEENAGER: "a high school student",
            ReadingLevelOptions.COLLEGE: "a college student",
            ReadingLevelOptions.PHD: "a PhD student or expert in the field",
        }.get(state_reading_level, "a general audience")

        formatted_prompt = CHANGE_ARTIFACT_READING_LEVEL_PROMPT.format(
            newReadingLevel=level_description,
            artifactContent=current_markdown_artifact.fullMarkdown,
            reflections=memories_as_string
        )
    elif state_artifact_length:
        length_description = {
            ArtifactLengthOptions.SHORTEST: "significantly shorter, focusing only on the absolute key points",
            ArtifactLengthOptions.SHORT: "somewhat shorter and more concise",
            ArtifactLengthOptions.LONG: "somewhat longer with more detail or explanation",
            ArtifactLengthOptions.LONGEST: "significantly longer, expanding greatly on concepts with examples",
        }.get(state_artifact_length, "of a different length than it currently is")

        formatted_prompt = CHANGE_ARTIFACT_LENGTH_PROMPT.format(
            newLength=length_description,
            artifactContent=current_markdown_artifact.fullMarkdown,
            reflections=memories_as_string
        )
    elif state_regen_emojis is True: # Explicitly check for True
        formatted_prompt = ADD_EMOJIS_TO_ARTIFACT_PROMPT.format(
            artifactContent=current_markdown_artifact.fullMarkdown,
            reflections=memories_as_string
        )
    else:
        # If no specific theme change is identified, this node might have been routed incorrectly.
        error_message = AIMessage(content="Cannot rewrite artifact theme: No relevant theme modification option was specified.")
        return {"messages": [error_message], "_messages": state.get("_messages", []) + [error_message]} # type: ignore

    response_ai_message = await llm.ainvoke([HumanMessage(content=formatted_prompt)], {"run_name": "rewrite_artifact_theme_llm_call"})

    thinking_message_for_state: Optional[AIMessage] = None
    actual_artifact_text_response = response_ai_message.content
    if not isinstance(actual_artifact_text_response, str):
        actual_artifact_text_response = str(actual_artifact_text_response)

    if is_thinking_model(model_name):
        extracted = extract_thinking_and_response_tokens(actual_artifact_text_response)
        if extracted["thinking"]:
            thinking_message_for_state = AIMessage(
                id=f"thinking-{uuid.uuid4()}",
                content=extracted["thinking"],
                additional_kwargs={"oc_hide_from_ui": True}
                )
        actual_artifact_text_response = extracted["response"]

    existing_artifact_v3: Optional[ArtifactV3] = state.get("artifact")
    if not existing_artifact_v3: # Should be caught by earlier check
         error_message = AIMessage(content="Critical error: Original artifact missing during theme rewrite.")
         return {"messages": [error_message], "_messages": state.get("_messages", []) + [error_message]} # type: ignore

    # Create new content item (copy of current, but with new text and index)
    new_markdown_content_item = ArtifactMarkdownV3(
        index=len(existing_artifact_v3.contents),
        type=ArtifactType.TEXT,
        title=current_markdown_artifact.title, # Title remains the same
        fullMarkdown=actual_artifact_text_response
    )

    updated_artifact_contents = list(existing_artifact_v3.contents) + [new_markdown_content_item]

    final_artifact_v3 = ArtifactV3(
        currentIndex=len(updated_artifact_contents) - 1,
        contents=updated_artifact_contents
    )

    update_dict: Dict[str, Any] = {"artifact": final_artifact_v3}
    # Reset the state field that triggered this rewrite
    if state_lang: update_dict["language"] = None
    if state_reading_level: update_dict["readingLevel"] = None
    if state_artifact_length: update_dict["artifactLength"] = None
    if state_regen_emojis: update_dict["regenerateWithEmojis"] = None

    if thinking_message_for_state:
        update_dict["messages"] = [thinking_message_for_state]
        update_dict["_messages"] = [thinking_message_for_state]

    return update_dict


async def generate_followup(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: generate_followup")
    if config is None: config = {}
    # graph_config should be the 'configurable' part of RunnableConfig
    graph_config = config.get("configurable", {})


    # Pass max_tokens and is_tool_calling to influence model choice/behavior as in TS
    # The 'is_tool_calling' here might be a heuristic from TS to get a certain type of model
    # or behavior, not necessarily to force a tool call in *this* node.
    llm = await get_model_from_config(graph_config, extra={"max_tokens": 250, "is_tool_calling": True})

    # Placeholder for reflections. The TS version calls formatReflections with { onlyContent: true }.
    # The current Python placeholder get_formatted_reflections doesn't have this option.
    # This might mean the reflections string includes more than just content, which could affect the prompt.
    # For now, using the existing general placeholder.
    memories_as_string = await get_formatted_reflections(graph_config)

    current_artifact_content_model = get_artifact_content(state.get("artifact"))
    artifact_content_str = "User has not generated an artifact yet." # Default if no artifact
    if current_artifact_content_model:
        if current_artifact_content_model.type == ArtifactType.CODE and isinstance(current_artifact_content_model, ArtifactCodeV3):
            artifact_content_str = current_artifact_content_model.code
        elif isinstance(current_artifact_content_model, ArtifactMarkdownV3): # TEXT
            artifact_content_str = current_artifact_content_model.fullMarkdown

    # Format entire message history from _messages for the prompt
    # The TS version uses a specific format for messages in the prompt.
    # _format_messages_for_custom_action_prompt creates <type>...</type> blocks.
    # FOLLOWUP_ARTIFACT_PROMPT has {conversation} and also {lastMessage} in TS,
    # but the Python version from prompts.py seems to only have {conversation}.
    # Let's ensure FOLLOWUP_ARTIFACT_PROMPT is suitable for _format_messages_for_custom_action_prompt output.
    # The current FOLLOWUP_ARTIFACT_PROMPT in Python has {{artifact}} and {{lastMessage}} and {{reflections}}.
    # It does NOT have a {{conversation}} placeholder. This needs alignment.

    # Re-checking the Python FOLLOWUP_ARTIFACT_PROMPT from prompts.py:
    # FOLLOWUP_ARTIFACT_PROMPT = f"""You are an AI assistant tasked with generating a follow-up question...
    # Current artifact:\n<artifact>\n{{artifact}}\n</artifact>\n\nUser's last message:\n<user-message>\n{{lastMessage}}\n</user-message>..."""
    # This means we need the *last message* specifically, not the whole history formatted.

    last_message_str = "No previous messages."
    if state.get("_messages"):
        # The prompt asks for "User's last message". So, find the last actual message, not system/hidden ones.
        # However, the TS logic for `generateFollowup` prompt just takes the entire history.
        # The Python prompt `FOLLOWUP_ARTIFACT_PROMPT` was ported from a different TS prompt (`FOLLOWUP_PROMPT` from `generateFollowup.ts`)
        # which has `lastMessage` placeholder, not `conversation`.
        # The TS node `generateFollowup` uses `FOLLOWUP_PROMPT` with `lastMessage: state.messages[state.messages.length -1].content`

        # Let's align with the Python prompt for now, which expects 'lastMessage'.
        # We should probably use the last message from `state.messages` (user-visible) if that's the intent.
        # Or, if it's the last message from `_messages` (internal history):

        # For now, using last from _messages for simplicity, but this might need refinement.
        last_msg_obj = state.get("_messages", [])[-1] if state.get("_messages") else None
        if last_msg_obj:
            last_message_str = get_string_from_content(last_msg_obj.content)

    from app.agents.prompts import FOLLOWUP_ARTIFACT_PROMPT # Ensure it's imported

    prompt_str = FOLLOWUP_ARTIFACT_PROMPT.format( # Using the Python version of the prompt
        artifact=artifact_content_str, # Renamed from artifactContent to match Python prompt
        reflections=memories_as_string,
        lastMessage=last_message_str # Added lastMessage
    )

    # The prompt for generate_followup in TS is a simple HumanMessage.
    response_ai_message = await llm.ainvoke([HumanMessage(content=prompt_str)], {"run_name": "generate_followup_llm_call"})

    # The response from this LLM call is expected to be a follow-up question (as an AIMessage).
    # This AIMessage should be added to both 'messages' (for UI) and '_messages' (for history).
    # The TS pattern of returning `messages: [response], _messages: [response]` implies replacement
    # if the graph state reducer for these keys is the default.
    # Given this is a *new* follow-up message, it should typically be appended to history.
    # However, to strictly follow the pattern from other nodes if that's intended for specific reducer reasons:
    return {
        "messages": [response_ai_message],
        "_messages": [response_ai_message]
    }


async def rewrite_code_artifact_theme(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: rewrite_code_artifact_theme")
    if config is None: config = {}
    # graph_config should be the 'configurable' part for helpers
    graph_config = config.get("configurable", {})

    model_details = get_model_config(graph_config)
    model_name = model_details.get("model_name", "unknown_model")
    llm = await get_model_from_config(graph_config)

    current_artifact_content_model = get_artifact_content(state.get("artifact"))
    if not current_artifact_content_model:
        error_message = AIMessage(content="Cannot rewrite code artifact theme: No artifact found in state.")
        return {"messages": [error_message], "_messages": state.get("_messages", []) + [error_message]} # type: ignore

    if current_artifact_content_model.type != ArtifactType.CODE:
        error_message = AIMessage(content=f"Cannot rewrite code artifact theme: Artifact is not code-based (type: {current_artifact_content_model.type}).")
        return {"messages": [error_message], "_messages": state.get("_messages", []) + [error_message]} # type: ignore

    current_code_artifact: ArtifactCodeV3 = current_artifact_content_model # type: ignore

    formatted_prompt = ""
    # State fields that trigger this node
    add_comments: Optional[bool] = state.get("addComments")
    port_language: Optional[ProgrammingLanguageOptions] = state.get("portLanguage")
    add_logs: Optional[bool] = state.get("addLogs")
    fix_bugs: Optional[bool] = state.get("fixBugs")

    # To reset the trigger field later
    field_to_reset: Optional[str] = None

    # Import prompts
    from app.agents.prompts import (
        ADD_COMMENTS_TO_CODE_ARTIFACT_PROMPT,
        ADD_LOGS_TO_CODE_ARTIFACT_PROMPT,
        FIX_BUGS_CODE_ARTIFACT_PROMPT,
        PORT_LANGUAGE_CODE_ARTIFACT_PROMPT
    )

    if add_comments:
        field_to_reset = "addComments"
        formatted_prompt = ADD_COMMENTS_TO_CODE_ARTIFACT_PROMPT.format(artifactContent=current_code_artifact.code)
    elif port_language:
        field_to_reset = "portLanguage"
        language_display_name = port_language.value
        formatted_prompt = PORT_LANGUAGE_CODE_ARTIFACT_PROMPT.format(
            newLanguage=language_display_name,
            artifactContent=current_code_artifact.code
        )
    elif add_logs:
        field_to_reset = "addLogs"
        formatted_prompt = ADD_LOGS_TO_CODE_ARTIFACT_PROMPT.format(artifactContent=current_code_artifact.code)
    elif fix_bugs:
        field_to_reset = "fixBugs"
        formatted_prompt = FIX_BUGS_CODE_ARTIFACT_PROMPT.format(artifactContent=current_code_artifact.code)
    else:
        error_message = AIMessage(content="Cannot rewrite code artifact theme: No relevant code modification option selected in state.")
        return {"messages": [error_message], "_messages": state.get("_messages", []) + [error_message]} # type: ignore

    response_ai_message = await llm.ainvoke([HumanMessage(content=formatted_prompt)], {"run_name": "rewrite_code_artifact_theme_llm_call"})

    thinking_message_for_state: Optional[AIMessage] = None
    actual_artifact_text_response = response_ai_message.content
    if not isinstance(actual_artifact_text_response, str):
         actual_artifact_text_response = str(actual_artifact_text_response)

    if is_thinking_model(model_name):
        extracted = extract_thinking_and_response_tokens(actual_artifact_text_response)
        if extracted["thinking"]:
            thinking_message_for_state = AIMessage(
                id=f"thinking-{uuid.uuid4()}",
                content=extracted["thinking"],
                additional_kwargs={"oc_hide_from_ui": True}
                )
        actual_artifact_text_response = extracted["response"]

    existing_artifact_v3: Optional[ArtifactV3] = state.get("artifact")
    if not existing_artifact_v3: # Should be caught by earlier check
         error_message = AIMessage(content="Critical error: Original artifact missing during code theme rewrite.")
         return {"messages": [error_message], "_messages": state.get("_messages", []) + [error_message]} # type: ignore

    new_code_language = port_language if port_language else current_code_artifact.language

    new_content_item = ArtifactCodeV3(
        index=len(existing_artifact_v3.contents),
        type=ArtifactType.CODE,
        title=current_code_artifact.title,
        language=new_code_language,
        code=actual_artifact_text_response
    )

    updated_artifact_contents = list(existing_artifact_v3.contents) + [new_content_item]

    final_artifact_v3 = ArtifactV3(
        currentIndex=len(updated_artifact_contents) - 1,
        contents=updated_artifact_contents
    )

    return_dict: Dict[str, Any] = {
        "artifact": final_artifact_v3,
    }
    if thinking_message_for_state:
        return_dict["messages"] = [thinking_message_for_state]
        return_dict["_messages"] = [thinking_message_for_state]

    if field_to_reset:
        return_dict[field_to_reset] = None

    return return_dict


# Helper to format messages for custom_action prompt (from TS node)
def _format_messages_for_custom_action_prompt(messages: List[BaseMessage]) -> str:
    return "\n".join(
        # Using type attribute as per BaseMessage standard
        [f"<{msg.type}>\n{get_string_from_content(msg.content)}\n</{msg.type}>" for msg in messages]
    )

async def custom_action(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: custom_action")
    if config is None: config = {}
    # graph_config is the 'configurable' part of RunnableConfig
    graph_config = config.get("configurable", {})


    custom_action_id = state.get("customQuickActionId")
    if not custom_action_id:
        error_msg ="Cannot execute custom action: No customQuickActionId found in state."
        # Return an error message to be displayed in chat
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore


    # LLM for this node uses temperature 0.5 as per TS
    llm = await get_model_from_config(graph_config, extra={"temperature": 0.5})

    # --- Placeholder for Store Interaction: Fetching Custom Action and Reflections ---
    # from app.schemas.common import CustomQuickAction # Already imported at top if done globally

    # In a real scenario, this would involve:
    # supabase_user_id = graph_config.get("supabase_user_id")
    # store = get_store_from_config(config) # Assuming a utility to get the checkpointer/store
    # all_custom_actions = await store.get(f"custom_actions:{supabase_user_id}")
    # custom_action_details_dict = all_custom_actions.get(custom_action_id) if all_custom_actions else None
    # if not custom_action_details_dict:
    #     error_msg = f"Custom action with ID '{custom_action_id}' not found."
    #     return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)], "customQuickActionId": None}

    print(f"Warning: Using placeholder for custom action details (ID: {custom_action_id}).")
    custom_quick_action_details_dict = { # Dummy data
        "id": custom_action_id,
        "title": f"Dummy Action: {custom_action_id}",
        "prompt": "Please summarize the following artifact content.", # Example prompt
        "includeReflections": True,
        "includePrefix": True,
        "includeRecentHistory": True
    }
    try:
        custom_action_details = CustomQuickAction(**custom_quick_action_details_dict)
    except Exception as e: # Pydantic ValidationError
        error_msg = f"Invalid format for custom action details (ID: {custom_action_id}): {e}"
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)], "customQuickActionId": None} # type: ignore

    memories_as_string = await get_formatted_reflections(graph_config) # Existing placeholder
    # --- End Placeholder ---

    current_artifact_content_model = get_artifact_content(state.get("artifact"))

    # Import prompts here or ensure they are available at module level
    from app.agents.prompts import (
        REFLECTIONS_QUICK_ACTION_PROMPT,
        CUSTOM_QUICK_ACTION_ARTIFACT_PROMPT_PREFIX,
        CUSTOM_QUICK_ACTION_CONVERSATION_CONTEXT,
        CUSTOM_QUICK_ACTION_ARTIFACT_CONTENT_PROMPT
    )

    prompt_parts = []
    if custom_action_details.includePrefix:
        prompt_parts.append(CUSTOM_QUICK_ACTION_ARTIFACT_PROMPT_PREFIX)

    prompt_parts.append(f"<custom-instructions>\n{custom_action_details.prompt}\n</custom-instructions>")

    if custom_action_details.includeReflections and memories_as_string and "No specific user reflections available" not in memories_as_string:
        prompt_parts.append(REFLECTIONS_QUICK_ACTION_PROMPT.format(reflections=memories_as_string))

    if custom_action_details.includeRecentHistory:
        recent_history = state.get("_messages", [])[-5:] # Get last 5 messages
        if recent_history:
            formatted_history = _format_messages_for_custom_action_prompt(recent_history)
            prompt_parts.append(CUSTOM_QUICK_ACTION_CONVERSATION_CONTEXT.format(conversation=formatted_history))

    artifact_content_for_prompt_str = "No artifacts generated yet."
    if current_artifact_content_model:
        if current_artifact_content_model.type == ArtifactType.CODE and isinstance(current_artifact_content_model, ArtifactCodeV3):
            artifact_content_for_prompt_str = current_artifact_content_model.code
        elif isinstance(current_artifact_content_model, ArtifactMarkdownV3): # TEXT
            artifact_content_for_prompt_str = current_artifact_content_model.fullMarkdown

    prompt_parts.append(CUSTOM_QUICK_ACTION_ARTIFACT_CONTENT_PROMPT.format(artifactContent=artifact_content_for_prompt_str))

    final_prompt = "\n\n".join(prompt_parts)

    response_ai_message = await llm.ainvoke([HumanMessage(content=final_prompt)], {"run_name": f"custom_action_{custom_action_id}_llm_call"})
    llm_response_content = response_ai_message.content
    if not isinstance(llm_response_content, str):
        llm_response_content = str(llm_response_content)

    if not current_artifact_content_model:
        print("Warning: Custom action was invoked, but no current artifact exists to apply it to. Discarding LLM response.")
        return {"customQuickActionId": None} # Reset trigger, no artifact change

    existing_artifact_v3: Optional[ArtifactV3] = state.get("artifact")
    if not existing_artifact_v3: # Should be caught by current_artifact_content_model check
         error_msg = "Critical error: Artifact missing during custom action."
         return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)], "customQuickActionId": None} # type: ignore

    new_index = len(existing_artifact_v3.contents)
    new_content_item: Union[ArtifactCodeV3, ArtifactMarkdownV3]

    if current_artifact_content_model.type == ArtifactType.CODE and isinstance(current_artifact_content_model, ArtifactCodeV3):
        new_content_item = ArtifactCodeV3(
            index=new_index, type=ArtifactType.CODE,
            title=current_artifact_content_model.title,
            language=current_artifact_content_model.language,
            code=llm_response_content
        )
    elif isinstance(current_artifact_content_model, ArtifactMarkdownV3): # TEXT
        new_content_item = ArtifactMarkdownV3(
            index=new_index, type=ArtifactType.TEXT,
            title=current_artifact_content_model.title,
            fullMarkdown=llm_response_content
        )
    else: # Should not happen due to earlier checks
        error_msg = f"Unsupported artifact type for custom action: {current_artifact_content_model.type}"
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)], "customQuickActionId": None} # type: ignore

    updated_artifact_contents = list(existing_artifact_v3.contents) + [new_content_item]
    final_artifact_v3 = ArtifactV3(
        currentIndex=new_index,
        contents=updated_artifact_contents
    )

    return {
        "artifact": final_artifact_v3,
        "customQuickActionId": None # Reset the trigger
    }


async def update_artifact(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: update_artifact")
    if config is None: config = {}
    # graph_config is the 'configurable' part of RunnableConfig
    graph_config = config.get("configurable", {})

    original_model_details = get_model_config(graph_config)
    model_provider = original_model_details.get("model_provider", "").lower()
    model_name = original_model_details.get("model_name", "").lower()

    llm_config_to_use = graph_config # By default, use the graph's main config

    # This list of "capable models" might need refinement based on testing.
    # These are models assumed to be good at precise code editing tasks.
    capable_models_for_update = ["claude-3-5-sonnet", "gpt-4o", "gpt-4.1", "gpt-4-turbo"]

    is_openai_provider = "openai" in model_provider
    is_anthropic_provider = "anthropic" in model_provider

    # Check if the current model is considered capable for this specific task
    is_current_model_capable = False
    if is_openai_provider or is_anthropic_provider: # Only consider these providers for now
        if any(cap_model in model_name for cap_model in capable_models_for_update):
            is_current_model_capable = True

    if not is_current_model_capable:
        print(f"Original model {model_name} (provider: {model_provider}) not deemed suitable for precise artifact update. Overriding to gpt-4o.")
        # Create a new config dictionary for override
        override_configurable = graph_config.copy() # Start with a copy of existing 'configurable'
        override_configurable["customModelName"] = "gpt-4o"

        # Ensure modelConfig provider is also updated for get_model_from_config
        if "modelConfig" not in override_configurable or not isinstance(override_configurable.get("modelConfig"), dict):
            override_configurable["modelConfig"] = {"provider": "openai"}
        else:
            override_configurable["modelConfig"]["provider"] = "openai"

        # Ensure temperature 0.0 is set for the override
        # get_model_from_config will use this if passed in 'extra', or it can be part of modelConfig
        # For clarity, we can ensure it's in the modelConfig for the override.
        if "temperature" in override_configurable.get("modelConfig", {}): # Check if CustomModelConfig has temperature
             override_configurable["modelConfig"]["temperature"] = 0.0

        llm_config_to_use = override_configurable # Use the modified 'configurable' dict
        llm = await get_model_from_config(llm_config_to_use, extra={"temperature": 0.0})
    else:
        # Use original config but ensure temperature is 0.0 for this specific call
        llm = await get_model_from_config(llm_config_to_use, extra={"temperature": 0.0})

    memories_as_string = await get_formatted_reflections(graph_config) # Placeholder

    current_artifact_content_model = get_artifact_content(state.get("artifact"))
    if not current_artifact_content_model:
        error_msg = "Cannot update artifact: No artifact found in state."
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore

    if current_artifact_content_model.type != ArtifactType.CODE:
        error_msg = f"Cannot update artifact: Artifact is not code-based (type: {current_artifact_content_model.type}). This action is for code artifacts."
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore

    current_code_artifact: ArtifactCodeV3 = current_artifact_content_model # type: ignore

    highlighted_code_details_dict = state.get("highlightedCode")
    if not highlighted_code_details_dict:
        error_msg = "Cannot update artifact: No highlightedCode details found in state."
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore

    from app.schemas.common import CodeHighlight # Import here to avoid circularity if moved
    try:
        highlighted_code = CodeHighlight(**highlighted_code_details_dict)
    except Exception as e:
        error_msg = f"Invalid highlightedCode format in state: {e}"
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore

    context_window = 500 # Characters before and after highlight
    start_context_idx = max(0, highlighted_code.startCharIndex - context_window)
    end_context_idx = min(len(current_code_artifact.code), highlighted_code.endCharIndex + context_window)

    code_to_display_before_highlight = current_code_artifact.code[start_context_idx : highlighted_code.startCharIndex]
    highlighted_text_str = current_code_artifact.code[highlighted_code.startCharIndex : highlighted_code.endCharIndex]
    code_to_display_after_highlight = current_code_artifact.code[highlighted_code.endCharIndex : end_context_idx]

    from app.agents.prompts import UPDATE_HIGHLIGHTED_ARTIFACT_PROMPT # Import prompt

    formatted_prompt = UPDATE_HIGHLIGHTED_ARTIFACT_PROMPT.format(
        highlightedText=highlighted_text_str,
        beforeHighlight=code_to_display_before_highlight,
        afterHighlight=code_to_display_after_highlight,
        reflections=memories_as_string
    )

    recent_human_message = next((msg for msg in reversed(state.get("_messages", [])) if msg.type == "human"), None)
    if not recent_human_message:
        error_msg = "Cannot update artifact: No recent human message found for context."
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore

    context_document_llm_parts = await create_context_document_messages(graph_config)
    context_docs_messages_for_llm = [HumanMessage(content=context_document_llm_parts, additional_kwargs={"oc_hide_from_ui": True})] if context_document_llm_parts else []

    final_model_details_for_role_check = get_model_config(llm_config_to_use)
    prompt_is_user_role = is_using_o1_mini_model(final_model_details_for_role_check) # Pass the config used for LLM

    llm_messages_for_invoke: List[BaseMessage] = []
    if prompt_is_user_role:
        llm_messages_for_invoke.append(HumanMessage(content=formatted_prompt))
    else:
        llm_messages_for_invoke.append(SystemMessage(content=formatted_prompt))

    llm_messages_for_invoke.extend(context_docs_messages_for_llm)
    llm_messages_for_invoke.append(recent_human_message)

    response_ai_message = await llm.ainvoke(llm_messages_for_invoke, {"run_name": "update_artifact_llm_call"})
    updated_highlight_content = response_ai_message.content
    if not isinstance(updated_highlight_content, str):
        updated_highlight_content = str(updated_highlight_content)

    full_code_before_highlight = current_code_artifact.code[:highlighted_code.startCharIndex]
    full_code_after_highlight = current_code_artifact.code[highlighted_code.endCharIndex:]

    entire_updated_code = f"{full_code_before_highlight}{updated_highlight_content}{full_code_after_highlight}"

    existing_artifact_v3: Optional[ArtifactV3] = state.get("artifact")
    if not existing_artifact_v3: # Should be caught by earlier check
         error_message = AIMessage(content="Critical error: Original artifact missing during update_artifact.")
         return {"messages": [error_message], "_messages": state.get("_messages", []) + [error_message]} # type: ignore

    new_content_item = ArtifactCodeV3(
        index=len(existing_artifact_v3.contents),
        type=ArtifactType.CODE,
        title=current_code_artifact.title,
        language=current_code_artifact.language,
        code=entire_updated_code
    )

    updated_artifact_contents = list(existing_artifact_v3.contents) + [new_content_item]
    final_artifact_v3 = ArtifactV3(
        currentIndex=len(updated_artifact_contents) - 1,
        contents=updated_artifact_contents
    )

    return_dict: Dict[str, Any] = {
        "artifact": final_artifact_v3,
        "highlightedCode": None # Reset the trigger
    }
    # This node, per TS, does not add its own AIMessage about the update to the chat.
    # It just updates the artifact and resets highlightedCode.
    return return_dict


UPDATE_HIGHLIGHTED_TEXT_NODE_PROMPT = """You are an expert AI writing assistant, tasked with rewriting some text a user has selected. The selected text is nested inside a larger 'block'. You should always respond with ONLY the updated text block in accordance with the user's request.
You should always respond with the full markdown text block, as it will simply replace the existing block in the artifact.
The blocks will be joined later on, so you do not need to worry about the formatting of the blocks, only make sure you keep the formatting and structure of the block you are updating.

# Selected text
{highlightedText}

# Text block
{textBlocks}

Your task is to rewrite the sourounding content to fulfill the users request. The selected text content you are provided above has had the markdown styling removed, so you can focus on the text itself.
However, ensure you ALWAYS respond with the full markdown text block, including any markdown syntax.
NEVER wrap your response in any additional markdown syntax, as this will be handled by the system. Do NOT include a triple backtick wrapping the text block, unless it was present in the original text block.
You should NOT change anything EXCEPT the selected text. The ONLY instance where you may update the sourounding text is if it is necessary to make the selected text make sense.
You should ALWAYS respond with the full, updated text block, including any formatting, e.g newlines, indents, markdown syntax, etc. NEVER add extra syntax or formatting unless the user has specifically requested it.
If you observe partial markdown, this is OKAY because you are only updating a partial piece of the text.

Ensure you reply with the FULL text block, including the updated selected text. NEVER include only the updated selected text, or additional prefixes or suffixes."""


async def update_highlighted_text(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: update_highlighted_text")
    if config is None: config = {}
    graph_config = config.get("configurable", {}) # Use 'configurable' part for helpers

    original_model_details = get_model_config(graph_config)
    model_provider = original_model_details.get("model_provider", "").lower()
    model_name = original_model_details.get("model_name", "").lower()

    llm_config_to_use = graph_config
    capable_models_for_update = ["claude-3-5-sonnet", "gpt-4o", "gpt-4.1", "gpt-4-turbo"]
    is_openai_provider = "openai" in model_provider
    is_anthropic_provider = "anthropic" in model_provider

    is_current_model_capable = False
    if is_openai_provider or is_anthropic_provider:
        if any(cap_model in model_name for cap_model in capable_models_for_update):
            is_current_model_capable = True

    if not is_current_model_capable:
        print(f"Original model {model_name} (provider: {model_provider}) not deemed suitable for precise text update. Overriding to gpt-4o.")
        override_configurable = graph_config.copy()
        override_configurable["customModelName"] = "gpt-4o"
        if "modelConfig" not in override_configurable or not isinstance(override_configurable.get("modelConfig"), dict):
            override_configurable["modelConfig"] = {"provider": "openai"}
        else:
            override_configurable["modelConfig"]["provider"] = "openai"

        # Ensure temperature 0.0 is set for the override
        if "temperature" in override_configurable.get("modelConfig", {}):
             override_configurable["modelConfig"]["temperature"] = 0.0

        llm_config_to_use = override_configurable
        llm = await get_model_from_config(llm_config_to_use, extra={"temperature": 0.0})
    else:
        llm = await get_model_from_config(llm_config_to_use, extra={"temperature": 0.0})

    current_artifact_content_model = get_artifact_content(state.get("artifact"))
    if not current_artifact_content_model:
        error_msg = "Cannot update highlighted text: No artifact found in state."
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore

    if current_artifact_content_model.type != ArtifactType.TEXT:
        error_msg = f"Cannot update highlighted text: Artifact is not text-based (type: {current_artifact_content_model.type})."
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore

    current_markdown_artifact: ArtifactMarkdownV3 = current_artifact_content_model # type: ignore

    highlighted_text_details_dict = state.get("highlightedText")
    if not highlighted_text_details_dict:
        error_msg = "Cannot update highlighted text: No highlightedText details found in state."
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore

    from app.schemas.common import TextHighlight # Import here
    try:
        highlighted_text_info = TextHighlight(**highlighted_text_details_dict)
    except Exception as e:
        error_msg = f"Invalid highlightedText format in state: {e}"
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore

    formatted_prompt = UPDATE_HIGHLIGHTED_TEXT_NODE_PROMPT.format(
        highlightedText=highlighted_text_info.selectedText,
        textBlocks=highlighted_text_info.markdownBlock
    )

    recent_human_message = next((msg for msg in reversed(state.get("_messages", [])) if msg.type == "human"), None)
    if not recent_human_message:
        error_msg = "Cannot update highlighted text: No recent human message found for context."
        return {"messages": [AIMessage(content=error_msg)], "_messages": state.get("_messages", []) + [AIMessage(content=error_msg)]} # type: ignore

    context_document_llm_parts = await create_context_document_messages(graph_config)
    context_docs_messages_for_llm = [HumanMessage(content=context_document_llm_parts, additional_kwargs={"oc_hide_from_ui": True})] if context_document_llm_parts else []

    final_model_details_for_role_check = get_model_config(llm_config_to_use)
    prompt_is_user_role = is_using_o1_mini_model(final_model_details_for_role_check)

    llm_messages_for_invoke: List[BaseMessage] = []
    if prompt_is_user_role:
        llm_messages_for_invoke.append(HumanMessage(content=formatted_prompt))
    else:
        llm_messages_for_invoke.append(SystemMessage(content=formatted_prompt))

    llm_messages_for_invoke.extend(context_docs_messages_for_llm)
    llm_messages_for_invoke.append(recent_human_message)

    response_ai_message = await llm.ainvoke(llm_messages_for_invoke, {"run_name": "update_highlighted_text_llm_call"})
    updated_markdown_block_response = response_ai_message.content
    if not isinstance(updated_markdown_block_response, str):
        updated_markdown_block_response = str(updated_markdown_block_response)

    # Reconstruct the full markdown
    if highlighted_text_info.markdownBlock not in highlighted_text_info.fullMarkdown:
        print(f"Warning: Original markdownBlock (length {len(highlighted_text_info.markdownBlock)}) not found in fullMarkdown (length {len(highlighted_text_info.fullMarkdown)}) from TextHighlight. This may lead to incorrect replacement.")
        # Fallback: attempt replacement, but it might fail or replace wrong part.
        # A more robust solution might involve diffing or more advanced string matching if this is common.
        # For now, proceed with replace, but log the warning.
        # If block is truly not there, this replace will do nothing.
        new_full_artifact_markdown = highlighted_text_info.fullMarkdown.replace(
            highlighted_text_info.markdownBlock,
            updated_markdown_block_response,
            1 # Replace only the first occurrence, just in case block is repeated
        )
        if new_full_artifact_markdown == highlighted_text_info.fullMarkdown and updated_markdown_block_response != highlighted_text_info.markdownBlock :
             print("Fallback replacement failed. Appending new block to full markdown.")
             # If replace had no effect and blocks are different, append. This is a last resort.
             new_full_artifact_markdown = highlighted_text_info.fullMarkdown + "\n\n" + updated_markdown_block_response

    else:
        new_full_artifact_markdown = highlighted_text_info.fullMarkdown.replace(
            highlighted_text_info.markdownBlock,
            updated_markdown_block_response,
            1 # Replace only the first occurrence
        )

    existing_artifact_v3: Optional[ArtifactV3] = state.get("artifact")
    if not existing_artifact_v3:
         error_message = AIMessage(content="Critical error: Original artifact missing during highlighted text update.")
         return {"messages": [error_message], "_messages": state.get("_messages", []) + [AIMessage(content=error_message)]} # type: ignore

    new_content_item = ArtifactMarkdownV3(
        index=len(existing_artifact_v3.contents),
        type=ArtifactType.TEXT,
        title=current_markdown_artifact.title,
        fullMarkdown=new_full_artifact_markdown
    )

    updated_artifact_contents = list(existing_artifact_v3.contents) + [new_content_item]
    final_artifact_v3 = ArtifactV3(
        currentIndex=len(updated_artifact_contents) - 1,
        contents=updated_artifact_contents
    )

    return_dict: Dict[str, Any] = {
        "artifact": final_artifact_v3,
        "highlightedText": None
    }
    return return_dict

async def update_highlighted_text(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # type: ignore
    print("Executing Node: update_highlighted_text")
    new_ai_message = AIMessage(content="Highlighted text update placeholder.")
    return {
        "messages": [new_ai_message],
        "_messages": state.get("_messages", []) + [new_ai_message],
        # "artifact": updated_artifact, "highlightedText": None
    }

async def rewrite_artifact_theme(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: rewrite_artifact_theme")
    new_ai_message = AIMessage(content="Artifact theme rewrite placeholder.")
    return {
        "messages": [new_ai_message],
        "_messages": state.get("_messages", []) + [new_ai_message],
        # "artifact": updated_artifact, "language": None, etc.
    }

async def rewrite_code_artifact_theme(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: rewrite_code_artifact_theme")
    new_ai_message = AIMessage(content="Code artifact theme rewrite placeholder.")
    return {
        "messages": [new_ai_message],
        "_messages": state.get("_messages", []) + [new_ai_message],
        # "artifact": updated_artifact, "addComments": None, etc.
    }

async def custom_action(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: custom_action")
    new_ai_message = AIMessage(content="Custom action placeholder.")
    return {
        "messages": [new_ai_message],
        "_messages": state.get("_messages", []) + [new_ai_message],
        # "artifact": updated_artifact, "customQuickActionId": None
    }

async def web_search_node(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: web_search_node (placeholder for webSearchGraph call)")
    # This node would typically:
    # 1. Get the actual user query from the last HumanMessage in state.get("_messages", [])
    # 2. Invoke a separate web search agent/graph with that query.
    # 3. The result of that sub-graph (list of SearchResult) comes back.
    # 4. Update the state with webSearchResults and webSearchEnabled=False.
    # It does NOT directly determine the next node for the main graph here.
    # That's done by routePostWebSearchNode.
    mock_search_results = [{"page_content": "Mock search result from web_search_node", "metadata": {"title": "Mock Site", "url":"http://mock.com"}}]
    return {
        "webSearchResults": mock_search_results,
        "webSearchEnabled": False # Critically, turn this off
    }

async def route_post_web_search_node(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: route_post_web_search_node")
    # This node decides where to go AFTER web_search_node has populated webSearchResults.
    # It also formats the webSearchResults into an AIMessage for the LLM.

    from app.schemas.common import SearchResult # For type hint
    web_results: Optional[List[SearchResult]] = state.get("webSearchResults")

    next_node_decision = "generateArtifact" if not state.get("artifact") else "rewriteArtifact"

    if not web_results: # Should not happen if web_search_node ran and found results
        print("Warning: No web search results found in route_post_web_search_node.")
        return {"next_node": next_node_decision} # No message to add about web results

    # Format web results into a message. This message goes into _messages for LLM context.
    # It does NOT go into the main 'messages' for UI display.
    # TODO: Port create_ai_message_from_web_results from TS if a complex format is needed.
    # For now, a simple concatenation:
    formatted_results_str = "\n".join(
        [f"Title: {res.get('metadata', {}).get('title', 'N/A')}\nURL: {res.get('metadata', {}).get('url', 'N/A')}\nContent Snippet: {res.get('page_content', '')[:200]}..."
         for res in web_results]
    )
    web_search_ai_message = AIMessage(
        content=f"Here are some web search results relevant to your query:\n{formatted_results_str}",
        additional_kwargs={"oc_hide_from_ui": True} # This message is for LLM context, not direct display
    )

    # Get current _messages, append the new web search AI message
    current_internal_messages = list(state.get("_messages", []))
    current_internal_messages.append(web_search_ai_message)

    return {
        "next_node": next_node_decision,
        "_messages": current_internal_messages, # Update _messages with the context from web search
        # webSearchEnabled should have been set to False by web_search_node
        # webSearchResults are kept for now, cleanStateNode might clear them later or they might be used by generate/rewrite.
    }


async def generate_followup(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: generate_followup")
    new_ai_message = AIMessage(content="Follow-up placeholder.")
    return {
        "messages": [new_ai_message],
        "_messages": state.get("_messages", []) + [new_ai_message]
    }

async def reflect_node(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: reflect_node")
    # This node might update 'reflections' in state based on state.get("messages")
    # This main graph node will now be a placeholder for triggering the reflection sub-graph.
    # from app.agents.reflection_graph.graph import reflection_graph_app # For later direct call
    # from fastapi import BackgroundTasks # Example for background execution

    print("Executing Node: reflect_node (Main Graph - Placeholder for triggering reflection)")
    if config is None: config = {}
    # graph_config is the full RunnableConfig of the main graph

    configurable_main_graph = config.get("configurable", {})
    main_assistant_id = configurable_main_graph.get("assistant_id") # Or other key if named differently

    if not main_assistant_id:
        print("Warning: 'assistant_id' not found in main graph's config. Cannot trigger reflection graph without it.")
        return {}

    # Prepare input for the reflection_graph_app
    # The reflection graph expects 'messages' and 'artifact' in its state.
    # It uses _messages from the main graph as its 'messages' input.
    reflection_graph_input = {
        "messages": state.get("_messages", []),
        "artifact": state.get("artifact")
    }

    # Prepare the config for the reflection_graph_app run
    # Pass the main assistant_id as 'open_canvas_assistant_id' for the reflection graph's context.
    # Also, if the reflection graph needs to access the same checkpointer/store, it should be passed.
    reflection_graph_run_config = {
        "configurable": {
            "open_canvas_assistant_id": main_assistant_id,
            # Example: if checkpointer is part of config and needed by reflection_graph for its store operations
            # "checkpointer": config.get("checkpointer"),
        },
        # Ensure recursion limit is handled if calling graphs from graphs
        "recursion_limit": config.get("recursion_limit", 25) -1 if config.get("recursion_limit") else 24,
    }

    print(f"Reflect Node (Main Graph): Would trigger 'reflection_graph_app'.")
    print(f"  - Reflection Input (first 100 chars of messages): {{'messages': {str(reflection_graph_input['messages'])[:100]}..., 'artifact': {str(reflection_graph_input['artifact'])[:100]}...}}")
    print(f"  - Reflection Config: {{'configurable': {{'open_canvas_assistant_id': '{main_assistant_id}'}}}}")

    # TODO - Phase 2: Implement actual call to reflection_graph_app.
    # This could be a direct `await reflection_graph_app.ainvoke(...)` if synchronous behavior is acceptable,
    # or a background task if the main flow shouldn't wait for reflections to complete.
    # Example (synchronous call - blocks the main graph flow until reflection is done):
    # try:
    #     await reflection_graph_app.ainvoke(reflection_graph_input, config=reflection_graph_run_config)
    #     print("Reflection graph invocation completed.")
    # except Exception as e:
    #     print(f"Error invoking reflection graph: {e}")
    #
    # Example for background task (requires FastAPI's BackgroundTasks or similar mechanism):
    # This assumes `background_tasks` is available in this scope, e.g., via FastAPI dependency injection
    # if this node were part of a FastAPI route handler that creates and runs the graph.
    # If running graph outside FastAPI, another backgrounding method (e.g., asyncio.create_task, Celery) needed.
    #
    # if "background_tasks" in config.get("configurable", {}): # Check if available
    #    bg_tasks = config["configurable"]["background_tasks"]
    #    bg_tasks.add_task(reflection_graph_app.ainvoke, reflection_graph_input, config=reflection_graph_run_config)
    #    print("Reflection graph task added to background.")
    # else:
    #    print("No background task runner found in config, reflection would be synchronous or not run.")

    return {} # reflect_node in main graph doesn't directly change main graph state; reflection graph updates store.

async def clean_state_node(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: clean_state_node")
    # Returns a dictionary representing the fields to reset to their defaults.
    # LangGraph will merge this partial state into the main state.
    # It should only return fields that need to be reset.
    # For example, 'messages' and '_messages' are usually preserved by cleanState in TS.

    fields_to_reset = DEFAULT_INPUTS_PYTHON.copy()
    del fields_to_reset["messages"] # Don't clear conversation history
    del fields_to_reset["_messages"] # Don't clear internal message history
    del fields_to_reset["artifact"] # Don't clear artifact unless specified by logic

    # Specific logic from TS cleanState:
    # webSearchEnabled is reset.
    # webSearchResults are reset.
    # highlightedCode, highlightedText are reset.
    # customQuickActionId, language, artifactLength, etc. are reset.

    # This means DEFAULT_INPUTS_PYTHON should accurately reflect all clearable fields.
    return fields_to_reset


# --- Web Search Sub-Graph Integration ---
# Placeholder for a more sophisticated message creation from web results
def create_ai_message_from_web_results_placeholder(results: List[Dict[str,Any]]) -> AIMessage:
    if not results:
        return AIMessage(content="No web search results found or an error occurred during search.", additional_kwargs={"webSearchStatus": "error_or_empty"})

    # Simplified content for now
    content_parts = ["Web search results:\n"]
    for i, r in enumerate(results[:3]): # Show top 3 results
        title = r.get('metadata',{}).get('title', 'N/A')
        url = r.get('metadata',{}).get('url', 'N/A')
        # Snippet of page_content could be added if desired
        # page_snippet = r.get('page_content', '')[:100] + "..." if r.get('page_content') else ""
        content_parts.append(f"{i+1}. {title}: {url}")

    return AIMessage(
        content="\n".join(content_parts),
        additional_kwargs={
            "webSearchResults": results, # Attach full results for potential later use
            "webSearchStatus": "done",
            "oc_hide_from_ui": True # This message is for LLM context, not direct UI display
            }
        )

async def web_search_node(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: web_search_node (Main Graph - calling WebSearchGraph)")
    if config is None: config = {}
    # graph_config is the full RunnableConfig for the main graph.
    # The web_search_graph_app might not need all of it, but passing it for consistency.
    # Key parts like API keys (EXA_API_KEY) are expected as environment variables by the sub-graph's nodes.

    # Input for web_search_graph is 'messages' (which will be state._messages from main graph)
    # Ensure we are passing the correct message list for web search context.
    # TS uses state._messages for the sub-graph invocation.
    web_search_input = {"messages": state.get("_messages", [])}

    from app.agents.web_search_graph.graph import web_search_graph_app # Import the compiled graph

    try:
        # Invoke the sub-graph. It will return its entire state (WebSearchState).
        web_search_result_state: WebSearchState = await web_search_graph_app.ainvoke(web_search_input, config=config) # type: ignore

        # Extract webSearchResults to be merged into OpenCanvasState.
        # Also, ensure webSearchEnabled is reset.
        return {
            "webSearchResults": web_search_result_state.get("webSearchResults", []),
            "webSearchEnabled": False
        }
    except Exception as e:
        print(f"Error invoking web_search_graph_app: {e}")
        # Ensure webSearchEnabled is reset even on error.
        return {"webSearchResults": [], "webSearchEnabled": False}


async def route_post_web_search_node(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: route_post_web_search_node (Main Graph)")

    next_node_val = "generateArtifact" # Default if no artifact exists
    current_artifact_v3 = state.get("artifact")
    if current_artifact_v3 and current_artifact_v3.contents:
        next_node_val = "rewriteArtifact" # Default if artifact exists

    web_search_results = state.get("webSearchResults")

    messages_to_add_to_internal_history = []
    if web_search_results:
        ai_message_with_results = create_ai_message_from_web_results_placeholder(web_search_results)
        messages_to_add_to_internal_history.append(ai_message_with_results)

    update_dict: Dict[str, Any] = {
        "next_node": next_node_val,
        # webSearchEnabled was already set to False by web_search_node.
        # No need to set it again unless there's a specific reason.
    }
    if messages_to_add_to_internal_history:
        # This should append to _messages. The graph reducer for _messages needs to handle this.
        # If _messages is a list, and the reducer appends, this is fine.
        # The TS reducer for _messages prepends: (right ?? []).concat(left ?? [])
        # So, to achieve append: current _messages + new messages
        # However, the node output is merged. If we return `{"_messages": new_list}`,
        # it will depend on the operator.add or custom reducer behavior.
        # For now, returning just the new messages to be added.
        # The reducer for _messages should be `lambda existing, new: (existing or []) + (new or [])` for append.
        # Or, if it's the TS-like prepend: `lambda existing, new: (new or []) + (existing or [])`
        # If returning just `messages_to_add_to_internal_history`, it implies the reducer for `_messages`
        # should be configured to append this to the existing `_messages` list.
        # Let's assume the reducer for _messages will append if it gets a list.
        update_dict["_messages"] = messages_to_add_to_internal_history

    return update_dict

# This conditional edge function remains mostly the same, relying on 'next_node' in state.
def route_post_web_search_conditional(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> str:
    print(f"Executing Conditional Edge: route_post_web_search_conditional, next_node is {state.get('next_node')}")
    return state.get("next_node", END) # Default to END if not set by the node


async def generate_title_node(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: generate_title_node (Main Graph - Placeholder for triggering title generation)")
    if config is None: config = {}
    # graph_config is the full RunnableConfig of the main graph

    configurable_main_graph = config.get("configurable", {})
    # The main graph's thread_id is needed by the title sub-graph
    main_thread_id = configurable_main_graph.get("thread_id")

    if not main_thread_id:
        print("Warning: 'thread_id' not found in main graph's config. Cannot trigger title generation graph without it.")
        return {}

    # Safeguard from TS version: only generate title for very new conversations
    # TS uses `state.messages.length > 2`. `messages` in OpenCanvasState refers to the user-facing messages.
    if len(state.get("messages", [])) > 2: # Check user-facing messages
        print("Skipping title generation: Not considered a new enough conversation (more than 2 user-facing messages).")
        return {}

    # Prepare input for the thread_title_graph_app
    # The title graph expects 'messages' (for conversation context) and 'artifact'.
    # TS uses `state.messages` (user-facing) not `state._messages` (internal) for title generation context.
    title_graph_input = {
        "messages": state.get("messages", []),
        "artifact": state.get("artifact")
    }

    # Prepare the config for the thread_title_graph_app run
    # Pass the main thread_id as 'open_canvas_thread_id' for the title graph's context.
    title_graph_run_config = {
        "configurable": {
            "open_canvas_thread_id": main_thread_id,
            # If the sub-graph needed other specific configs, pass them here.
        },
        # Ensure recursion limit is handled if calling graphs from graphs
        "recursion_limit": config.get("recursion_limit", 25) -1 if config.get("recursion_limit") else 24,
    }

    print(f"Generate Title Node (Main Graph): Would trigger 'thread_title_graph_app'.")
    print(f"  - Title Graph Input (first 100 chars of messages): {{'messages': {str(title_graph_input['messages'])[:100]}..., 'artifact': {str(title_graph_input['artifact'])[:100]}...}}")
    print(f"  - Title Graph Config: {{'configurable': {{'open_canvas_thread_id': '{main_thread_id}'}}}}")

    # TODO - Phase 2: Implement actual call to thread_title_graph_app.
    # This could be a direct `await thread_title_graph_app.ainvoke(...)` if synchronous behavior is acceptable,
    # or a background task.
    # Example (synchronous call):
    # from app.agents.thread_title_graph.graph import thread_title_graph_app
    # try:
    #     await thread_title_graph_app.ainvoke(title_graph_input, config=title_graph_run_config)
    #     print("Thread title graph invocation completed.")
    # except Exception as e:
    #     print(f"Error invoking thread title graph: {e}")

    return {} # generate_title_node in main graph doesn't directly change main graph state; sub-graph updates external metadata.


async def summarizer_node(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: summarizer_node")
    # This node summarizes state.get("_messages", [])
    # The new summarized list becomes the new state._messages
    print("Executing Node: summarizer_node (Main Graph - calling SummarizerGraph)")
    if config is None: config = {}
    # graph_config is the full RunnableConfig of the main graph.

    messages_to_summarize = state.get("_messages", [])
    if not messages_to_summarize:
        print("Summarizer node (Main Graph): No messages to summarize.")
        return {} # No change to state if no messages

    # Import the summarizer sub-graph application
    from app.agents.summarizer_graph.graph import summarizer_graph_app
    from app.agents.summarizer_graph.state import SummarizerGraphState # For type hint

    summarizer_graph_input = {"messages_in": messages_to_summarize}

    # The summarizer_graph_app itself doesn't require special 'configurable' values like thread_id
    # unless its internal nodes were to use them (currently they don't, model is hardcoded).
    # We pass the main graph's config through, in case any future enhancements to the
    # summarizer sub-graph might need it (e.g., for model selection via config).
    summarizer_run_config = config

    print(f"Summarizer Node (Main Graph): Invoking 'summarizer_graph_app'.")
    try:
        result_state: SummarizerGraphState = await summarizer_graph_app.ainvoke(
            summarizer_graph_input,
            config=summarizer_run_config
        ) # type: ignore

        new_summarized_message = result_state.get("summarized_message_out")

        if new_summarized_message:
            # This new message, marked with OC_SUMMARIZED_MESSAGE_KEY,
            # will trigger the custom reducer for '_messages' in the main graph
            # to replace the entire history with this single summary message.
            print("Summarizer node (Main Graph): Received summary from sub-graph. Updating _messages.")
            return {"_messages": [new_summarized_message]}
        else:
            print("Summarizer node (Main Graph): Sub-graph did not return a summarized message.")
            return {} # No change if summarization failed or returned None

    except Exception as e:
        print(f"Error invoking summarizer_graph_app: {e}")
        # Optionally, return an error message to be added to the chat or log.
        # For now, returning no change to avoid breaking the main flow.
        return {}


# --- Conditional Routing Functions (Updated signatures) ---
from .reducers import update_internal_messages, update_user_facing_messages # Import reducers

def route_node(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    # This conditional edge function receives the *entire current state* of the graph.
    # The preceding node (e.g., generate_path) must have updated the 'next_node' field in the state.
    print(f"Conditional Edge: route_node, current state's next_node is '{state.get('next_node')}'")
    next_node_val = state.get("next_node") # Read 'next_node' from the current graph state
    if not next_node_val:
        print("Warning: 'next_node' is not set in graph state for route_node. Defaulting to END.")
        return END
    return next_node_val


def conditionally_generate_title(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> Literal["generateTitleNode", "summarizerNode", END]: # type: ignore
    print("Conditional Edge: conditionally_generate_title")
    if len(state.get("messages", [])) <= 2: # User-facing messages count
         return "generateTitleNode"
    # Check if summarization is needed based on _messages length/tokens
    # This is where simple_token_calculator logic would effectively be.
    # Using CHARACTER_MAX from TS as an example.
    CHARACTER_MAX = 300000  # Example token/char count from TS
    current_total_chars = sum(len(get_string_from_content(m.content)) for m in state.get("_messages", []))
    if current_total_chars > CHARACTER_MAX:
        return "summarizerNode"
    return END

# simple_token_calculator is effectively merged into conditionally_generate_title's logic.
# No need for a separate conditional edge if conditionally_generate_title handles both paths.

def route_post_web_search_conditional(state: OpenCanvasState, config: Optional[Dict[str, Any]] = None) -> str:
    # This conditional edge follows 'route_post_web_search_node'.
    # That node should have set 'next_node' in its output, which updates the graph state.
    print(f"Conditional Edge: route_post_web_search_conditional, current state's next_node is '{state.get('next_node')}'")
    return state.get("next_node", END) # Read 'next_node' from graph state


# --- Graph Definition (Placeholders for node names, ensure they match .add_node calls) ---
# builder = StateGraph(OpenCanvasState) # Old placeholder
# --- Graph Definition ---
builder = StateGraph(OpenCanvasState,
                     channels={
                         "_messages": update_internal_messages,
                         "messages": update_user_facing_messages
                         # Other keys like 'artifact', 'highlightedCode', etc., will use the default
                         # 'last write wins' reducer, which is usually what's needed for them.
                     })

# Adding nodes (ensure names match strings in conditional edges)
builder.add_node("generatePath", generate_path)
# builder.add_node("replyToGeneralInput", reply_to_general_input)
# builder.add_node("generateArtifact", generate_artifact)
# builder.add_node("updateArtifact", update_artifact)
# builder.add_node("updateHighlightedText", update_highlighted_text)
# builder.add_node("rewriteArtifactTheme", rewrite_artifact_theme)
# builder.add_node("rewriteCodeArtifactTheme", rewrite_code_artifact_theme)
# builder.add_node("customAction", custom_action)
# builder.add_node("webSearch", web_search_node)
# builder.add_node("routePostWebSearchNode", route_post_web_search_node) # Node before conditional edge
# builder.add_node("generateFollowup", generate_followup)
builder.add_node("reflectNode", reflect_node)
builder.add_node("cleanStateNode", clean_state_node)
builder.add_node("generateTitleNode", generate_title_node)
builder.add_node("summarizerNode", summarizer_node)

# Entry point
builder.add_edge(START, "generatePath")

# Conditional routing from generatePath (based on its 'next_node' output from state)
builder.add_conditional_edges(
    "generatePath",
    route_node,
    {
        "updateArtifact": "updateArtifact",
        "rewriteArtifactTheme": "rewriteArtifactTheme",
        "rewriteCodeArtifactTheme": "rewriteCodeArtifactTheme",
        "replyToGeneralInput": "replyToGeneralInput",
        "generateArtifact": "generateArtifact",
        "rewriteArtifact": "rewriteArtifact",
        "customAction": "customAction",
        "updateHighlightedText": "updateHighlightedText",
        "webSearch": "webSearchNode", # Matched node name
        END: END
    }
)

# Direct Edges after artifact main operations to followup
artifact_op_nodes = [
    "generateArtifact", "updateArtifact", "updateHighlightedText",
    "rewriteArtifact", "rewriteArtifactTheme", "rewriteCodeArtifactTheme", "customAction"
]
for node_name in artifact_op_nodes:
    builder.add_edge(node_name, "generateFollowup")

# Web search flow
builder.add_edge("webSearchNode", "routePostWebSearchNode") # Matched node name

# Conditional routing after web search results are processed by routePostWebSearchNode
builder.add_conditional_edges(
    "routePostWebSearchNode",
    route_post_web_search_conditional,
    {
        "generateArtifact": "generateArtifact",
        "rewriteArtifact": "rewriteArtifact", # TS had 'updateArtifact', assuming 'rewriteArtifact' is more general
        END: END
    }
)

# Other direct edges
builder.add_edge("replyToGeneralInput", "cleanStateNode") # Matched node name
builder.add_edge("generateFollowup", "reflectNode")    # Matched node name
builder.add_edge("reflectNode", "cleanStateNode")       # Matched node name

# Conditional routing from cleanStateNode
builder.add_conditional_edges(
    "cleanStateNode", # Matched node name
    conditionally_generate_title,
    {
        "generateTitleNode": "generateTitleNode", # Matched node name
        "summarizerNode": "summarizerNode",       # Matched node name
        END: END
    }
)

# Final terminal edges
builder.add_edge("generateTitleNode", END) # Matched node name
builder.add_edge("summarizerNode", END)    # Matched node name

# Compile the graph
# Checkpointer will be added later during FastAPI integration or main app setup.
open_canvas_graph_app = builder.compile(checkpointer=None)
print("OpenCanvas Graph compiled successfully (without checkpointer).")

# Example test execution (commented out, for local testing)
# async def main_test_graph():
#     from langchain_core.messages import HumanMessage
#     import asyncio

#     inputs = {"messages": [HumanMessage(content="Hello! Generate an artifact about a sunny day.")]}
#     config = {
#         "configurable": {
#             "assistant_id": "test-assistant-id-123",
#             "thread_id": "test-thread-id-456",
#             "user_id": "test-user-789", # Example, if needed by any node/helper
#             # modelConfig might be needed if not hardcoded in get_model_from_config fallbacks
#             "modelConfig": {"provider": "openai", "customModelName": "gpt-4o-mini"}
#         }
#     }
#     print(f"Invoking graph with inputs: {inputs}")
#     print(f"Using config: {config}")

#     async for event in open_canvas_graph_app.astream_events(inputs, config=config, version="v2"):
#         print("\n--- Event ---")
#         print(f"Event Type: {event['event']}")
#         print(f"Node Name: {event['name']}")
#         # print(f"Node Input: {event.get('data', {}).get('input')}") # Can be verbose
#         # print(f"Node Output: {event.get('data', {}).get('output')}") # Can be verbose

#         if event['event'] == 'on_chain_end' and event['name'] == 'generatePath':
#             print(f"Output of generatePath: {event['data'].get('output')}")

#         if event['event'] == 'on_chain_stream' and event['name'] == 'replyToGeneralInput':
#             # For streaming tokens from replyToGeneralInput
#             chunk = event['data'].get('chunk')
#             if chunk:
#                 if hasattr(chunk, 'content'):
#                     print(f"Streaming output (replyToGeneralInput): {chunk.content}", end="", flush=True)
#                 elif isinstance(chunk, dict) and 'content' in chunk:
#                      print(f"Streaming output (replyToGeneralInput): {chunk['content']}", end="", flush=True)


# if __name__ == "__main__":
#    # Ensure event loop is running for asyncio if testing directly
#    # asyncio.run(main_test_graph())
#    pass
