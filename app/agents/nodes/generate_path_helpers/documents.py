# app/agents/nodes/generate_path_helpers/documents.py
import uuid
from typing import List, Optional, Tuple, Union, Any, Dict

from langchain_core.messages import BaseMessage, HumanMessage # type: ignore
# from langchain_core.messages import AIMessage, SystemMessage, ToolMessage # Add if needed

from app.utils.langchain_helpers import (
    create_context_document_messages,
    get_model_config,
    convert_pdf_to_text, # This is synchronous
    ContextDocument,
)
from app.utils.text_processing import get_string_from_content # For processing message content

OC_HIDE_FROM_UI_KEY = "oc_hide_from_ui"

class RemoveMessage(BaseMessage):
    """Represents a message to be removed from state by its ID."""
    # This is a conceptual representation. LangGraph state updates typically involve
    # returning a new list of messages for the 'messages' key in the state,
    # or using specific update mechanisms if available for message lists.
    # For now, if a node needs to signal a removal, it might need to be handled
    # by the node that aggregates messages into the final state.
    # Alternatively, the graph could have a mechanism to process a list of "operations"
    # like [RemoveMessage(id="123"), AddMessage(HumanMessage(...))]
    # For simplicity, a node modifying messages might return the full new list of messages.
    # However, to match the TS structure which implies targeted removal and addition,
    # this class is kept as a placeholder for how such an operation might be indicated.
    id_to_remove: str
    # 'type' is a standard field in BaseMessage that LangGraph might use internally.
    # We give it a custom value to distinguish it if needed, though it might not be directly used by LangGraph's core message handling.
    type: str = "remove_message_operation"

    def __init__(self, id_to_remove: str, **kwargs: Any):
        # BaseMessage requires 'content'. For an operation like this, it's not primary.
        super().__init__(content=f"Operation to remove message with ID: {id_to_remove}", **kwargs)
        self.id_to_remove = id_to_remove


async def convert_context_document_to_human_message(
    messages: List[BaseMessage],
    config: Dict[str, Any], # LangGraphRunnableConfig's 'configurable' field
) -> Optional[HumanMessage]:
    if not messages:
        return None
    last_message = messages[-1]

    documents_data = last_message.additional_kwargs.get("documents") if last_message.additional_kwargs else None

    if not documents_data or not isinstance(documents_data, list):
        return None

    try:
        # Ensure that ContextDocument can be initialized from dicts in documents_data
        documents = [ContextDocument(**doc_data) for doc_data in documents_data if isinstance(doc_data, dict)]
    except Exception as e: # Catch Pydantic validation errors or others
        print(f"Error parsing 'documents' from last message's additional_kwargs: {e}")
        return None

    if not documents:
        return None

    # create_context_document_messages expects the 'configurable' part of the config
    context_message_parts = await create_context_document_messages(config, documents) # type: ignore

    if not context_message_parts:
        return None

    return HumanMessage(
        id=str(uuid.uuid4()), # Generate a new ID for this derived message
        content=context_message_parts,
        additional_kwargs={
            OC_HIDE_FROM_UI_KEY: True,
            "original_message_id": last_message.id # Keep track of source message
        },
    )

async def fix_misformatted_context_doc_message(
    message: HumanMessage, # The message potentially containing misformatted context docs
    config: Dict[str, Any], # LangGraphRunnableConfig's 'configurable' field
) -> Optional[List[Union[RemoveMessage, HumanMessage]]]:
    # This function's purpose was to correct messages if they were formatted
    # by a different LLM provider than the one currently in use (e.g. OpenAI received Anthropic/Gemini formatted docs).
    # Given create_context_document_messages now centralizes formatting (defaulting to OpenAI style),
    # this specific transformation might be less critical if all context docs are pre-formatted
    # before being added to HumanMessage.content by convert_context_document_to_human_message.
    # However, if messages can come from other sources with provider-specific formats, it's relevant.

    if isinstance(message.content, str): # Already a simple string, no parts to fix
        return None

    # get_model_config expects the 'configurable' part of the config
    model_params = get_model_config(config)
    model_provider = model_params["model_provider"]

    new_msg_id = str(uuid.uuid4())
    changes_made = False
    new_content_parts = []

    # message.content is List[Union[str, Dict]]
    original_content_parts = message.content if isinstance(message.content, list) else [message.content]


    # This logic assumes we're converting *from* Anthropic/Gemini specific formats *to* OpenAI style if needed.
    # However, create_context_document_messages_openai (called by convert_context_document_to_human_message)
    # should already produce OpenAI-compatible format.
    # This function might be more about handling legacy formats or direct user inputs with such structures.
    # For now, let's assume it's checking for non-OpenAI formats and converting them.

    if model_provider == "openai": # If current model is OpenAI, ensure content is OpenAI-native
        for part in original_content_parts:
            if isinstance(part, dict):
                # Example: Anthropic format {type: "document", source: {type: "base64", data: "...", media_type: "application/pdf"}}
                if part.get("type") == "document" and isinstance(part.get("source"), dict):
                    source = part["source"]
                    if source.get("type") == "base64" and source.get("media_type") == "application/pdf" and "data" in source:
                        try:
                            text = convert_pdf_to_text(source["data"]) # convert_pdf_to_text is sync
                            new_content_parts.append({"type": "text", "text": f"Context from PDF: {text[:2000]}"}) # Truncate for safety
                            changes_made = True
                        except Exception as e:
                            print(f"PDF conversion error in fix_misformatted: {e}")
                            new_content_parts.append({"type": "text", "text": "[Error processing PDF content]"})
                            changes_made = True # Changed to error message
                    else: # Other "document" subtypes, pass through or adapt
                        new_content_parts.append(part)
                # Example: Gemini format {type: "application/pdf", data: "..."} -> This is not Gemini's typical format for LLM.
                # Gemini typically uses `parts: [{"fileData": {"mimeType": "application/pdf", "fileUri": "..."}}]` or inline text.
                # The schema used in create_context_document_messages_openai is more aligned with OpenAI's own.
                # This part of the original TS logic might need re-evaluation based on actual cross-provider message formats encountered.
                # For now, if a part is already {"type": "text", "text": "..."} or {"type": "image_url", ...}, it's likely fine for OpenAI.
                else:
                    new_content_parts.append(part)
            elif isinstance(part, str): # Should be wrapped in {"type": "text"} by now
                new_content_parts.append({"type": "text", "text": part})
                changes_made = True # Technically a change in format
            else:
                new_content_parts.append(part) # Unknown part, pass through

    # Add similar conversion logic if model_provider is "anthropic" and content seems to be from OpenAI/Gemini, etc.
    # elif model_provider == "anthropic":
    #     # ... check for OpenAI/Gemini specific formats and convert to Anthropic's expected format ...
    #     pass

    if changes_made:
        if not message.id:
            print("Error: Original message for fix_misformatted_context_doc_message is missing an ID.")
            # Cannot create RemoveMessage without an ID. Could raise error or return None.
            return None

        return [
            RemoveMessage(id_to_remove=message.id), # type: ignore
            HumanMessage(id=new_msg_id, content=new_content_parts, additional_kwargs=message.additional_kwargs), # type: ignore
        ]
    return None
