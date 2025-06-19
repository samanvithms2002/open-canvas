# app/agents/reducers.py
from typing import List, Optional, Union, Any
from langchain_core.messages import BaseMessage
from app.schemas.common import OC_SUMMARIZED_MESSAGE_KEY, OC_HIDE_FROM_UI_KEY # Import keys

def update_internal_messages(
    left: Optional[List[BaseMessage]],
    right: Optional[Union[BaseMessage, List[BaseMessage]]]
) -> List[BaseMessage]:
    """
    Custom reducer for the '_messages' key in OpenCanvasState.
    - If a new message (or list of messages) contains a message with
      OC_SUMMARIZED_MESSAGE_KEY, that message replaces the entire history.
    - Otherwise, new messages are appended to the existing history.
    """
    current_history = left if left is not None else []

    if right is None:
        return current_history

    new_incoming_messages: List[BaseMessage] = [right] if not isinstance(right, list) else right

    # Check if any of the new incoming messages is a summary message
    for msg in new_incoming_messages:
        # Check additional_kwargs exists and is a dict before key lookup
        if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict) and \
           OC_SUMMARIZED_MESSAGE_KEY in msg.additional_kwargs:
            # If it's a summary message, it replaces the entire history.
            # The summarizer_node is expected to return a list containing just this summary message.
            print("Reducer: Summary message detected. Replacing _messages history.")
            return new_incoming_messages

    # Default behavior: append new messages to the current history
    return current_history + new_incoming_messages

def update_user_facing_messages(
    left: Optional[List[BaseMessage]],
    right: Optional[Union[BaseMessage, List[BaseMessage]]]
) -> List[BaseMessage]:
    """
    Custom reducer for the 'messages' key (user-facing messages).
    - Filters out messages intended to be hidden from the UI.
    - Appends new visible messages.
    """
    current_history = left if left is not None else []

    if right is None:
        return current_history

    new_incoming_messages: List[BaseMessage] = [right] if not isinstance(right, list) else right

    visible_new_messages: List[BaseMessage] = []
    for msg in new_incoming_messages:
        if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict) and \
           OC_HIDE_FROM_UI_KEY in msg.additional_kwargs and msg.additional_kwargs[OC_HIDE_FROM_UI_KEY]:
            print(f"Reducer: Hiding message ID {msg.id if hasattr(msg, 'id') else 'N/A'} from user-facing chat.")
            continue # Skip hidden messages
        visible_new_messages.append(msg)

    return current_history + visible_new_messages

# Example of how to use this in StateGraph channels:
# from langgraph.graph import StateGraph
# from operator import add # For simple list append if no special logic needed for other keys
#
# channels = {
#     "_messages": update_internal_messages,
#     "messages": update_user_facing_messages,
#     "some_other_list_key": add, # operator.add for lists
#     "some_value_key": None, # Default reducer (overwrite)
# }
# workflow = StateGraph(OpenCanvasState, channels=channels)
