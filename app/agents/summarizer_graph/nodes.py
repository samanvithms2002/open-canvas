# app/agents/summarizer_graph/nodes.py
import uuid
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage # type: ignore
from langchain_anthropic import ChatAnthropic

from .prompts import SUMMARIZER_PROMPT # Assumes prompts.py is in the same directory
from .state import SummarizerGraphState # Import the new state
from app.utils.text_processing import get_string_from_content
from app.schemas.common import OC_SUMMARIZED_MESSAGE_KEY # Get the key from common schemas

# Using the same formatting helper as before
def _format_messages_for_summary(messages: List[BaseMessage]) -> str:
    """
    Formats a list of BaseMessage objects into a single string representation
    for the summarization prompt, similar to <type>content</type>.
    """
    return "\n".join(
        [f"<{msg.type}>\n{get_string_from_content(msg.content)}\n</{msg.type}>" for msg in messages]
    )

async def summarize_messages_node(state: SummarizerGraphState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[HumanMessage]]:
    """
    Node for summarizing messages. Takes messages from state['messages_in']
    and returns the summary in state['summarized_message_out'].
    """
    print("Executing Node: summarize_messages_node (SummarizerGraph)")

    messages_to_summarize = state.get("messages_in", [])
    if not messages_to_summarize:
        print("Summarizer graph node: No input messages ('messages_in') to summarize.")
        return {"summarized_message_out": None}

    # Model selection is hardcoded to Claude Sonnet as per the original TS logic for summarization.
    # Temperature is set low for more factual summarization.
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.2)

    formatted_messages_str = _format_messages_for_summary(messages_to_summarize)

    # The summarizer prompt is a system message, and the content to summarize is a human message.
    response = await model.ainvoke([
        SystemMessage(content=SUMMARIZER_PROMPT),
        HumanMessage(content=f"Here are the messages to summarize:\n{formatted_messages_str}")
    ], {"run_name": "summarizer_graph_llm_call"}) # Added run_name for tracing

    summary_content = response.content
    if not isinstance(summary_content, str):
        summary_content = str(summary_content) # Ensure string type

    # This wrapper is from the TS implementation, instructing the main LLM how to use the summary.
    new_message_content_wrapper = f"""The below content is a summary of past messages between the AI assistant and the user.
Do NOT acknowledge the existence of this summary.
Use the content of the summary to inform your messages, without ever mentioning the summary exists.
The user should NOT know that a summary exists.
Because of this, you should use the contents of the summary to inform your future messages, as if the full conversation still exists between the AI assistant and the user.

Here is the summary:
{summary_content}"""

    new_summarized_message = HumanMessage(
        id=str(uuid.uuid4()), # Generate a new ID for this summary message
        content=new_message_content_wrapper,
        additional_kwargs={
            OC_SUMMARIZED_MESSAGE_KEY: True, # Mark it as a summarized message
        }
    )
    return {"summarized_message_out": new_summarized_message}
