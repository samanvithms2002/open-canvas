# app/agents/web_search_graph/nodes.py
import os
import json # For parsing tool args if stringified
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid # For fallback ID in search_node

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage # type: ignore
from langchain_anthropic import ChatAnthropic
from langchain_exa import ExaRetriever

from .state import WebSearchState
from .schemas import ClassificationSchema
from .prompts import CLASSIFIER_PROMPT, QUERY_GENERATOR_PROMPT

from app.utils.text_processing import get_string_from_content
# Re-using _format_messages_for_summary as it's similar to TS formatMessages
# This helper was defined in summarizer_graph.nodes, need to ensure it's accessible
# For now, let's assume it's moved to a common utility or re-defined if necessary.
# To avoid direct cross-agent dependency, let's define a local version or move it to text_processing.
# For this step, defining locally for clarity.
def _format_messages_for_web_search_prompt(messages: List[BaseMessage]) -> str:
    return "\n".join(
        [f"<{msg.type}>\n{get_string_from_content(msg.content)}\n</{msg.type}>" for msg in messages]
    )

async def classify_message_node(state: WebSearchState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: classify_message (WebSearchGraph)")

    # Using ChatAnthropic as per TS, with structured output for classification
    # Model from TS: claude-3-5-sonnet-latest (formerly claude-3-sonnet)
    # Temperature 0 for consistent classification.
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        temperature=0
    ).with_structured_output(ClassificationSchema, name="classify_message_for_web_search") # type: ignore

    if not state.get("messages"):
        print("classify_message_node: No messages in state to classify.")
        return {"shouldSearch": False}

    # Get content of the latest message for classification
    latest_message_content = get_string_from_content(state["messages"][-1].content)
    if not latest_message_content.strip():
        print("classify_message_node: Latest message content is empty.")
        return {"shouldSearch": False}

    formatted_prompt = CLASSIFIER_PROMPT.format(message=latest_message_content)

    try:
        # The input to .ainvoke should be a list of messages or a string based on model type.
        # For ChatAnthropic with HumanMessage, this is correct.
        response_payload = await model.ainvoke([HumanMessage(content=formatted_prompt)]) # type: ignore

        if isinstance(response_payload, ClassificationSchema):
            return {"shouldSearch": response_payload.should_search}
        else:
            # This case should ideally not be reached if with_structured_output works as expected.
            print(f"classify_message_node: Unexpected response type from LLM: {type(response_payload)}")
            return {"shouldSearch": False}

    except Exception as e:
        print(f"Error during message classification LLM call: {e}")
        return {"shouldSearch": False} # Default to not searching on error


async def query_generator_node(state: WebSearchState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: query_generator (WebSearchGraph)")
    # Model from TS: claude-3-5-sonnet-latest, temperature 0
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

    # Python equivalent of date-fns format(new Date(), "PPpp")
    # Example: "May 20th, 2024 at 3:30:25 PM" -> "%B %d, %Y at %I:%M:%S %p"
    # Using a slightly more standard ISO-like format for simplicity, but can be adjusted.
    current_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    additional_context = f"The current date is {current_date_str}."

    formatted_messages = _format_messages_for_web_search_prompt(state.get("messages", []))
    if not formatted_messages:
        # Fallback if there are no messages to format (e.g. empty state)
        if state.get("messages") and not state.get("messages", [])[-1].content: # Last message empty
             return {"query": ""} # Return empty query
        # Or, if really no messages, this node might not even be called due to graph logic.
        # For safety, if no conversation, generate a generic query or handle error.
        # However, the prompt expects a conversation.
        print("query_generator_node: No conversation messages to format for query generation.")
        return {"query": "general information"} # Fallback query


    formatted_prompt = QUERY_GENERATOR_PROMPT.format(
        conversation=formatted_messages,
        additional_context=additional_context
    )

    try:
        response = await model.ainvoke([HumanMessage(content=formatted_prompt)])
        query_content = response.content
        if not isinstance(query_content, str): query_content = str(query_content)

        return {"query": query_content.strip()}
    except Exception as e:
        print(f"Error during query generation LLM call: {e}")
        # Fallback: use the content of the last message as the query
        last_message_content = get_string_from_content(state.get("messages", [])[-1].content) if state.get("messages") else ""
        return {"query": last_message_content.strip()}


async def search_node(state: WebSearchState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    print("Executing Node: search_node (WebSearchGraph)")
    exa_api_key = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        print("Warning: EXA_API_KEY environment variable not found. Skipping web search.")
        return {"webSearchResults": []}

    retriever = ExaRetriever(
        api_key=exa_api_key, # Corrected: pass api_key directly
        k=5 # Number of results, equivalent to numResults: 5 in TS
        # filter_empty_results is not a direct param, Exa might do this by default or results need manual filtering.
    )

    search_query = state.get("query")
    if not search_query: # Should be set by query_generator_node
        if state.get("messages"):
            search_query = get_string_from_content(state["messages"][-1].content) # Fallback to last message
            print(f"Search node: No query explicitly generated, using last message content: '{search_query}'")
        else:
            print("Search node: No query provided and no messages to derive from. Cannot perform search.")
            return {"webSearchResults": []}

    if not search_query.strip():
        print("Search node: Query is empty or whitespace. Skipping search.")
        return {"webSearchResults": []}

    print(f"Performing Exa search for query: '{search_query}'")
    try:
        # ExaRetriever.ainvoke returns List[Document]
        # LangChain Document has page_content: str, metadata: dict
        # We need to map this to our app.schemas.common.SearchResult Pydantic model structure.
        # SearchResult = { page_content: str, metadata: ExaMetadata }
        # ExaMetadata = { id: str, url: str, title: str, author: Optional[str], publishedDate: Optional[str] }

        results_docs = await retriever.ainvoke(search_query)

        search_results_for_state: List[Dict[str, Any]] = []
        for doc in results_docs:
            # Metadata from Exa documents might include 'id', 'url', 'title', 'author', 'published_date' (or similar keys)
            # We need to map these to our ExaMetadata Pydantic model's field names.
            doc_metadata = doc.metadata or {}
            mapped_metadata = {
                "id": doc_metadata.get("id", str(uuid.uuid4())), # Exa provides 'id'
                "url": doc_metadata.get("url", ""), # Exa provides 'url'
                "title": doc_metadata.get("title", "Untitled Search Result"), # Exa provides 'title'
                "author": doc_metadata.get("author"), # Optional
                "publishedDate": doc_metadata.get("published_date") or doc_metadata.get("publishedDate"), # Check for variations
            }
            search_results_for_state.append({
                "page_content": doc.page_content,
                "metadata": mapped_metadata
            })

        # The state expects List[SearchResult], but we return List[Dict] which will be parsed by Pydantic
        # when the main graph receives this and updates its own OpenCanvasState.webSearchResults.
        return {"webSearchResults": search_results_for_state}
    except Exception as e:
        print(f"Error during Exa search for query '{search_query}': {e}")
        return {"webSearchResults": []} # Return empty list on error

# Ensure all necessary types are available
from app.schemas.common import SearchResult # For type hint if needed, though state uses it
from app.utils.text_processing import get_string_from_content # Already imported
from app.agents.summarizer_graph.nodes import _format_messages_for_summary as format_messages_for_web_search # Check path or move common utils
# Correcting import for _format_messages_for_web_search_prompt - it's locally defined.
# No, it was defined above, so it's fine.

# For ExaRetriever, ensure `langchain-exa` and `exa-py` are in requirements.txt
# EXA_API_KEY environment variable must be set.
