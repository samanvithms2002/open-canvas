# app/agents/web_search_graph/graph.py
from langgraph.graph import StateGraph, START, END
from typing import Literal # For Literal type hint
from .state import WebSearchState
from .nodes import classify_message_node, query_generator_node, search_node

def search_or_end_conditional(state: WebSearchState) -> Literal["queryGenerator", END]:
    """
    Determines the next step after classification.
    If shouldSearch is true, proceed to queryGenerator. Otherwise, end the graph.
    """
    print(f"WebSearchGraph Conditional: shouldSearch is {state.get('shouldSearch')}")
    if state.get("shouldSearch"):
        return "queryGenerator"
    return END

# Create a new state graph instance for web search
builder = StateGraph(WebSearchState)

# Add nodes to the graph
builder.add_node("classifyMessage", classify_message_node)
builder.add_node("queryGenerator", query_generator_node)
builder.add_node("search", search_node)

# Define the graph's flow
builder.add_edge(START, "classifyMessage") # Start with message classification

# Add conditional edge from classifyMessage:
# If shouldSearch is true, go to queryGenerator. Otherwise, END.
builder.add_conditional_edges(
    "classifyMessage",
    search_or_end_conditional,
    {
        "queryGenerator": "queryGenerator", # Path if shouldSearch is true
        END: END                             # Path if shouldSearch is false
    }
)

builder.add_edge("queryGenerator", "search") # After generating a query, perform the search
builder.add_edge("search", END)              # End after the search is performed

# Compile the graph into a runnable application
web_search_graph_app = builder.compile()

# This web_search_graph_app expects an input dictionary matching WebSearchState (primarily 'messages')
# and returns its final state, which includes 'webSearchResults'.
# The main OpenCanvas graph's web_search_node will invoke this app.
