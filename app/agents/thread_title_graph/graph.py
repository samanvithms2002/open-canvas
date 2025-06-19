# app/agents/thread_title_graph/graph.py
from langgraph.graph import StateGraph, START, END
from .state import TitleGenerationState
from .nodes import generate_title_internal

# Create a new state graph instance for title generation
builder = StateGraph(TitleGenerationState)

# Add the single node responsible for generating the title
builder.add_node("title", generate_title_internal)

# Define the entry and exit points of the graph
builder.add_edge(START, "title") # Start at the 'title' node
builder.add_edge("title", END)   # End after the 'title' node completes

# Compile the graph into a runnable application
thread_title_graph_app = builder.compile()

# This `thread_title_graph_app` can now be invoked by other parts of the system,
# specifically by the main OpenCanvas agent's `generate_title_node`.
# It expects an input dictionary matching `TitleGenerationState` and a config.
# Example invocation from main graph's generate_title_node:
#
# title_input = {
#     "messages": relevant_messages_from_main_state_messages, # Usually first few messages
#     "artifact": current_artifact_from_main_state,
# }
# title_run_config = {
#     "configurable": {
#         "open_canvas_thread_id": main_graph_thread_id,
#         # Potentially pass checkpointer/store if needed for other reasons
#     }
# }
# await thread_title_graph_app.ainvoke(title_input, config=title_run_config)
