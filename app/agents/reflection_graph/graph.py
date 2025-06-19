# app/agents/reflection_graph/graph.py
from langgraph.graph import StateGraph, START, END
from .state import ReflectionGraphState
from .nodes import reflect_node_internal

# Create a new state graph instance for reflections
builder = StateGraph(ReflectionGraphState)

# Add the single node responsible for performing reflections
builder.add_node("reflect", reflect_node_internal)

# Define the entry and exit points of the graph
builder.add_edge(START, "reflect") # Start at the 'reflect' node
builder.add_edge("reflect", END)   # End after the 'reflect' node completes

# Compile the graph into a runnable application
reflection_graph_app = builder.compile()

# This `reflection_graph_app` can now be invoked by other parts of the system,
# specifically by the main OpenCanvas agent's `reflect_node`.
# It expects an input dictionary matching `ReflectionGraphState` and a config.
# Example invocation from main graph's reflect_node:
#
# reflection_input = {
#     "messages": relevant_messages_from_main_state,
#     "artifact": current_artifact_from_main_state,
# }
# reflection_run_config = {
#     "configurable": {
#         "open_canvas_assistant_id": main_graph_assistant_id,
#         # Potentially pass checkpointer/store if not globally accessible
#         # "checkpointer": main_graph_checkpointer
#     }
# }
# await reflection_graph_app.ainvoke(reflection_input, config=reflection_run_config)
