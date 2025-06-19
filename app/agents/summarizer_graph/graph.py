# app/agents/summarizer_graph/graph.py
from langgraph.graph import StateGraph, START, END
from .state import SummarizerGraphState
from .nodes import summarize_messages_node

builder = StateGraph(SummarizerGraphState)

# Add the single node responsible for summarization
builder.add_node("summarize", summarize_messages_node)

# Define the entry and exit points for this simple graph
builder.add_edge(START, "summarize") # Start at the 'summarize' node
builder.add_edge("summarize", END)   # End after the 'summarize' node completes

# Compile the graph into a runnable application
summarizer_graph_app = builder.compile()
print("Summarizer Graph compiled successfully.")

# This `summarizer_graph_app` can be invoked by other parts of the system.
# It expects an input dictionary matching `SummarizerGraphState` (i.e., {"messages_in": List[BaseMessage]})
# and will return the full `SummarizerGraphState` including `summarized_message_out`.
#
# Example invocation from the main OpenCanvas graph's summarizer_node:
#
# summarizer_input = {"messages_in": messages_from_main_graph_state}
# result_state = await summarizer_graph_app.ainvoke(summarizer_input, config=run_config)
# new_summary_message = result_state.get("summarized_message_out")
# The main graph would then use this new_summary_message.
