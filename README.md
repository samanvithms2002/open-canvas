# Python LangGraph Agent Server (OpenCanvas Port)

## Overview

This project is a Python-based backend server that implements the agentic workflows originally found in the OpenCanvas TypeScript LangGraph server. It uses FastAPI for the web framework and LangGraph for orchestrating complex agent behaviors. The server is designed to be compatible with the existing OpenCanvas frontend.

Key functionalities include:
- Chat interaction with an AI agent.
- Artifact generation and modification (text and code).
- Web search capabilities.
- Conversation summarization.
- User-defined custom quick actions.
- Persistent conversation state and history.

## Prerequisites

- Python 3.9+
- pip (Python package installer)
- Access to necessary API keys (OpenAI, Anthropic, Exa, etc.)

## Setup Instructions

1.  **Clone the repository** (if you haven't already):
    ```bash
    # git clone ... (repo URL)
    # cd path/to/this/python_server_directory
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and fill in your API keys and any other required configurations:
        ```env
        OPENAI_API_KEY="your_openai_api_key_here"
        LANGCHAIN_API_KEY="your_langchain_api_key_here" # For LangSmith
        LANGCHAIN_TRACING_V2="true"
        LANGCHAIN_PROJECT="OpenCanvas FastAPI"

        ANTHROPIC_API_KEY="your_anthropic_api_key_here"
        EXA_API_KEY="your_exa_api_key_here"
        # FIRECRAWL_API_KEY="your_firecrawl_api_key_here" # If using Firecrawl in web search

        LANGGRAPH_SQLITE_PATH="opencanvas_checkpoints.sqlite"
        ```

## Running the Server

To run the FastAPI server locally:

```bash
uvicorn app.main:app --reload --port 8000
```

-   `--reload`: Enables auto-reloading when code changes (useful for development).
-   `--port 8000`: Specifies the port to run on. You can change this if needed.

The server will be accessible at `http://localhost:8000`.

## API Documentation

Once the server is running, FastAPI provides automatic interactive API documentation:

-   **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
-   **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

These interfaces allow you to explore and test the API endpoints.

## Key Environment Variables

Make sure the following environment variables are set in your `.env` file for full functionality:

-   `OPENAI_API_KEY`: For OpenAI models.
-   `ANTHROPIC_API_KEY`: For Anthropic (Claude) models.
-   `EXA_API_KEY`: For the Exa web search tool.
-   `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT`: For LangSmith tracing (optional but recommended for debugging).
-   `LANGGRAPH_SQLITE_PATH`: Path to the SQLite database file for conversation state persistence. Defaults to `opencanvas_checkpoints.sqlite` in the root directory.

## Project Structure

-   `app/`: Main application code.
    -   `main.py`: FastAPI application entry point, CORS setup.
    -   `api/`: API route definitions (e.g., `agent.py`).
    -   `agents/`: LangGraph agent definitions.
        -   `open_canvas.py`: Main OpenCanvas agent graph.
        -   `reducers.py`: Custom state reducers for the main graph.
        -   `prompts.py`: Prompts for the main agent.
        -   `state.py`: State definition for the main agent.
        -   `nodes/`: Helper modules for specific nodes in the main agent.
        -   `reflection_graph/`: Sub-graph for generating reflections.
        -   `summarizer_graph/`: Logic for conversation summarization.
        -   `thread_title_graph/`: Sub-graph for generating thread titles.
        -   `web_search_graph/`: Sub-graph for performing web searches.
    -   `config/`: Configuration settings (e.g., `settings.py` for Pydantic settings).
    -   `schemas/`: Pydantic models for API requests/responses and common data structures.
    -   `services/`: Business logic, including `agent_service.py` which interacts with LangGraph.
    -   `utils/`: Utility functions (text processing, LangChain helpers).
-   `requirements.txt`: Python dependencies.
-   `.env.example`: Example environment variable file.
-   `opencanvas_checkpoints.sqlite`: Default database for conversation state (created on first run if `LANGGRAPH_SQLITE_PATH` is not changed).
-   `README.md`: This file.
