# app/config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    OPENAI_API_KEY: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None # For LangSmith if used
    LANGCHAIN_TRACING_V2: Optional[str] = "false" # For LangSmith
    LANGCHAIN_PROJECT: Optional[str] = "OpenCanvas FastAPI" # For LangSmith

    ANTHROPIC_API_KEY: Optional[str] = None
    EXA_API_KEY: Optional[str] = None
    FIRECRAWL_API_KEY: Optional[str] = None # Added as it was mentioned in include_url_contents

    LANGGRAPH_SQLITE_PATH: str = "opencanvas_checkpoints.sqlite"

    # Example for frontend URL if needed for more specific CORS
    # FRONTEND_URL: str = "http://localhost:3000"

    # Supabase keys (optional, if backend needs direct Supabase access)
    # NEXT_PUBLIC_SUPABASE_URL: Optional[str] = None
    # SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None

    # Pydantic V2 style for model_config:
    # Loads variables from a .env file.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Create a single instance of the settings to be imported by other modules
settings = Settings()

# How to use:
# from app.config.settings import settings
# api_key = settings.OPENAI_API_KEY

# Note on API key environment variables:
# LangChain and other libraries often pick up environment variables like OPENAI_API_KEY directly
# if they are set in the environment (e.g., via python-dotenv loading .env).
# These Pydantic settings provide a structured way to access them and set defaults.
# If an env var is set, it will override the default here. If not set, and no default here, it'll be None.
# If a field is not Optional and has no default, Pydantic will raise an error if it's not set via env var.
