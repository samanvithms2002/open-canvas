# app/utils/langchain_helpers.py
import base64
import os
import re # Added re import
from typing import List, Dict, Any, Optional, Union, Literal, Sequence

from pydantic import BaseModel, HttpUrl
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage # type: ignore
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI # Example for later
# from langchain_community.chat_models import ChatFireworks, ChatGroq # Example for later
# from langchain_community.chat_models.ollama import ChatOllama # Example for later

from app.schemas.common import ArtifactV3, ArtifactCodeV3, ArtifactMarkdownV3, ArtifactType, ProgrammingLanguageOptions, SearchResult


# --- Constants (Ported from @opencanvas/shared/models.ts) ---
LANGCHAIN_USER_ONLY_MODELS = [
  "o1",
  "gpt-4o",
  "gpt-4.5-preview",
  "claude-3-5-sonnet-latest",
  "claude-3-7-sonnet-latest",
  "gemini-2.0-flash-thinking-exp-01-21",
  "gemini-2.5-pro-preview-05-06",
  "claude-sonnet-4-0",
  "claude-opus-4-0",
  "gpt-4.1",
]

TEMPERATURE_EXCLUDED_MODELS = [
  "o1-mini",
  "o3-mini",
  "o1",
  "o4-mini",
]

# OC_WEB_SEARCH_RESULTS_MESSAGE_KEY = "oc_web_search_results" # From @opencanvas/shared/constants
# CONTEXT_DOCUMENTS_NAMESPACE = "context_documents" # From @opencanvas/shared/constants


class CustomModelConfig(BaseModel): # Simplified from TS ModelConfigurationParams.config
    provider: str
    # Not including detailed range/default/current for temp/tokens for now
    temperature: Optional[float] = None # Maps to temperatureRange.current in TS
    max_tokens: Optional[int] = None # Maps to maxTokens.current in TS
    # azure_config: Optional[Dict[str, str]] = None # This will be handled in get_model_config


def get_model_config(
    config: Dict[str, Any], # Represents LangGraphRunnableConfig's 'configurable' field usually
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    # In LangGraph, 'configurable' usually holds dynamic per-invocation settings.
    # We expect 'customModelName' and 'modelConfig' (which is CustomModelConfig) to be in 'configurable'.

    custom_model_name = config.get("customModelName")
    if not custom_model_name:
        # Fallback to a default model if not specified, or raise error
        # Based on TS DEFAULT_MODEL_NAME = OPENAI_MODELS[1].name which is "gpt-4.1-mini"
        # For simplicity, let's use a common one or raise error.
        print("Warning: 'customModelName' not found in config. Defaulting to gpt-4o-mini.")
        custom_model_name = "gpt-4o-mini"
        # raise ValueError("Model name ('customModelName') is missing in 'configurable' part of the config.")

    model_config_data = config.get("modelConfig") # This should be a dict matching CustomModelConfig
    parsed_model_config: Optional[CustomModelConfig] = None
    if model_config_data and isinstance(model_config_data, dict):
        try:
            parsed_model_config = CustomModelConfig(**model_config_data)
        except Exception as e:
            print(f"Warning: Could not parse 'modelConfig' from config: {e}. Using defaults.")
            # Fallback to trying to infer provider from name if modelConfig is bad/missing
            if "azure/" in custom_model_name:
                 parsed_model_config = CustomModelConfig(provider="azure_openai")
            elif any(prefix in custom_model_name for prefix in ["gpt-", "o1", "o3", "o4"]):
                 parsed_model_config = CustomModelConfig(provider="openai")
            elif "claude" in custom_model_name:
                 parsed_model_config = CustomModelConfig(provider="anthropic")
            # Add more provider inferences here...
            else: # Absolute fallback
                 parsed_model_config = CustomModelConfig(provider="openai", temperature=0.5, max_tokens=1000)


    elif not model_config_data:
        print(f"Warning: 'modelConfig' not found in config for model {custom_model_name}. Inferring provider.")
        # Infer provider if modelConfig is missing
        if "azure/" in custom_model_name:
             parsed_model_config = CustomModelConfig(provider="azure_openai")
        elif any(prefix in custom_model_name for prefix in ["gpt-", "o1", "o3", "o4"]): # Approximate OpenAI
             parsed_model_config = CustomModelConfig(provider="openai")
        elif "claude" in custom_model_name:
             parsed_model_config = CustomModelConfig(provider="anthropic")
        # Add more provider inferences here...
        else: # Absolute fallback
             print(f"Warning: Could not infer provider for {custom_model_name}. Defaulting provider to OpenAI.")
             parsed_model_config = CustomModelConfig(provider="openai", temperature=0.5, max_tokens=1000)


    is_tool_calling = extra.get("isToolCalling", False) if extra else False

    temperature = parsed_model_config.temperature if parsed_model_config and parsed_model_config.temperature is not None else 0.5
    if custom_model_name in TEMPERATURE_EXCLUDED_MODELS:
        temperature = None # Or 0.0 if API requires it, Langchain handles None by not sending the param

    max_tokens = parsed_model_config.max_tokens if parsed_model_config and parsed_model_config.max_tokens is not None else 1000

    params: Dict[str, Any] = {
        "model_name": custom_model_name,
        "model_provider": parsed_model_config.provider,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if parsed_model_config.provider == "azure_openai":
        params["model_name"] = custom_model_name.replace("azure/", "")
        if is_tool_calling and "o1" in params["model_name"]: # Example fallback from TS
            params["model_name"] = "gpt-4o" # Ensure this is a valid Azure deployment name
        # These should come from environment variables, specific to Azure setup
        params["azure_deployment"] = params["model_name"] # Often deployment name matches model name or is specific
        params["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        params["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        params["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION")
    elif parsed_model_config.provider == "openai":
        if is_tool_calling and "o1" in custom_model_name: # Example fallback
             params["model_name"] = "gpt-4o"
        params["api_key"] = os.getenv("OPENAI_API_KEY")
    elif parsed_model_config.provider == "anthropic":
        params["api_key"] = os.getenv("ANTHROPIC_API_KEY")
    # Add other providers (google-genai, fireworks, groq, ollama) similarly
    # For ollama, you'd add model_name (e.g. "ollama-llama3.3" -> "llama3.3") and potentially a base_url
    # elif parsed_model_config.provider == "ollama":
    #    params["model_name"] = custom_model_name.replace("ollama-", "")
    #    params["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Remove None temperature for models that don't support it or if user wants default
    if params["temperature"] is None:
        del params["temperature"]

    return params


async def get_model_from_config(
    config: Dict[str, Any], # LangGraphRunnableConfig's 'configurable' field
    extra: Optional[Dict[str, Any]] = None
) -> Any: # Should be BaseChatModel
    model_params = get_model_config(config, extra)

    provider = model_params.pop("model_provider")
    # model_name = model_params.pop("model_name") # model_params still has model_name

    # TODO: Check for LANGCHAIN_USER_ONLY_MODELS and user auth as in TS
    # This would require knowing the user_id from the config, which is not directly handled here.
    # configurable_user_id = config.get("configurable", {}).get("user_id")
    # if model_params.get("model_name") in LANGCHAIN_USER_ONLY_MODELS and not configurable_user_id:
    #     raise ValueError(f"Model {model_params.get('model_name')} requires user authentication.")

    if provider == "openai":
        return ChatOpenAI(**model_params)
    elif provider == "anthropic":
        return ChatAnthropic(**model_params)
    elif provider == "azure_openai":
        # AzureOpenAI takes 'azure_deployment' as a key param, not 'model_name' for deployment
        # 'model_name' might be used for underlying model reference if API supports/needs it
        return ChatOpenAI(**model_params) # Uses azure_deployment, api_key, azure_endpoint, api_version
    # Add other providers
    # elif provider == "google-genai":
    #     model_params["google_api_key"] = model_params.pop("api_key", None)
    #     return ChatGoogleGenerativeAI(**model_params)
    # elif provider == "ollama":
    #     return ChatOllama(**model_params)
    # ...
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

def clean_base64(base64_string: str) -> str:
    return re.sub(r'^data:.*?;base64,', '', base64_string)

def convert_pdf_to_text(base64_pdf: str) -> str:
    try:
        from pypdf import PdfReader
        import io
        cleaned_base64 = clean_base64(base64_pdf)
        pdf_buffer = base64.b64decode(cleaned_base64)
        pdf_file = io.BytesIO(pdf_buffer)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error converting PDF to text: {e}")
        # Depending on desired error handling, either raise or return empty/error string
        raise # Or return "" or f"Error: {e}"


class ContextDocument(BaseModel):
    name: str
    type: str # e.g. "application/pdf", "text/plain", "image/png"
    data: str # base64 encoded for files, or plain text if type is "text"
    metadata: Optional[Dict[str, Any]] = None


async def create_context_document_messages_openai(documents: List[ContextDocument]) -> List[Dict[str, Any]]:
    # OpenAI can handle multiple text parts and image_url parts in a single HumanMessage content list.
    content_parts: List[Dict[str, Any]] = []
    for doc in documents:
        if doc.type == "application/pdf":
            try:
                text = convert_pdf_to_text(doc.data)
                if text:
                    content_parts.append({"type": "text", "text": f"Context from PDF '{doc.name}':\n{text}"})
            except Exception as e:
                content_parts.append({"type": "text", "text": f"Error processing PDF '{doc.name}': {e}"})
        elif doc.type.startswith("text/"):
            try:
                text = base64.b64decode(clean_base64(doc.data)).decode('utf-8')
                content_parts.append({"type": "text", "text": f"Context from text file '{doc.name}':\n{text}"})
            except Exception as e:
                 content_parts.append({"type": "text", "text": f"Error processing text file '{doc.name}': {e}"})
        elif doc.type == "text": # Plain text data, not base64
             content_parts.append({"type": "text", "text": f"Context from text input '{doc.name}':\n{doc.data}"})
        elif doc.type.startswith("image/"):
            cleaned_b64_image = clean_base64(doc.data)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{doc.type};base64,{cleaned_b64_image}"}
            })
        # Add other type handlers if necessary
    return content_parts

# TODO: Implement create_context_document_messages_anthropic, create_context_document_messages_gemini
# Anthropic: Similar to OpenAI, may need to adjust if image handling differs.
# Gemini: Handles images and text. Ensure format matches their API.

async def get_context_documents(config: Dict[str, Any]) -> List[ContextDocument]:
    # This is a placeholder.
    # Actual implementation would fetch from a vector store or other document source
    # using details from the 'config' (e.g., thread_id, user_id for filtering).
    # The TS version uses `config.store.Mget(`${CONTEXT_DOCUMENTS_NAMESPACE}:${configurable.thread_id}`);`
    # which implies a key-value store (like Redis used in LangServe examples).
    # For now, it returns an empty list.
    print("Warning: `get_context_documents` is a placeholder and currently returns no documents from store.")
    return []


async def create_context_document_messages(
    config: Dict[str, Any], # LangGraphRunnableConfig's 'configurable' field
    context_documents_override: Optional[List[ContextDocument]] = None
) -> List[Dict[str, Any]]: # Returns list of message content parts for a HumanMessage

    documents_to_process: List[ContextDocument] = []
    if context_documents_override is not None:
        documents_to_process.extend(context_documents_override)

    # Option to fetch from store if no override is provided.
    # if not documents_to_process and config:
    #     docs_from_store = await get_context_documents(config) # Needs actual store interaction
    #     documents_to_process.extend(docs_from_store)

    if not documents_to_process:
        return []

    # Get model provider to dispatch to the correct formatter if needed
    # model_params = get_model_config(config) # config here is 'configurable' part
    # model_provider = model_params["model_provider"]

    # For now, using OpenAI's format as the primary one.
    # Adapt if other models have significantly different ways of consuming context.
    # if model_provider == "anthropic":
    #    formatted_doc_content_parts = await create_context_document_messages_anthropic(documents_to_process)
    # elif model_provider == "google-genai":
    #    formatted_doc_content_parts = await create_context_document_messages_gemini(documents_to_process)
    # else: # Default to OpenAI compatible
    formatted_doc_content_parts = await create_context_document_messages_openai(documents_to_process)

    if not formatted_doc_content_parts:
        return []

    # Construct the structure for HumanMessage(content=[...])
    return [
        {
            "type": "text",
            "text": "Use the file(s) and/or text below as context when generating your response.",
        },
        *formatted_doc_content_parts,
    ]

def is_artifact_code_content(content: Union[ArtifactCodeV3, ArtifactMarkdownV3]) -> bool:
    return content.type == ArtifactType.CODE

def get_artifact_content(artifact: Optional[ArtifactV3]) -> Optional[Union[ArtifactCodeV3, ArtifactMarkdownV3]]:
    if not artifact or not artifact.contents:
        return None

    current_content = next((c for c in artifact.contents if c.index == artifact.currentIndex), None)

    if not current_content:
        # Fallback to the last content if currentIndex is out of sync or points to no content
        return artifact.contents[-1]
    return current_content


def format_artifact_content(
    content: Union[ArtifactCodeV3, ArtifactMarkdownV3],
    shorten_content: bool = False,
    max_length: int = 500 # Max length for shortening
) -> str:
    artifact_text: str
    if is_artifact_code_content(content):
        # Ensure content is the correct type for type checkers after is_artifact_code_content
        assert isinstance(content, ArtifactCodeV3), "Content type mismatch for code artifact"
        artifact_text = content.code[:max_length] if shorten_content else content.code
    else:
        assert isinstance(content, ArtifactMarkdownV3), "Content type mismatch for markdown artifact"
        artifact_text = content.fullMarkdown[:max_length] if shorten_content else content.fullMarkdown

    title = content.title if content.title else "Untitled"
    return f"Title: {title}\nArtifact type: {content.type.value}\nContent: {artifact_text}"


def format_artifact_content_with_template(
    template: str, # An f-string template like "Context:\n{artifact}\nUser query:"
    content: Union[ArtifactCodeV3, ArtifactMarkdownV3],
    shorten_content: bool = False,
    max_length: int = 500
) -> str:
    formatted_artifact = format_artifact_content(content, shorten_content, max_length)
    # Using .format() for safety if template comes from external source
    # If template is always a trusted f-string literal source, f-strings are fine too.
    return template.replace("{artifact}", formatted_artifact)


def is_using_o1_mini_model(config: Dict[str, Any]) -> bool:
    # graph_config is expected to be the 'configurable' part of RunnableConfig
    model_params = get_model_config(config)
    return "o1-mini" in model_params.get("model_name", "")


# Helper for app/agents/open_canvas.py to get string from BaseMessage content
# Copied from app.utils.text_processing to avoid circular dependency if text_processing needs langchain_helpers
def get_string_from_message_content(message_content: Union[str, Sequence[Union[str, Dict]]]) -> str:
    if isinstance(message_content, str):
        return message_content
    texts: List[str] = []
    if isinstance(message_content, list):
        for part in message_content:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
    return "\n".join(texts)

THINKING_MODELS = [ # From packages/shared/src/models.ts
    "accounts/fireworks/models/deepseek-r1",
    "groq/deepseek-r1-distill-llama-70b",
]
