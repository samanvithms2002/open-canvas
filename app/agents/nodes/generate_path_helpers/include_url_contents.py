# app/agents/nodes/generate_path_helpers/include_url_contents.py
import os
import asyncio
import json # For parsing model output if it's a string
from typing import List, Optional, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage # type: ignore
from langchain_core.pydantic_v1 import BaseModel, Field
# from langsmith import traceable # If using langsmith

from app.utils.langchain_helpers import get_model_from_config, get_string_from_message_content
from langchain_community.document_loaders import FirecrawlLoader

# Equivalent to the Zod schema in TypeScript
class DetermineIncludeUrlContentsSchema(BaseModel):
    should_include_url_contents: bool = Field(..., description="Whether or not to include the contents of the URL in the prompt.")

determine_include_url_contents_tool_definition = {
    "name": "determine_include_url_contents",
    "description": "Whether or not the user's message indicates the contents of the URL should be included in the prompt.",
    "parameters": DetermineIncludeUrlContentsSchema.schema()
}

LLM_PROMPT_TEMPLATE = """You're an advanced AI assistant.
You have been tasked with analyzing the user's message and determining if the user wants the contents of the URL included in their message included in their prompt.
You should ONLY answer 'true' if it is explicitly clear the user included the URL in their message so that its contents would be included in the prompt, otherwise, answer 'false'.

Here is the user's message:
<message>
{message_content}
</message>

Now, given their message, determine whether or not they want the contents of that webpage to be included in the prompt."""

# @traceable(name="fetch_url_contents") # If using langsmith
async def fetch_url_contents_func(url: str) -> Dict[str, str]:
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        print(f"Warning: FIRECRAWL_API_KEY not set. Cannot fetch URL {url}.")
        return {"url": url, "page_content": f"Error: FIRECRAWL_API_KEY not configured."}

    loader = FirecrawlLoader(api_key=api_key, url=url, mode="scrape")
    try:
        docs = await loader.aload()
        page_content = docs[0].page_content if docs and docs[0].page_content else ""
        if not page_content:
             return {"url": url, "page_content": "Error: No content found or content was empty."}
        return {"url": url, "page_content": page_content}
    except Exception as e:
        print(f"Error fetching URL {url} with Firecrawl: {e}")
        return {"url": url, "page_content": f"Error fetching content: {e}"}


# @traceable(name="include_url_contents") # If using langsmith
async def include_url_contents_func(
    message: HumanMessage,
    urls: List[str],
    config: Dict[str, Any], # LangGraphRunnableConfig's 'configurable' part
) -> Optional[HumanMessage]:
    try:
        # Ensure message.content can be converted to a string for the prompt
        message_content_str = get_string_from_message_content(message.content)
        if not message_content_str:
            print("Warning: include_url_contents message content is empty or not string-convertible.")
            return None

        # Config to select a specific model (e.g., gemini-2.0-flash in TS)
        # Ensure get_model_from_config can handle this.
        # The 'configurable' dict from the main graph config is passed directly.
        # We might need to create a *new* config dict here if we want to override the model
        # specifically for this LLM call.
        llm_selection_config = config.copy() # Start with a copy of the main 'configurable'
        llm_selection_config["customModelName"] = os.getenv("URL_INCLUDE_LLM_MODEL", "gpt-4o-mini") # Use a default or env var

        # Ensure provider is set if modelConfig is missing or incomplete
        if "modelConfig" not in llm_selection_config or not isinstance(llm_selection_config.get("modelConfig"), dict):
            llm_selection_config["modelConfig"] = {"provider": "openai"} # Default provider if gpt-4o-mini
        elif "provider" not in llm_selection_config["modelConfig"]:
            # Infer provider or set a default based on the chosen model name
            if "gemini" in llm_selection_config["customModelName"]:
                llm_selection_config["modelConfig"]["provider"] = "google-genai"
            else: # Default to openai if model is like gpt-*
                llm_selection_config["modelConfig"]["provider"] = "openai"

        model = await get_model_from_config(llm_selection_config)

        model_with_tool = model.bind_tools(
            tools=[determine_include_url_contents_tool_definition],
            tool_choice="determine_include_url_contents" # Force calling this tool
        )

        formatted_prompt = LLM_PROMPT_TEMPLATE.format(message_content=message_content_str)

        llm_response_message = await model_with_tool.ainvoke([HumanMessage(content=formatted_prompt)])

        args = {}
        if isinstance(llm_response_message, AIMessage) and llm_response_message.tool_calls:
            # Langchain AIMessage tool_calls is a list of dicts
            # Each dict has 'name', 'args' (dict), 'id'
            # The TS code seems to expect args directly.
            # Check if there are any tool calls
            if llm_response_message.tool_calls:
                 # Check if the 'args' is a string that needs parsing, or already a dict
                raw_args = llm_response_message.tool_calls[0].get("args")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        print(f"Error: Could not parse tool call arguments string: {raw_args}")
                        args = {} # Fallback to empty dict
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {} # Fallback
            else: # No tool calls found
                args = {}
        else: # Response was not an AIMessage or had no tool_calls
            print(f"Warning: Model did not return expected tool call. Response: {llm_response_message}")
            args = {}

        should_include = args.get("should_include_url_contents", False)

        if not should_include:
            return None # Return None if content should not be included

        # Fetch content for all URLs concurrently
        url_contents_results = await asyncio.gather(*(fetch_url_contents_func(url) for url in urls))

        transformed_prompt_content = message_content_str
        for res in url_contents_results:
            # Replace the URL with its content, wrapped in a specific tag
            transformed_prompt_content = transformed_prompt_content.replace(
                res["url"],
                f'<page-contents url="{res["url"]}">\n{res["page_content"]}\n</page-contents>'
            )

        # Create a new HumanMessage, do not modify the original message object
        new_message_kwargs = {"content": transformed_prompt_content}
        if message.id: # Preserve ID if present
            new_message_kwargs["id"] = message.id
        # Preserve additional_kwargs, ensuring it's a new dictionary
        new_message_kwargs["additional_kwargs"] = message.additional_kwargs.copy() if message.additional_kwargs else {}

        return HumanMessage(**new_message_kwargs) # type: ignore

    except Exception as e:
        import traceback
        print(f"Failed to process URLs for inclusion: {e}")
        traceback.print_exc() # More detailed error for debugging
        return None # Or return the original message, or handle error differently

# Example usage (for testing purposes)
async def main_test():
    class MockHumanMessage(HumanMessage):
        def __init__(self, content, id=None, additional_kwargs=None):
            super().__init__(content=content, id=id, additional_kwargs=additional_kwargs or {})

    # Test 1: URL should be included
    msg1 = MockHumanMessage(content="Check out this site https://example.com and tell me about it.")
    urls1 = ["https://example.com"]
    # Mock config - ensure your get_model_from_config can work with this or provide a fuller mock
    cfg1 = {"configurable": {"customModelName": "gpt-4o-mini", "modelConfig": {"provider": "openai", "temperature": 0.1}}}

    # Set FIRECRAWL_API_KEY environment variable before running if you want to test actual fetching
    # For CI/testing without API key, fetch_url_contents_func will return an error string.
    if not os.getenv("FIRECRAWL_API_KEY"):
        print("FIRECRAWL_API_KEY not set, URL fetching will be skipped or use mocked data if available.")
        # You might mock fetch_url_contents_func for tests not requiring live calls

    # result1 = await include_url_contents_func(msg1, urls1, cfg1["configurable"])
    # print(f"Test 1 Result: {result1.content if result1 else 'None'}")

    # Test 2: URL should not be included (e.g., just a reference)
    # msg2 = MockHumanMessage(content="I found this link: http://another-example.com, it's interesting.")
    # urls2 = ["http://another-example.com"]
    # result2 = await include_url_contents_func(msg2, urls2, cfg1["configurable"])
    # print(f"Test 2 Result: {result2.content if result2 else 'None'}")

if __name__ == "__main__":
    # To run this test, you'll need OPENAI_API_KEY (or your chosen model's API key)
    # and potentially FIRECRAWL_API_KEY set in your environment.
    # asyncio.run(main_test())
    pass
