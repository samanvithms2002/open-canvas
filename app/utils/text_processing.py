# app/utils/text_processing.py
import re
from typing import List, Union, Sequence

# Langchain Core MessageContent is complex:
# MessageContent = Union[str, List[Union[str, Dict]]]
# For an AI Message, it could be:
# content=[{"type": "text", "text": "Hello!"}, {"type": "image_url", "image_url": {"url": "..."}}]
# or content="Just text"
# This simplified version handles common cases for now.
# A more robust version might be needed if other content types are used.
from langchain_core.messages import BaseMessage
from typing import Dict # Added Dict
from app.utils.langchain_helpers import THINKING_MODELS # Import the list


def get_string_from_content(content: Union[str, Sequence[Union[str, dict]]]) -> str:
    if isinstance(content, str):
        return content

    texts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, str): # Should not happen with current BaseMessage content types
                texts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
    return "\n".join(texts)

def extract_urls(text: str) -> List[str]:
    # Regex for markdown links: [text](url)
    markdown_link_regex = r'\[([^\]]+)\]\((https?://[^\s)]+)\)'
    urls = set()

    # First, find and store URLs from markdown links, and replace them to avoid double matching
    processed_text = text
    for match in re.finditer(markdown_link_regex, text):
        urls.add(match.group(2))
        # Replace the found markdown link with spaces to maintain positions for other regexes if needed,
        # though for URL extraction, simple removal or replacement is fine.
        processed_text = processed_text.replace(match.group(0), " " * len(match.group(0)))

    # Regex for plain URLs
    # This regex is a common one, but URL regex can be complex.
    # Source: https://urlregex.com/ (Python specific version)
    plain_url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    plain_urls_found = re.findall(plain_url_regex, processed_text)
    for url in plain_urls_found:
        urls.add(url)

    return list(urls)
