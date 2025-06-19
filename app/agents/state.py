# app/agents/state.py
from typing import List, Optional, TypedDict, Any, Literal
from app.schemas.common import (
    CodeHighlight,
    TextHighlight,
    ArtifactV3,
    LanguageOptions,
    ArtifactLengthOptions,
    ReadingLevelOptions,
    ProgrammingLanguageOptions,
    SearchResult,
)
from langchain_core.messages import BaseMessage

# Based on OpenCanvasGraphAnnotation and shared types
class OpenCanvasState(TypedDict):
    messages: List[BaseMessage] # Full list of messages for conversation
    _messages: List[BaseMessage] # List of messages passed to the model (can include summarized)

    highlightedCode: Optional[CodeHighlight]
    highlightedText: Optional[TextHighlight]
    artifact: Optional[ArtifactV3]
    next_node: Optional[str] # Renamed 'next' to 'next_node' to avoid keyword clash if it were a class attribute

    language: Optional[LanguageOptions]
    artifactLength: Optional[ArtifactLengthOptions]
    regenerateWithEmojis: Optional[bool]
    readingLevel: Optional[ReadingLevelOptions]

    addComments: Optional[bool]
    addLogs: Optional[bool]
    portLanguage: Optional[ProgrammingLanguageOptions]
    fixBugs: Optional[bool]
    customQuickActionId: Optional[str]

    webSearchEnabled: Optional[bool]
    webSearchResults: Optional[List[SearchResult]]

    # Fields from DEFAULT_INPUTS in the TS graph if any, might be needed for cleanState
    # For example, if DEFAULT_INPUTS had {userInput: "", someOtherField: null}
    # userInput: Optional[str]
    # someOtherField: Optional[Any]
