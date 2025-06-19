# app/schemas/common.py
from typing import List, Optional, Union, Any, Literal
from pydantic import BaseModel, HttpUrl
from enum import Enum

# Enums from shared types
class ArtifactLengthOptions(str, Enum):
    SHORTEST = "shortest"
    SHORT = "short"
    LONG = "long"
    LONGEST = "longest"

class ArtifactType(str, Enum):
    CODE = "code"
    TEXT = "text"

class LanguageOptions(str, Enum):
    ENGLISH = "english"
    MANDARIN = "mandarin"
    SPANISH = "spanish"
    FRENCH = "french"
    HINDI = "hindi"

class ProgrammingLanguageOptions(str, Enum):
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    JAVA = "java"
    PHP = "php"
    PYTHON = "python"
    HTML = "html"
    SQL = "sql"
    JSON = "json"
    RUST = "rust"
    XML = "xml"
    CLOJURE = "clojure"
    CSHARP = "csharp"
    OTHER = "other"

class ReadingLevelOptions(str, Enum):
    PIRATE = "pirate"
    CHILD = "child"
    TEENAGER = "teenager"
    COLLEGE = "college"
    PHD = "phd"

# Pydantic models from shared types
class CodeHighlight(BaseModel):
    startCharIndex: int
    endCharIndex: int

class ArtifactMarkdownV3(BaseModel):
    index: int
    type: Literal[ArtifactType.TEXT] = ArtifactType.TEXT
    title: str
    fullMarkdown: str

class ArtifactCodeV3(BaseModel):
    index: int
    type: Literal[ArtifactType.CODE] = ArtifactType.CODE
    title: str
    language: ProgrammingLanguageOptions
    code: str

class ArtifactV3(BaseModel):
    currentIndex: int
    contents: List[Union[ArtifactMarkdownV3, ArtifactCodeV3]]

class TextHighlight(BaseModel):
    fullMarkdown: str
    markdownBlock: str
    selectedText: str

class ExaMetadata(BaseModel):
    id: str
    url: HttpUrl
    title: str
    author: Optional[str] = None
    publishedDate: Optional[str] = None # Could be date type if format is known
    image: Optional[HttpUrl] = None
    favicon: Optional[HttpUrl] = None

class DocumentInterface(BaseModel):
    page_content: str
    metadata: ExaMetadata # In the TS, SearchResult is DocumentInterface<ExaMetadata>

class SearchResult(DocumentInterface): # Alias for clarity
    pass

class CustomQuickAction(BaseModel):
    id: str
    title: str
    prompt: str # This is the user-defined prompt for the custom action
    includeReflections: bool
    includePrefix: bool # Whether to include the standard CUSTOM_QUICK_ACTION_ARTIFACT_PROMPT_PREFIX
    includeRecentHistory: bool

# Shared constants (can be moved to a dedicated app/constants.py later if many accumulate)
OC_SUMMARIZED_MESSAGE_KEY = "__oc_summarized_message"
OC_HIDE_FROM_UI_KEY = "oc_hide_from_ui" # From generate_path_helpers/documents.py, centralizing
