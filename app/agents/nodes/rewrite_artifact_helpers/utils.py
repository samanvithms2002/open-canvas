# app/agents/nodes/rewrite_artifact_helpers/utils.py
from typing import Dict, Any, Union, Tuple, Optional, TypedDict # Added TypedDict, Optional

from langchain_core.messages import BaseMessage, HumanMessage # type: ignore

from app.agents.state import OpenCanvasState
from app.schemas.common import ArtifactCodeV3, ArtifactMarkdownV3, ProgrammingLanguageOptions, ArtifactType
from app.utils.langchain_helpers import get_artifact_content, is_artifact_code_content
from app.agents.prompts import OPTIONALLY_UPDATE_META_PROMPT, UPDATE_ENTIRE_ARTIFACT_PROMPT
from .schemas import OptionallyUpdateArtifactMetaSchema


def validate_state(state: OpenCanvasState) -> Tuple[Union[ArtifactCodeV3, ArtifactMarkdownV3], BaseMessage]:
    """
    Validates that the necessary parts of the state are present for rewriting an artifact.

    Returns:
        A tuple of (current_artifact_content_model, recent_human_message).
    Raises:
        ValueError if the artifact or human message is missing.
    """
    current_artifact_model = get_artifact_content(state.get("artifact"))
    if not current_artifact_model:
        raise ValueError("No artifact content found in state for rewrite_artifact.")

    recent_human_message: Optional[BaseMessage] = None
    # Iterate from the end of _messages to find the last human message
    for msg in reversed(state.get("_messages", [])):
        if msg.type == "human": # Check type attribute as per BaseMessage
            recent_human_message = msg
            break
    if not recent_human_message:
        raise ValueError("No recent human message found in state for rewrite_artifact.")

    return current_artifact_model, recent_human_message

def _build_meta_prompt(artifact_meta_tool_call: OptionallyUpdateArtifactMetaSchema) -> str:
    """
    Builds the meta prompt string based on the determined artifact metadata.
    This prompt part informs the main rewrite LLM about the intended title/type.
    """
    title_section = ""
    # Only include title in meta prompt if it's explicitly provided by the meta-update LLM
    # AND if the type is NOT code (as per TS logic, title for code is handled differently or less emphasized in prompt)
    if artifact_meta_tool_call.title and artifact_meta_tool_call.type != ArtifactType.CODE:
        title_section = f"And its title is (do NOT include this in your response):\n{artifact_meta_tool_call.title}"

    try:
        return OPTIONALLY_UPDATE_META_PROMPT.format(
            artifactType=artifact_meta_tool_call.type.value, # Use .value for enum to string
            artifactTitle=title_section
        )
    except KeyError as e:
        print(f"Error: Placeholder {{{e.args[0]}}} missing in OPTIONALLY_UPDATE_META_PROMPT. Check prompts.py.")
        # Fallback or re-raise
        if e.args[0] == "artifactType":
            return OPTIONALLY_UPDATE_META_PROMPT.replace("{artifactTitle}", title_section).replace("{artifactType}", artifact_meta_tool_call.type.value)
        elif e.args[0] == "artifactTitle":
            return OPTIONALLY_UPDATE_META_PROMPT.replace("{artifactType}", artifact_meta_tool_call.type.value).replace("{artifactTitle}", title_section)
        raise

class BuildPromptArgs(TypedDict):
    artifact_content_str: str
    memories_as_string: str
    is_new_type: bool # True if artifact type changed (e.g. text -> code)
    artifact_meta_tool_call: OptionallyUpdateArtifactMetaSchema


def build_prompt(args: BuildPromptArgs) -> str:
    """
    Builds the main prompt for the LLM that will perform the artifact rewrite.
    """
    # The meta_prompt is only included if the artifact type is new (changed during meta update)
    # This tells the LLM to be mindful of the new type/title.
    update_meta_prompt_section = _build_meta_prompt(args["artifact_meta_tool_call"]) if args["is_new_type"] else ""

    try:
        return UPDATE_ENTIRE_ARTIFACT_PROMPT.format(
            artifactContent=args["artifact_content_str"],
            reflections=args["memories_as_string"],
            updateMetaPrompt=update_meta_prompt_section # This is the conditional part
        )
    except KeyError as e:
        print(f"Error: Placeholder {{{e.args[0]}}} missing in UPDATE_ENTIRE_ARTIFACT_PROMPT. Check prompts.py.")
        # Fallback or re-raise, attempting to fill manually for known keys
        if e.args[0] == "updateMetaPrompt": # Most likely if it's empty and format expects it
             return UPDATE_ENTIRE_ARTIFACT_PROMPT.replace("{artifactContent}", args["artifact_content_str"]).replace("{reflections}", args["memories_as_string"]).replace("{updateMetaPrompt}", update_meta_prompt_section)
        # Add more specific fallbacks if necessary
        raise


class CreateNewArtifactContentArgs(TypedDict):
    artifact_type: ArtifactType # The determined new type for the artifact
    state: OpenCanvasState
    current_artifact_content_model: Union[ArtifactCodeV3, ArtifactMarkdownV3] # The *original* content model being rewritten
    artifact_meta_tool_call: OptionallyUpdateArtifactMetaSchema # Metadata determined by the first LLM call
    new_content_text: str # The actual new text/code generated by the main LLM


def _get_language_for_new_content(
    artifact_meta_tool_call: OptionallyUpdateArtifactMetaSchema,
    current_artifact_content_model: Union[ArtifactCodeV3, ArtifactMarkdownV3]
) -> ProgrammingLanguageOptions:
    """
    Determines the language for the new artifact content.
    Priority:
    1. Language from the artifact_meta_tool_call (if provided by LLM and type is CODE).
    2. Language from the current_artifact_content_model (if it's code).
    3. Default to OTHER.
    """
    if artifact_meta_tool_call.type == ArtifactType.CODE:
        if artifact_meta_tool_call.language: # LLM specified a language for the new code
            return artifact_meta_tool_call.language
        # If LLM didn't specify, but original was code, keep original language
        if is_artifact_code_content(current_artifact_content_model) and isinstance(current_artifact_content_model, ArtifactCodeV3):
            return current_artifact_content_model.language
    return ProgrammingLanguageOptions.OTHER # Default for text or if code language is undetermined


def create_new_artifact_content(args: CreateNewArtifactContentArgs) -> Union[ArtifactCodeV3, ArtifactMarkdownV3]:
    """
    Creates the new ArtifactCodeV3 or ArtifactMarkdownV3 object for the rewritten artifact.
    This new content item will be added to the existing artifact's 'contents' list.
    """
    existing_artifact_v3 = args["state"].get("artifact")
    # New content gets the next available index.
    # If current contents list has 1 item (at index 0), new index is 1.
    new_content_index = len(existing_artifact_v3.contents) if existing_artifact_v3 and existing_artifact_v3.contents else 0

    # Title: Use title from meta call if provided, otherwise keep original title.
    new_title = args["artifact_meta_tool_call"].title or args["current_artifact_content_model"].title

    base_content_params = {
        "index": new_content_index,
        "title": new_title,
    }

    if args["artifact_type"] == ArtifactType.CODE:
        return ArtifactCodeV3(
            **base_content_params,
            type=ArtifactType.CODE, # Explicitly set type
            language=_get_language_for_new_content(args["artifact_meta_tool_call"], args["current_artifact_content_model"]),
            code=args["new_content_text"],
        )
    else: # ArtifactType.TEXT
        return ArtifactMarkdownV3(
            **base_content_params,
            type=ArtifactType.TEXT, # Explicitly set type
            fullMarkdown=args["new_content_text"],
        )
