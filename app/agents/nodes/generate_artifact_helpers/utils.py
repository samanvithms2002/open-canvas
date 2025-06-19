# app/agents/nodes/generate_artifact_helpers/utils.py
from typing import Union # Ensure Union is imported

from app.agents.prompts import NEW_ARTIFACT_PROMPT
from app.schemas.common import ArtifactCodeV3, ArtifactMarkdownV3, ProgrammingLanguageOptions, ArtifactType
from .schemas import ArtifactToolSchema # Local import for ArtifactToolSchema

def format_new_artifact_prompt(memories_as_string: str, model_name: str) -> str:
    """
    Formats the prompt for generating a new artifact.

    Args:
        memories_as_string: A string containing reflections/memories.
        model_name: The name of the model being used, to conditionally add instructions.

    Returns:
        A formatted prompt string.
    """
    disable_cot_instruction = ""
    # Model names are like "gpt-4o-mini", "azure/gpt-4o-mini", "claude-3-opus-20240229"
    # Check if 'claude' is part of the model_name string.
    if "claude" in model_name.lower():
        disable_cot_instruction = "\n\nIMPORTANT: Do NOT perform chain of thought beforehand. Instead, go STRAIGHT to generating the tool response. This is VERY important."

    # NEW_ARTIFACT_PROMPT is expected to have {reflections} and {disableChainOfThought} placeholders.
    try:
        return NEW_ARTIFACT_PROMPT.format(
            reflections=memories_as_string,
            disableChainOfThought=disable_cot_instruction
        )
    except KeyError as e:
        print(f"Error: Placeholder {{{e.args[0]}}} missing in NEW_ARTIFACT_PROMPT. Check prompts.py.")
        # Fallback to a version without the missing placeholder, or re-raise
        # For now, try to continue if one is missing, though this indicates a prompt setup issue.
        if e.args[0] == "reflections":
            return NEW_ARTIFACT_PROMPT.replace("{disableChainOfThought}", disable_cot_instruction).replace("{reflections}", memories_as_string)
        elif e.args[0] == "disableChainOfThought":
            return NEW_ARTIFACT_PROMPT.replace("{reflections}", memories_as_string).replace("{disableChainOfThought}", disable_cot_instruction)
        raise # Re-raise if it's another key


def create_artifact_content(tool_call_args: ArtifactToolSchema) -> Union[ArtifactCodeV3, ArtifactMarkdownV3]:
    """
    Creates either an ArtifactCodeV3 or ArtifactMarkdownV3 object based on the
    parsed arguments from an LLM tool call.

    Args:
        tool_call_args: Parsed arguments from the ArtifactToolSchema.

    Returns:
        An instance of ArtifactCodeV3 or ArtifactMarkdownV3.
    """
    artifact_type = tool_call_args.type

    # The index is 1 because this function is for creating a *new* artifact.
    # In the TS code, new artifacts are added with index 1, and subsequent updates
    # might increment this or manage versions differently.
    # The `ArtifactV3` Pydantic model has `currentIndex` which points to one of the
    # contents in its list. For a new artifact, it will have one content item.

    if artifact_type == ArtifactType.CODE:
        # Ensure language is set; default to OTHER if LLM provides None for a code artifact
        language = tool_call_args.language if tool_call_args.language is not None else ProgrammingLanguageOptions.OTHER
        return ArtifactCodeV3(
            index=0, # Per TS logic, new artifact contents start at index 0 for the first item
            type=ArtifactType.CODE,
            title=tool_call_args.title,
            code=tool_call_args.artifact,
            language=language
        )
    else: # ArtifactType.TEXT
        return ArtifactMarkdownV3(
            index=0, # Per TS logic, new artifact contents start at index 0
            type=ArtifactType.TEXT,
            title=tool_call_args.title,
            fullMarkdown=tool_call_args.artifact
        )
