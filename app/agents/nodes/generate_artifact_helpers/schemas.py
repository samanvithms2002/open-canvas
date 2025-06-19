# app/agents/nodes/generate_artifact_helpers/schemas.py
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from app.schemas.common import ProgrammingLanguageOptions, ArtifactType

class ArtifactToolSchema(BaseModel):
    type: ArtifactType = Field(..., description="The content type of the artifact generated (e.g., 'text', 'code').")
    language: Optional[ProgrammingLanguageOptions] = Field(
        default=ProgrammingLanguageOptions.OTHER,
        description=(
            "The programming language of the artifact if type is 'code'. "
            "Should be one of the predefined languages or 'other'. "
            "If type is 'text', this should ideally be 'other' or None."
        )
    )
    artifact: str = Field(..., description="The main content of the artifact to be generated.")
    title: str = Field(..., description="A concise title for the artifact, preferably under 5 words.")

    # Ensure that language is set to OTHER if type is TEXT
    # Pydantic v2 validator syntax, adjust if using v1 only features elsewhere
    # @validator('language', pre=True, always=True)
    # def check_language_for_text_type(cls, v, values):
    #     if values.get('type') == ArtifactType.TEXT and v != ProgrammingLanguageOptions.OTHER:
    #         # Users of the schema should ensure this, or we can force it.
    #         # Forcing it might silently change data from LLM.
    #         # print("Warning: Language for TEXT artifact type should be OTHER. Forcing to OTHER.")
    #         # return ProgrammingLanguageOptions.OTHER
    #         pass # For now, let the LLM decide, but the description guides it.
    #     if values.get('type') == ArtifactType.CODE and v is None:
    #          return ProgrammingLanguageOptions.OTHER # Default for code if not specified
    #     return v
