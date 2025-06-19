# app/agents/nodes/rewrite_artifact_helpers/schemas.py
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from app.schemas.common import ProgrammingLanguageOptions, ArtifactType

class OptionallyUpdateArtifactMetaSchema(BaseModel):
    type: ArtifactType = Field(..., description="The type of the artifact content (e.g. 'text' or 'code').")
    title: Optional[str] = Field(
        default=None,
        description="The new title for the artifact. This should ONLY be updated if the user's request implies a change in the overall subject or topic of the artifact. Otherwise, do not provide this field."
    )
    language: Optional[ProgrammingLanguageOptions] = Field(
        default=None, # Changed default to None to better distinguish if LLM omits it vs. sets to OTHER
        description="The programming language of the code artifact. If the artifact type is 'code', this should be populated. If the type is 'text', this should be 'other' or omitted."
    )

    # Pydantic v2 validator example (if needed, currently not strictly enforced by this schema alone)
    # from pydantic import model_validator
    # @model_validator(mode='after')
    # def check_language_for_type(self) -> 'OptionallyUpdateArtifactMetaSchema':
    #     if self.type == ArtifactType.TEXT and self.language is not None and self.language != ProgrammingLanguageOptions.OTHER:
    #         # This would be a good place for a warning or auto-correction if desired.
    #         # For now, the description guides the LLM.
    #         # self.language = ProgrammingLanguageOptions.OTHER
    #         pass
    #     if self.type == ArtifactType.CODE and self.language is None:
    #         # If it's code and language is omitted, default it to 'other'.
    #         self.language = ProgrammingLanguageOptions.OTHER
    #     return self
