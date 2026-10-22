import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceReference(BaseModel):
    """One piece of evidence the model says it used. Unknown fields are dropped: the model
    cannot attach anything to a citation beyond which chunk it is."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: uuid.UUID


class TicketResolution(BaseModel):
    """What the LLM *proposes* as a resolution. Nothing is trusted until it validates against
    this AND every source_references entry is confirmed (by the caller) to be a chunk that
    was actually retrieved for this ticket — the schema alone cannot check that.

    `resolution_confidence` is the model's own stated confidence, like classification's. It is
    a signal, not a measured probability, and Phase 7 does not act on it (no auto-resolution).
    """

    # Unknown fields (for example the model deciding to add "action_taken") are dropped,
    # never stored or used, the same defensive stance as classification.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    suggested_resolution: str = Field(min_length=1, max_length=2000)
    resolution_confidence: float = Field(ge=0.0, le=1.0)
    source_references: list[SourceReference] = Field(
        default_factory=list, max_length=20, description="Chunks this resolution is grounded in. Empty means insufficient evidence."
    )
    reason: str = Field(min_length=1, max_length=500, description="One short sentence, no reasoning trace.")

    @field_validator("resolution_confidence", mode="before")
    @classmethod
    def confidence_must_be_a_number(cls, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return value
