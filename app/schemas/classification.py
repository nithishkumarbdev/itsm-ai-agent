from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Category(StrEnum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    VPN = "vpn"
    ACCESS = "access"
    ACCOUNT = "account"
    EMAIL = "email"
    SECURITY = "security"
    OTHER = "other"


class Priority(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Impact(StrEnum):
    INDIVIDUAL = "individual"
    DEPARTMENT = "department"
    ORGANIZATION = "organization"


class TicketClassification(BaseModel):
    """What the LLM *proposes* for a ticket. Nothing is trusted until it validates against this.

    `confidence` is the model's own stated confidence. It is a signal, not a measured
    probability: 0.9 does not mean "90% likely to be correct".
    """

    # Unknown fields from the model (for example an "assigned_team") are dropped, never
    # stored or used. The team is decided by the backend (app/services/routing.py).
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    category: Category
    priority: Priority
    urgency: Urgency
    impact: Impact
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=500, description="One short sentence, no reasoning trace.")

    @field_validator("confidence", mode="before")
    @classmethod
    def confidence_must_be_a_number(cls, value):
        # Refuse "0.9" (a string) and True/False, which pydantic would otherwise coerce.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return value


# The same contract, written as JSON Schema for the provider. Hand-written and inlined
# (no $ref) so every provider handles it; a test checks it stays in sync with the model above.
CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": [c.value for c in Category]},
        "priority": {"type": "string", "enum": [p.value for p in Priority]},
        "urgency": {"type": "string", "enum": [u.value for u in Urgency]},
        "impact": {"type": "string", "enum": [i.value for i in Impact]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 500},
    },
    "required": ["category", "priority", "urgency", "impact", "confidence", "reason"],
    "additionalProperties": False,
}
