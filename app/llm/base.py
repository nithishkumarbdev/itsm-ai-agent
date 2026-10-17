from typing import Any, Literal, Protocol


class LLMError(Exception):
    """The provider could not give us a response (timeout, outage, bad key, ...).

    This is different from a response we *reject*: that is a validation problem,
    handled by the classification service.
    """

    def __init__(self, kind: Literal["timeout", "provider_error"], detail: str | None = None):
        super().__init__(f"LLM {kind}" + (f": {detail}" if detail else ""))
        self.kind = kind
        self.detail = detail


class LLMClient(Protocol):
    """All the app needs from an LLM: ask for a JSON object that matches a schema."""

    name: str  # provider and model, for logs

    def generate_structured(
        self, *, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> Any:
        """Return the model's structured output (a dict). Raises LLMError on provider failure.

        The result is untrusted: the caller must validate it before using it.
        """
        ...
