from typing import Any

import anthropic

from app.llm.base import LLMError

MAX_OUTPUT_TOKENS = 512  # a classification is tiny


class AnthropicClient:
    """LLMClient backed by the Anthropic Messages API.

    Structured output: the model is forced to call one "tool" whose input schema is
    our JSON schema, so the reply is a JSON object, not free text we would have to parse.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str | None,
        timeout: float,
        max_retries: int,
        sdk_client: Any = None,  # lets tests pass a fake instead of the real SDK
    ):
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._sdk_client = sdk_client

    @property
    def name(self) -> str:
        return f"anthropic:{self._model or 'unconfigured'}"

    def _sdk(self):
        # Built lazily so the app (and /health, /tickets, ...) starts fine without LLM settings.
        if self._sdk_client is None:
            if not self._api_key or not self._model:
                raise LLMError("provider_error", "LLM_API_KEY and LLM_MODEL are not configured")
            self._sdk_client = anthropic.Anthropic(
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._sdk_client

    def generate_structured(
        self, *, system: str, user: str, schema_name: str, schema: dict[str, Any]
    ) -> Any:
        sdk = self._sdk()
        try:
            response = sdk.messages.create(
                model=self._model,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.0,  # classification should be as repeatable as possible
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[
                    {
                        "name": schema_name,
                        "description": "Return the structured result.",
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": schema_name},
            )
        except anthropic.APITimeoutError as exc:
            raise LLMError("timeout") from exc
        except anthropic.APIError as exc:
            raise LLMError("provider_error", type(exc).__name__) from exc

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == schema_name:
                return block.input
        raise LLMError("provider_error", "response contained no structured output")
