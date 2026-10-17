"""The real provider adapter, driven by a fake SDK object. No network, no API key."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMError

SCHEMA = {"type": "object", "properties": {}}


class FakeSDK:
    """Mimics `anthropic.Anthropic`: has .messages.create(...)."""

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.requests = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.result


def make_client(sdk):
    return AnthropicClient(
        api_key="test-key", model="test-model", timeout=5, max_retries=0, sdk_client=sdk
    )


def call(client):
    return client.generate_structured(system="SYS", user="USER", schema_name="submit_x", schema=SCHEMA)


def tool_use(name, payload):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name=name, input=payload)])


def test_returns_the_tool_input_and_forces_the_tool():
    sdk = FakeSDK(result=tool_use("submit_x", {"category": "vpn"}))

    result = call(make_client(sdk))

    assert result == {"category": "vpn"}
    request = sdk.requests[0]
    assert request["model"] == "test-model"
    assert request["system"] == "SYS"
    assert request["messages"] == [{"role": "user", "content": "USER"}]
    assert request["tools"][0]["name"] == "submit_x"
    assert request["tools"][0]["input_schema"] == SCHEMA
    assert request["tool_choice"] == {"type": "tool", "name": "submit_x"}


def test_response_without_structured_output_is_a_provider_error():
    text_only = SimpleNamespace(content=[SimpleNamespace(type="text", text="Category: VPN")])

    with pytest.raises(LLMError) as excinfo:
        call(make_client(FakeSDK(result=text_only)))

    assert excinfo.value.kind == "provider_error"  # free text is never parsed


def test_timeout_is_reported_as_timeout():
    request = httpx.Request("POST", "https://example.invalid")
    sdk = FakeSDK(error=anthropic.APITimeoutError(request=request))

    with pytest.raises(LLMError) as excinfo:
        call(make_client(sdk))

    assert excinfo.value.kind == "timeout"


def test_other_sdk_errors_are_provider_errors_with_only_the_class_name():
    request = httpx.Request("POST", "https://example.invalid")
    sdk = FakeSDK(error=anthropic.APIConnectionError(request=request))

    with pytest.raises(LLMError) as excinfo:
        call(make_client(sdk))

    assert excinfo.value.kind == "provider_error"
    assert excinfo.value.detail == "APIConnectionError"


@pytest.mark.parametrize(
    ("api_key", "model"),
    [(None, "some-model"), ("some-key", None), (None, None), ("", "")],
)
def test_missing_configuration_fails_safely_without_any_network_call(api_key, model):
    client = AnthropicClient(api_key=api_key, model=model, timeout=5, max_retries=0)

    with pytest.raises(LLMError) as excinfo:
        call(client)

    assert excinfo.value.kind == "provider_error"
    assert "not configured" in excinfo.value.detail


def test_name_identifies_provider_and_model_but_never_the_key():
    assert make_client(FakeSDK()).name == "anthropic:test-model"
    assert "test-key" not in make_client(FakeSDK()).name
