"""Tests for the top-level query convenience API."""

from __future__ import annotations

import importlib

import pytest

from adal_agent_sdk.options import AdalAgentOptions


class FakeClient:
    instances = []

    def __init__(self, options):
        self.options = options
        self.prompts = []
        self.closed = False
        FakeClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.closed = True

    async def query(self, prompt):
        self.prompts.append(prompt)

    async def receive_events(self):
        yield {"type": "assistant.delta", "text": "hello"}
        yield {"type": "command.completed", "result": {"ok": True}}


@pytest.fixture(autouse=True)
def reset_fake_client():
    FakeClient.instances = []


@pytest.mark.anyio
async def test_query_streams_events_with_keyword_options(monkeypatch):
    query_module = importlib.import_module("adal_agent_sdk.query")
    monkeypatch.setattr(query_module, "AdalAgentClient", FakeClient)

    events = [
        event
        async for event in query_module.query(
            "Say hello",
            workspace="/tmp/example",
            permission_mode="yolo",
            agent_mode="research",
        )
    ]

    assert events == [
        {"type": "assistant.delta", "text": "hello"},
        {"type": "command.completed", "result": {"ok": True}},
    ]

    client = FakeClient.instances[0]
    assert client.prompts == ["Say hello"]
    assert client.options.workspace == "/tmp/example"
    assert client.options.permission_mode == "yolo"
    assert client.options.agent_mode == "research"
    assert client.closed


@pytest.mark.anyio
async def test_query_accepts_options_object(monkeypatch):
    query_module = importlib.import_module("adal_agent_sdk.query")
    monkeypatch.setattr(query_module, "AdalAgentClient", FakeClient)

    options = AdalAgentOptions(workspace="/tmp/options", model="test-model")
    events = [event async for event in query_module.query("Use options", options=options)]

    assert events[-1]["type"] == "command.completed"
    assert FakeClient.instances[0].options is options


@pytest.mark.anyio
async def test_query_rejects_options_and_kwargs():
    query_module = importlib.import_module("adal_agent_sdk.query")

    with pytest.raises(ValueError, match="either options or keyword options"):
        async for _ in query_module.query(
            "Invalid",
            options=AdalAgentOptions(workspace="/tmp/options"),
            workspace="/tmp/kwargs",
        ):
            pass
