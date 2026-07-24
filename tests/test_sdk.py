"""Unit tests for the AdaL Agent SDK.

Tests the SDK's logic without spawning a real SDK runtime — uses a
mock transport. Covers:
- PermissionResult factory and ToolPermissionContext
- SDK protocol constants (parity with TS)
- Client event streaming, error handling, approval round-trip

Run: pytest tests/test_sdk.py -v
"""

import anyio
import pytest

from adal_agent_sdk.client import (
    CONTROL_RESPONSE_WATCHDOG_TIMEOUT,
    AdalAgentClient,
    WIRE_CAPABILITIES,
    WIRE_FRAME_TYPES,
)
from adal_agent_sdk.exceptions import (
    ProtocolError,
    QueryError,
    RuntimeUnresponsive,
    RuntimeUpgradeRequired,
    SdkRuntimeError,
)
from adal_agent_sdk.options import AdalAgentOptions
from adal_agent_sdk.permission import (
    PermissionResult,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)


# ---------------------------------------------------------------------------
# PermissionResult + ToolPermissionContext
# ---------------------------------------------------------------------------

class TestPermissionResultAllow:
    def test_default_allow(self):
        result = PermissionResult.allow()
        assert isinstance(result, PermissionResultAllow)
        assert result.behavior == "allow"
        assert result.updated_input is None

    def test_allow_with_updated_input(self):
        result = PermissionResult.allow(updated_input={"command": "echo hi"})
        assert isinstance(result, PermissionResultAllow)
        assert result.behavior == "allow"
        assert result.updated_input == {"command": "echo hi"}


class TestPermissionResultDeny:
    def test_default_deny(self):
        result = PermissionResult.deny()
        assert isinstance(result, PermissionResultDeny)
        assert result.behavior == "deny"
        assert result.message == ""

    def test_deny_with_message(self):
        result = PermissionResult.deny("Dangerous command")
        assert isinstance(result, PermissionResultDeny)
        assert result.behavior == "deny"
        assert result.message == "Dangerous command"


class TestToolPermissionContext:
    def test_basic_context(self):
        ctx = ToolPermissionContext(tool_call_id="tc-1")
        assert ctx.tool_call_id == "tc-1"
        assert ctx.confirmation is None
        assert ctx.display is None

    def test_context_with_metadata(self):
        ctx = ToolPermissionContext(
            tool_call_id="tc-2",
            confirmation={"type": "diff", "text": "+line"},
            display={"displayName": "Edit"},
        )
        assert ctx.tool_call_id == "tc-2"
        assert ctx.confirmation == {"type": "diff", "text": "+line"}
        assert ctx.display == {"displayName": "Edit"}


# ---------------------------------------------------------------------------
# SDK protocol constants (parity with CLI TS sdkProtocol.ts)
# ---------------------------------------------------------------------------

class TestWireProtocolConstants:
    def test_frame_types_match_ts_protocol(self):
        assert WIRE_FRAME_TYPES["READY"] == "ready"
        assert WIRE_FRAME_TYPES["CONTROL_REQUEST"] == "control_request"
        assert WIRE_FRAME_TYPES["CONTROL_RESPONSE"] == "control_response"
        assert WIRE_FRAME_TYPES["ERROR"] == "error"

    def test_frame_types_do_not_collide_with_event_types(self):
        event_types = {
            "command.started", "command.completed", "command.failed",
            "assistant.delta", "assistant.message.completed",
            "tool.started", "tool.completed",
            "tool.permission.requested",
        }
        frame_values = set(WIRE_FRAME_TYPES.values())
        assert frame_values.isdisjoint(event_types)


# ---------------------------------------------------------------------------
# Client event streaming (mock transport)
# ---------------------------------------------------------------------------

class MockTransport:
    """Mock transport that returns pre-configured messages."""

    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self._sent = []
        self.closed = False

    async def start(self):
        pass

    async def send(self, message):
        self._sent.append(message)

    async def receive(self, timeout=None):
        if not self._messages:
            raise anyio.EndOfStream()
        return self._messages.pop(0)

    async def close(self):
        self.closed = True

    @property
    def sent_messages(self):
        return self._sent


def make_client(transport=None, **kwargs):
    """Create a client with a mock transport for testing."""
    options = AdalAgentOptions(**kwargs)
    client = AdalAgentClient(options)
    if transport is not None:
        client._transport = transport
    return client


class TestClientAgentModeInitialization:
    @pytest.mark.anyio
    async def test_sends_agent_mode_and_records_confirmed_ready_state(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["READY"],
                "session_id": "session-1",
                "protocol_version": 1,
                "capabilities": [WIRE_CAPABILITIES["AGENT_MODE"]],
                "active_agent_mode": "research",
            },
        ])
        client = make_client(
            transport=transport,
            workspace="/tmp",
            agent_mode="  research  ",
        )

        await client._initialize()

        assert transport.sent_messages[0]["agent_mode"] == "research"
        assert client.session_id == "session-1"
        assert client.active_agent_mode == "research"
        assert client.capabilities == frozenset({WIRE_CAPABILITIES["AGENT_MODE"]})

    @pytest.mark.anyio
    async def test_whitespace_only_mode_is_treated_as_not_requested(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["READY"],
                "session_id": "old-runtime",
                "protocol_version": 1,
            },
        ])
        client = make_client(transport=transport, agent_mode="   ")

        await client._initialize()

        assert "agent_mode" not in transport.sent_messages[0]
        assert client.active_agent_mode is None
        assert not transport.closed

    @pytest.mark.anyio
    async def test_requested_mode_requires_new_runtime_capability(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["READY"],
                "session_id": "old-runtime",
                "protocol_version": 1,
            },
        ])
        client = make_client(transport=transport, agent_mode="research")

        with pytest.raises(RuntimeUpgradeRequired) as exc_info:
            await client._initialize()

        assert exc_info.value.code == "RUNTIME_UPGRADE_REQUIRED"
        assert exc_info.value.capability == WIRE_CAPABILITIES["AGENT_MODE"]
        assert transport.closed

    @pytest.mark.anyio
    async def test_old_runtime_still_works_when_mode_is_not_requested(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["READY"],
                "session_id": "old-runtime",
                "protocol_version": 1,
            },
        ])
        client = make_client(transport=transport)

        await client._initialize()

        assert client.capabilities == frozenset()
        assert client.active_agent_mode is None
        assert not transport.closed

    @pytest.mark.anyio
    async def test_mismatched_confirmed_mode_is_protocol_error(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["READY"],
                "session_id": "session-2",
                "protocol_version": 1,
                "capabilities": [WIRE_CAPABILITIES["AGENT_MODE"]],
                "active_agent_mode": "coding",
            },
        ])
        client = make_client(transport=transport, agent_mode="research")

        with pytest.raises(ProtocolError, match="expected 'research'"):
            await client._initialize()

        assert transport.closed

    @pytest.mark.anyio
    async def test_malformed_capabilities_are_rejected(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["READY"],
                "session_id": "session-3",
                "protocol_version": 1,
                "capabilities": "agent-mode.v1",
            },
        ])
        client = make_client(transport=transport)

        with pytest.raises(ProtocolError, match="list of strings"):
            await client._initialize()

        assert transport.closed

    @pytest.mark.anyio
    async def test_startup_health_watchdog_is_fatal(self, monkeypatch):
        class BlockingTransport(MockTransport):
            async def receive(self, timeout=None):
                await anyio.sleep_forever()

        monkeypatch.setattr(
            "adal_agent_sdk.client.RUNTIME_STARTUP_WATCHDOG_TIMEOUT",
            0.01,
        )
        transport = BlockingTransport()
        client = make_client(transport=transport, agent_mode="research")

        with pytest.raises(RuntimeUnresponsive) as exc_info:
            await client._initialize()

        assert exc_info.value.code == "RUNTIME_UNRESPONSIVE"
        assert exc_info.value.operation == "runtime initialization"
        assert transport.closed
        assert client._closed
        assert not client._initialized


class TestClientAgentModeControl:
    @pytest.mark.anyio
    async def test_switch_uses_the_generic_control_watchdog(self):
        client = make_client(transport=MockTransport())
        client._initialized = True
        client._capabilities = frozenset({WIRE_CAPABILITIES["AGENT_MODE"]})
        captured = {}

        async def send_control_request(
            request,
            timeout=CONTROL_RESPONSE_WATCHDOG_TIMEOUT,
        ):
            captured["request"] = request
            captured["timeout"] = timeout
            return {"active_mode": "research"}

        client._send_control_request = send_control_request

        await client.set_agent_mode("research")

        assert captured["request"]["subtype"] == "set_agent_mode"
        assert captured["timeout"] == CONTROL_RESPONSE_WATCHDOG_TIMEOUT

    @pytest.mark.anyio
    async def test_switch_updates_state_from_confirmed_response(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["CONTROL_RESPONSE"],
                "request_id": "req_1_ignored",
                "subtype": "success",
                "response": {"active_mode": "research"},
            },
        ])
        client = make_client(transport=transport)
        client._initialized = True
        client._capabilities = frozenset({WIRE_CAPABILITIES["AGENT_MODE"]})
        client._active_agent_mode = "coding"

        # Match the generated request ID without coupling the test to its UUID.
        original_receive = transport.receive

        async def receive_matching(timeout=None):
            message = await original_receive(timeout)
            message["request_id"] = transport.sent_messages[-1]["request_id"]
            return message

        transport.receive = receive_matching

        response = await client.set_agent_mode("research")

        assert response == {"active_mode": "research"}
        assert client.active_agent_mode == "research"
        request = transport.sent_messages[0]
        assert request["request"] == {
            "subtype": "set_agent_mode",
            "mode": "research",
        }

    @pytest.mark.anyio
    async def test_structured_runtime_timeout_preserves_error_code(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["CONTROL_RESPONSE"],
                "request_id": "placeholder",
                "subtype": "error",
                "error": "Agent mode switch timed out after 60 seconds",
                "code": "AGENT_MODE_SWITCH_TIMEOUT",
            },
        ])
        client = make_client(transport=transport)
        client._initialized = True
        client._capabilities = frozenset({WIRE_CAPABILITIES["AGENT_MODE"]})

        original_receive = transport.receive

        async def receive_matching(timeout=None):
            message = await original_receive(timeout)
            message["request_id"] = transport.sent_messages[-1]["request_id"]
            return message

        transport.receive = receive_matching

        with pytest.raises(QueryError) as exc_info:
            await client.set_agent_mode("research")

        assert exc_info.value.code == "AGENT_MODE_SWITCH_TIMEOUT"
        assert not transport.closed

    @pytest.mark.anyio
    async def test_control_watchdog_invalidates_unresponsive_session(self):
        class BlockingTransport(MockTransport):
            async def receive(self, timeout=None):
                await anyio.sleep_forever()

        transport = BlockingTransport()
        client = make_client(transport=transport)
        client._initialized = True
        client._capabilities = frozenset({WIRE_CAPABILITIES["AGENT_MODE"]})

        with pytest.raises(RuntimeUnresponsive) as exc_info:
            await client._send_control_request(
                {"subtype": "set_agent_mode", "mode": "research"},
                timeout=0.01,
            )

        assert exc_info.value.code == "RUNTIME_UNRESPONSIVE"
        assert exc_info.value.operation == "set_agent_mode"
        assert transport.closed
        assert client._closed
        assert not client._initialized

    @pytest.mark.anyio
    async def test_control_watchdog_invalidates_active_reader_path(self):
        transport = MockTransport()
        client = make_client(transport=transport)
        client._initialized = True
        client._receive_events_active = True

        with pytest.raises(RuntimeUnresponsive):
            await client._send_control_request(
                {"subtype": "set_model", "model": "test-model"},
                timeout=0.01,
            )

        assert transport.closed
        assert client._pending_control_responses == {}
        assert client._pending_control_results == {}
        assert not client._initialized

    @pytest.mark.anyio
    async def test_switch_requires_capability(self):
        client = make_client(transport=MockTransport())
        client._initialized = True

        with pytest.raises(RuntimeUpgradeRequired):
            await client.set_agent_mode("research")

    @pytest.mark.anyio
    async def test_switch_is_rejected_while_query_is_active(self):
        client = make_client(transport=MockTransport())
        client._initialized = True
        client._capabilities = frozenset({WIRE_CAPABILITIES["AGENT_MODE"]})
        client._query_active = True

        with pytest.raises(QueryError, match="while a query is active"):
            await client.set_agent_mode("research")

    @pytest.mark.anyio
    async def test_missing_confirmation_does_not_change_local_mode(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["CONTROL_RESPONSE"],
                "request_id": "placeholder",
                "subtype": "success",
                "response": {"mode": "research"},
            },
        ])
        client = make_client(transport=transport)
        client._initialized = True
        client._capabilities = frozenset({WIRE_CAPABILITIES["AGENT_MODE"]})
        client._active_agent_mode = "coding"

        original_receive = transport.receive

        async def receive_matching(timeout=None):
            message = await original_receive(timeout)
            message["request_id"] = transport.sent_messages[-1]["request_id"]
            return message

        transport.receive = receive_matching

        with pytest.raises(ProtocolError, match="confirmed active_mode"):
            await client.set_agent_mode("research")

        assert client.active_agent_mode == "coding"


class TestClientReceiveEvents:
    @pytest.mark.anyio
    async def test_query_stays_busy_until_terminal_event_is_consumed(self):
        transport = MockTransport(messages=[
            {"type": "command.completed", "result": {"ok": True}},
        ])
        client = make_client(transport=transport, workspace="/tmp")
        client._initialized = True

        await client.query("first")
        with pytest.raises(QueryError, match="already active"):
            await client.query("second")

        events = [event async for event in client.receive_events()]
        assert events[-1]["type"] == "command.completed"

        await client.query("third")
        assert transport.sent_messages[-1]["input"] == "third"

    @pytest.mark.anyio
    async def test_control_request_can_follow_terminal_event_inside_loop(self):
        transport = MockTransport(messages=[
            {"type": "command.completed", "result": {"ok": True}},
            {
                "type": WIRE_FRAME_TYPES["CONTROL_RESPONSE"],
                "request_id": "placeholder",
                "subtype": "success",
                "response": {"active_mode": "research"},
            },
        ])
        client = make_client(transport=transport, workspace="/tmp")
        client._initialized = True
        client._capabilities = frozenset({WIRE_CAPABILITIES["AGENT_MODE"]})
        client._active_agent_mode = "coding"

        original_receive = transport.receive

        async def receive_matching(timeout=None):
            message = await original_receive(timeout)
            if message["type"] == WIRE_FRAME_TYPES["CONTROL_RESPONSE"]:
                message["request_id"] = transport.sent_messages[-1]["request_id"]
            return message

        transport.receive = receive_matching

        await client.query("first")
        async for event in client.receive_events():
            if event["type"] == "command.completed":
                await client.set_agent_mode("research")

        assert client.active_agent_mode == "research"

    @pytest.mark.anyio
    async def test_yields_regular_events_until_command_completed(self):
        transport = MockTransport(messages=[
            {"type": "assistant.delta", "text": "hello"},
            {"type": "tool.started", "toolCallId": "t1", "name": "bash"},
            {"type": "command.completed", "result": {"ok": True}},
        ])
        client = make_client(transport=transport, workspace="/tmp")
        client._initialized = True

        events = []
        async for event in client.receive_events():
            events.append(event)

        assert len(events) == 3
        assert events[0]["type"] == "assistant.delta"
        assert events[1]["type"] == "tool.started"
        assert events[2]["type"] == "command.completed"

    @pytest.mark.anyio
    async def test_stops_on_command_failed(self):
        transport = MockTransport(messages=[
            {"type": "command.failed", "error": {"message": "oops"}},
        ])
        client = make_client(transport=transport, workspace="/tmp")
        client._initialized = True

        events = []
        async for event in client.receive_events():
            events.append(event)

        assert len(events) == 1
        assert events[0]["type"] == "command.failed"

    @pytest.mark.anyio
    async def test_raises_on_error_frame(self):
        transport = MockTransport(messages=[
            {"type": "error", "message": "Something broke", "code": "TEST"},
        ])
        client = make_client(transport=transport, workspace="/tmp")
        client._initialized = True

        with pytest.raises(SdkRuntimeError, match="Something broke"):
            async for _ in client.receive_events():
                pass

    @pytest.mark.anyio
    async def test_intercepts_control_request_for_approval(self):
        """Approval control_request should invoke can_use_tool and send response."""
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["CONTROL_REQUEST"],
                "request_id": "approval_1",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_call_id": "tc-1",
                    "tool_name": "bash",
                    "input": {"command": "ls"},
                },
            },
            {"type": "command.completed", "result": {"ok": True}},
        ])

        callback_called = False

        async def can_use_tool(tool_name, tool_input, ctx):
            nonlocal callback_called
            callback_called = True
            assert tool_name == "bash"
            assert tool_input == {"command": "ls"}
            return PermissionResult.allow()

        client = make_client(
            transport=transport,
            workspace="/tmp",
            can_use_tool=can_use_tool,
        )
        client._initialized = True

        events = []
        async for event in client.receive_events():
            events.append(event)

        # The control_request should NOT be yielded — only command.completed
        assert len(events) == 1
        assert events[0]["type"] == "command.completed"
        assert callback_called

        # Verify the control_response was sent back
        sent = transport.sent_messages
        assert len(sent) == 1
        assert sent[0]["type"] == "control_response"
        assert sent[0]["request_id"] == "approval_1"
        assert sent[0]["response"]["behavior"] == "allow"

    @pytest.mark.anyio
    async def test_approval_deny_sends_deny_response(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["CONTROL_REQUEST"],
                "request_id": "approval_2",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_call_id": "tc-2",
                    "tool_name": "bash",
                    "input": {"command": "rm -rf /"},
                },
            },
            {"type": "command.completed", "result": {"ok": True}},
        ])

        async def can_use_tool(tool_name, tool_input, ctx):
            return PermissionResult.deny("Dangerous command")

        client = make_client(
            transport=transport,
            workspace="/tmp",
            can_use_tool=can_use_tool,
        )
        client._initialized = True

        events = []
        async for event in client.receive_events():
            events.append(event)

        sent = transport.sent_messages
        assert sent[0]["response"]["behavior"] == "deny"
        assert sent[0]["response"]["message"] == "Dangerous command"

    @pytest.mark.anyio
    async def test_no_callback_denies_by_default(self):
        transport = MockTransport(messages=[
            {
                "type": WIRE_FRAME_TYPES["CONTROL_REQUEST"],
                "request_id": "approval_3",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_call_id": "tc-3",
                    "tool_name": "bash",
                    "input": {},
                },
            },
            {"type": "command.completed", "result": {"ok": True}},
        ])

        # No can_use_tool provided
        client = make_client(transport=transport, workspace="/tmp")
        client._initialized = True

        async for _ in client.receive_events():
            pass

        sent = transport.sent_messages
        assert sent[0]["response"]["behavior"] == "deny"
        assert "No can_use_tool callback" in sent[0]["response"]["message"]
