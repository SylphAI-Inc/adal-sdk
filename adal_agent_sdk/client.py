"""AdalAgentClient — persistent SDK client for multi-query agent sessions."""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import anyio

logger = logging.getLogger(__name__)

from .exceptions import (
    AdalConnectionError,
    ProtocolError,
    QueryError,
    SdkRuntimeError,
    SdkError,
)
from .options import AdalAgentOptions
from .permission import PermissionResultAllow, PermissionResultDeny, ToolPermissionContext
from .transport import SubprocessTransport

# SDK wire protocol constants — must stay in sync with the AdaL CLI SDK runtime.
WIRE_PROTOCOL_VERSION = 1
WIRE_COMMAND_TYPES = {
    "INITIALIZE": "initialize",
    "QUERY": "query",
    "CANCEL": "cancel",
    "SHUTDOWN": "shutdown",
    "CONTROL_REQUEST": "control_request",
    "CONTROL_RESPONSE": "control_response",
}
WIRE_FRAME_TYPES = {
    "READY": "ready",
    "CONTROL_REQUEST": "control_request",
    "CONTROL_RESPONSE": "control_response",
    "ERROR": "error",
}

# Default timeout for receiving events during a query (5 minutes).
# Long enough for real agent work (tool calls, LLM responses) but
# prevents infinite hangs if the server process is stuck.
DEFAULT_RECEIVE_TIMEOUT = 300.0


class AdalAgentClient:
    """Persistent client for multi-query agent sessions.

    Spawns the AdaL SDK runtime as a subprocess and communicates over
    stdio NDJSON pipes. The connection stays open for the entire session —
    multiple queries, model switches, and approval callbacks all flow over
    the same pipes.

    Usage::

        options = AdalAgentOptions(workspace="/my/project")
        async with AdalAgentClient(options) as client:
            await client.query("Fix the bug")
            async for event in client.receive_events():
                print(event)

    For tool approval callbacks::

        async def can_use_tool(tool_name, input, ctx):
            if "rm -rf" in input.get("command", ""):
                return PermissionResult.deny("Dangerous")
            return PermissionResult.allow()

        options = AdalAgentOptions(can_use_tool=can_use_tool)
    """

    def __init__(self, options: AdalAgentOptions):
        self._options = options
        self._transport = SubprocessTransport(
            runtime_path=options.runtime_path,
            cwd=str(options.workspace) if options.workspace else None,
        )
        self._session_id: str | None = None
        self._closed = False
        self._initialized = False
        # Pending control responses from our outbound control_requests
        self._pending_control_responses: dict[str, anyio.Event] = {}
        self._pending_control_results: dict[str, dict[str, Any] | Exception] = {}
        self._request_counter = 0
        # Track whether receive_events() is actively reading
        self._receive_events_active = False

    @property
    def session_id(self) -> str | None:
        """The active session ID (available after initialization)."""
        return self._session_id

    async def __aenter__(self) -> AdalAgentClient:
        await self._initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def _initialize(self) -> None:
        """Spawn runtime, send initialize command, wait for ready frame."""
        _t0 = time.monotonic()
        await self._transport.start()
        _spawn_ms = (time.monotonic() - _t0) * 1000

        # Send initialize command
        init_cmd: dict[str, Any] = {
            "type": WIRE_COMMAND_TYPES["INITIALIZE"],
            "protocol_version": WIRE_PROTOCOL_VERSION,
        }
        if self._options.workspace:
            init_cmd["workspace"] = str(self._options.workspace)
        if self._options.model:
            init_cmd["model"] = self._options.model
        if self._options.session_id:
            init_cmd["session_id"] = self._options.session_id
        if self._options.permission_mode:
            init_cmd["permission_mode"] = self._options.permission_mode
        if self._options.auth_token:
            init_cmd["auth_token"] = self._options.auth_token
        if self._options.allowed_tools:
            init_cmd["allowed_tools"] = self._options.allowed_tools
        if self._options.thinking_effort:
            init_cmd["thinking_effort"] = self._options.thinking_effort
        if self._options.prompt_file:
            init_cmd["prompt_file"] = str(self._options.prompt_file)

        await self._transport.send(init_cmd)

        # Wait for ready frame (with timeout)
        _t_ready = time.monotonic()
        with anyio.fail_after(60):
            msg = await self._transport.receive()
        _ready_ms = (time.monotonic() - _t_ready) * 1000

        msg_type = msg.get("type")
        if msg_type == WIRE_FRAME_TYPES["READY"]:
            self._session_id = msg.get("session_id")
            self._initialized = True
            logger.debug(
                "[sdk-init] spawn=%.0fms backend_boot_to_ready=%.0fms total=%.0fms",
                _spawn_ms, _ready_ms, (time.monotonic() - _t0) * 1000,
            )
        elif msg_type == WIRE_FRAME_TYPES["ERROR"]:
            raise SdkRuntimeError(msg.get("message", "Initialization failed"))
        else:
            raise ProtocolError(f"Expected ready frame, got: {msg_type}")

    async def query(self, input: str, **kwargs: Any) -> None:
        """Send a query to the agent. Events are received via :meth:`receive_events`.

        Args:
            input: The user input text.
            **kwargs: Optional keyword arguments:
                images: List of base64-encoded image paths or data URIs.
                display_text: Display text override.
                context_files: Additional context file paths.
        """
        if not self._initialized:
            raise SdkError("Client not initialized. Use 'async with AdalAgentClient(...)'")

        cmd: dict[str, Any] = {
            "type": WIRE_COMMAND_TYPES["QUERY"],
            "input": input,
        }
        if "images" in kwargs:
            cmd["images"] = kwargs["images"]
        if "display_text" in kwargs:
            cmd["display_text"] = kwargs["display_text"]
        if "context_files" in kwargs:
            cmd["context_files"] = kwargs["context_files"]

        await self._transport.send(cmd)

    async def receive_events(
        self, timeout: float | None = DEFAULT_RECEIVE_TIMEOUT
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield events from the active query until ``command.completed`` or ``command.failed``.

        This method handles control frames transparently:
        - ``control_request`` from server (approval callback) → invokes
          ``can_use_tool`` and sends ``control_response`` back
        - ``control_response`` from server (setup op result) → resolves
          pending control request promise
        - ``error`` → raises :class:`SdkRuntimeError`
        - Regular events (assistant.delta, tool.started, etc.) → yielded to caller

        The iterator stops after ``command.completed`` or ``command.failed``.

        Args:
            timeout: Maximum seconds to wait for each message. None = no timeout.
                Defaults to 300s (5 minutes) to prevent infinite hangs if the
                server process is stuck. Set to None for long-running tasks
                that may take hours.
        """
        self._receive_events_active = True
        try:
            while True:
                msg = await self._transport.receive(timeout=timeout)
                msg_type = msg.get("type")

                # ── Control frames (not yielded to consumer) ────────────────────
                if msg_type == WIRE_FRAME_TYPES["CONTROL_REQUEST"]:
                    await self._handle_server_control_request(msg)
                    continue

                if msg_type == WIRE_FRAME_TYPES["CONTROL_RESPONSE"]:
                    self._resolve_control_response(msg)
                    continue

                if msg_type == WIRE_FRAME_TYPES["ERROR"]:
                    raise SdkRuntimeError(
                        msg.get("message", "Unknown error"),
                    )

                # ── Regular events (yielded to consumer) ────────────────────────
                # Query end signals
                if msg_type in ("command.completed", "command.failed"):
                    yield msg
                    return

                yield msg
        finally:
            self._receive_events_active = False

    async def set_model(self, model: str) -> dict[str, Any]:
        """Switch the active model. Returns the server's response."""
        return await self._send_control_request({"subtype": "set_model", "model": model})

    async def set_permission_mode(self, mode: str) -> dict[str, Any]:
        """Set the permission mode ('default', 'acceptEdits', 'yolo')."""
        return await self._send_control_request(
            {"subtype": "set_permission_mode", "mode": mode}
        )

    async def cancel(self) -> None:
        """Cancel the active query."""
        await self._transport.send({"type": WIRE_COMMAND_TYPES["CANCEL"]})

    async def close(self) -> None:
        """Shut down the SDK runtime and close the connection."""
        if self._closed:
            return
        self._closed = True

        # Try graceful shutdown
        try:
            await self._transport.send({"type": WIRE_COMMAND_TYPES["SHUTDOWN"]})
        except (AdalConnectionError, anyio.BrokenResourceError):
            pass  # Already closed

        await self._transport.close()

    # ── Internal: control protocol ──────────────────────────────────────────

    async def _send_control_request(
        self, request: dict[str, Any], timeout: float = 30.0
    ) -> dict[str, Any]:
        """Send a control_request to the server and await the control_response.

        When ``receive_events()`` is NOT active, this method directly reads
        from the transport until the matching control_response arrives.
        When ``receive_events()`` IS active, it registers an event and lets
        the receive loop resolve it.
        """
        self._request_counter += 1
        request_id = f"req_{self._request_counter}_{uuid.uuid4().hex[:8]}"

        cmd = {
            "type": WIRE_COMMAND_TYPES["CONTROL_REQUEST"],
            "request_id": request_id,
            "request": request,
        }
        await self._transport.send(cmd)

        if self._receive_events_active:
            # receive_events() is reading — register an event and let it resolve
            event = anyio.Event()
            self._pending_control_responses[request_id] = event

            try:
                with anyio.fail_after(timeout):
                    await event.wait()
            except TimeoutError as e:
                self._pending_control_responses.pop(request_id, None)
                self._pending_control_results.pop(request_id, None)
                raise QueryError(f"Control request timed out: {request.get('subtype')}") from e

            result = self._pending_control_results.pop(request_id, None)
            self._pending_control_responses.pop(request_id, None)

            if isinstance(result, Exception):
                raise result

            if not result:
                return {}

            if result.get("subtype") == "error":
                raise QueryError(result.get("error", "Control request failed"))

            return result.get("response", {})
        else:
            # No active receive_events() loop — read directly from transport
            # until we get the matching control_response
            while True:
                msg = await self._transport.receive(timeout=timeout)

                msg_type = msg.get("type")

                # Handle server-initiated control_request (approval) that
                # might arrive while we're waiting
                if msg_type == WIRE_FRAME_TYPES["CONTROL_REQUEST"]:
                    await self._handle_server_control_request(msg)
                    continue

                if msg_type == WIRE_FRAME_TYPES["ERROR"]:
                    raise SdkRuntimeError(msg.get("message", "Unknown error"))

                if msg_type == WIRE_FRAME_TYPES["CONTROL_RESPONSE"]:
                    if msg.get("request_id") != request_id:
                        # Response for a different request — shouldn't happen
                        # but skip it
                        continue

                    if msg.get("subtype") == "error":
                        raise QueryError(msg.get("error", "Control request failed"))

                    return msg.get("response", {})

                # Regular events or other frames while waiting — skip
                # (shouldn't happen between queries, but be safe)
                continue

    def _resolve_control_response(self, msg: dict[str, Any]) -> None:
        """Resolve a pending control_request promise with the received response."""
        request_id = msg.get("request_id")
        if not request_id or request_id not in self._pending_control_responses:
            return

        event = self._pending_control_responses[request_id]

        if msg.get("subtype") == "error":
            self._pending_control_results[request_id] = QueryError(
                msg.get("error", "Unknown error")
            )
        else:
            self._pending_control_results[request_id] = msg

        event.set()

    async def _handle_server_control_request(self, msg: dict[str, Any]) -> None:
        """Handle a server-initiated control_request (e.g. tool approval)."""
        request_id = msg.get("request_id", "")
        request = msg.get("request", {})
        subtype = request.get("subtype", "")

        try:
            if subtype == "can_use_tool":
                response_payload = await self._handle_approval(request)
            else:
                raise ProtocolError(f"Unknown control request subtype: {subtype}")

            # Send control_response back to server
            await self._transport.send({
                "type": WIRE_COMMAND_TYPES["CONTROL_RESPONSE"],
                "request_id": request_id,
                "response": response_payload,
            })

        except Exception as e:
            # Send error response
            await self._transport.send({
                "type": WIRE_COMMAND_TYPES["CONTROL_RESPONSE"],
                "request_id": request_id,
                "response": {"behavior": "deny", "message": str(e)},
            })

    async def _handle_approval(self, request: dict[str, Any]) -> dict[str, Any]:
        """Invoke the user's can_use_tool callback and format the response."""
        if not self._options.can_use_tool:
            return {"behavior": "deny", "message": "No can_use_tool callback provided"}

        tool_name = request.get("tool_name", "")
        tool_input = request.get("input", {})
        tool_call_id = request.get("tool_call_id", "")

        ctx = ToolPermissionContext(
            tool_call_id=tool_call_id,
            confirmation=request.get("confirmation"),
            display=request.get("display"),
        )

        result = await self._options.can_use_tool(tool_name, tool_input, ctx)

        # Convert PermissionResult to wire format
        if isinstance(result, PermissionResultAllow):
            payload: dict[str, Any] = {"behavior": "allow"}
            if result.updated_input is not None:
                payload["updated_input"] = result.updated_input
            return payload
        elif isinstance(result, PermissionResultDeny):
            return {"behavior": "deny", "message": result.message}
        else:
            return {"behavior": "deny", "message": "Invalid permission result type"}
