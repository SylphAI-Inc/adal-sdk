"""Configuration options for the AdaL Agent SDK client."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .permission import CanUseTool


@dataclass
class AdalAgentOptions:
    """Options for configuring an :class:`AdalAgentClient` session.

    Attributes:
        workspace: Absolute path to the workspace root. Defaults to cwd.
        model: Model ID to use at startup.
        session_id: Existing session ID to resume, or None for new session.
        permission_mode: When a person would be asked to confirm a tool call:
            'default' (writes and shell commands), 'acceptEdits' (shell commands
            only; file edits run), or 'yolo' (never). Unknown values fail the
            session start. With ``can_use_tool`` set, the callback is consulted
            for every tool call regardless of mode; the mode only sets
            ``ctx.requires_human``.
        auth_token: JWT for platform auth. Omit for cached auth.

        enabled_default_tools: Positive set of core tools/groups to enable
            (e.g. ["Read", "Search", "Bash"]). ONLY these tools will be
            available. Cannot combine with disabled_default_tools.
        disabled_default_tools: Core tools/groups to disable entirely (e.g.
            ["Web", "Video"]). Disabled tools are invisible to the agent and
            unexecutable. Scope: core tools only.
        thinking_effort: 'low', 'medium', 'high', or 'max'.
        prompt_file: Path to a custom role-prompt file (overrides the default
            system/role prompt at startup; matches headless --prompt-file).
        can_use_tool: Async callback consulted for EVERY tool call the agent
            makes (read-only tools included). Return ``PermissionResult.allow()``,
            ``allow(updated_input=...)`` to change the arguments, or
            ``deny(message)``; the message is shown to the model and the turn
            ends. Without a callback, no tool asks the client.
        can_use_tool_timeout: Seconds the runtime waits for ``can_use_tool``
            before denying the call and ending the turn. Default 30.
        runtime_path: Path to the adal CLI binary. Defaults to PATH lookup.
    """

    workspace: str | Path | None = None
    model: str | None = None
    session_id: str | None = None
    permission_mode: str | None = None
    auth_token: str | None = None

    enabled_default_tools: list[str] | None = None
    disabled_default_tools: list[str] | None = None
    thinking_effort: str | None = None
    prompt_file: str | Path | None = None
    can_use_tool: CanUseTool | None = None
    can_use_tool_timeout: float = 30.0
    runtime_path: str | Path | None = None
