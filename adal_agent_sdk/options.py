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
        permission_mode: 'default', 'acceptEdits', or 'yolo'.
        auth_token: JWT for platform auth. Omit for cached auth.
        allowed_tools: Tool groups to auto-approve (e.g. ["Bash", "Edit"]).
        thinking_effort: 'low', 'medium', 'high', or 'max'.
        prompt_file: Path to a custom role-prompt file (overrides the default
            system/role prompt at startup; matches headless --prompt-file).
        can_use_tool: Async callback for tool permission decisions.
        runtime_path: Path to the adal CLI binary. Defaults to PATH lookup.
    """

    workspace: str | Path | None = None
    model: str | None = None
    session_id: str | None = None
    permission_mode: str | None = None
    auth_token: str | None = None
    allowed_tools: list[str] | None = None
    thinking_effort: str | None = None
    prompt_file: str | Path | None = None
    can_use_tool: CanUseTool | None = None
    runtime_path: str | Path | None = None
