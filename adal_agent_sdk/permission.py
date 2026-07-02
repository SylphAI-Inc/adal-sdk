"""Permission callback types for tool approval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Union


@dataclass
class ToolPermissionContext:
    """Context passed to the ``can_use_tool`` callback."""

    tool_call_id: str
    """Unique identifier for this tool call."""

    confirmation: dict[str, Any] | None = None
    """Optional confirmation metadata (diff, prompt text, etc.)."""

    display: dict[str, Any] | None = None
    """Optional display metadata."""


@dataclass
class PermissionResultAllow:
    """Allow a tool call to proceed."""

    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None
    """Optional updated tool input."""


@dataclass
class PermissionResultDeny:
    """Deny a tool call."""

    behavior: Literal["deny"] = "deny"
    message: str = ""
    """Denial reason message."""


# Type alias for the union of possible permission results
PermissionResultType = Union[PermissionResultAllow, PermissionResultDeny]

CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext],
    Awaitable[PermissionResultType],
]


class PermissionResult:
    """Factory for building permission result values in user callbacks.

    Usage::

        async def can_use_tool(tool_name, input, ctx):
            if tool_name == "bash" and "rm -rf" in input.get("command", ""):
                return PermissionResult.deny("Dangerous command")
            return PermissionResult.allow()
    """

    @staticmethod
    def allow(
        updated_input: dict[str, Any] | None = None,
    ) -> PermissionResultAllow:
        """Allow the tool call. Optionally provide updated input."""
        return PermissionResultAllow(updated_input=updated_input)

    @staticmethod
    def deny(message: str = "") -> PermissionResultDeny:
        """Deny the tool call with an optional reason."""
        return PermissionResultDeny(message=message)
