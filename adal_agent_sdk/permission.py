"""Permission callback types for tool approval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Union


PERMISSION_REASON_PROMPT = "prompt"
PERMISSION_REASON_POLICY = "policy"


@dataclass
class ToolPermissionContext:
    """Context passed to the ``can_use_tool`` callback."""

    tool_call_id: str
    """Unique identifier for this tool call."""

    confirmation: dict[str, Any] | None = None
    """Confirmation metadata for edit tools: ``fileName``, ``fileDiff``,
    ``title``. Keys are camelCase. Empty for tools without a preview."""

    display: dict[str, Any] | None = None
    """Optional display metadata."""

    reason: str = PERMISSION_REASON_PROMPT
    """Why the callback is asked. ``"prompt"``: a person would be asked about
    this call in an interactive session. ``"policy"``: nobody would be asked
    (read-only tool, session grant, or yolo); the callback is the only check."""

    @property
    def requires_human(self) -> bool:
        """True when an interactive session would show a confirmation dialog."""
        return self.reason == PERMISSION_REASON_PROMPT


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
