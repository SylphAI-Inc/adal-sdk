"""Exception hierarchy for the AdaL Agent SDK."""

from __future__ import annotations


class SdkError(Exception):
    """Base error for all SDK exceptions."""


class AdalConnectionError(SdkError):
    """Failed to spawn or communicate with the SDK runtime subprocess."""


class ProtocolError(SdkError):
    """SDK protocol violation — unexpected message type or malformed frame."""


class QueryError(SdkError):
    """A query failed during execution (non-fatal — session continues)."""


class SdkRuntimeError(SdkError):
    """The SDK runtime itself reported an error frame."""
