"""Exception hierarchy for the AdaL Agent SDK."""

from __future__ import annotations


class SdkError(Exception):
    """Base error for all SDK exceptions."""


class AdalConnectionError(SdkError):
    """Failed to spawn or communicate with the SDK runtime subprocess."""


class ProtocolError(SdkError):
    """SDK protocol violation — unexpected message type or malformed frame."""


class QueryError(SdkError):
    """A query or runtime operation failed without invalidating the session."""

    def __init__(self, message: str, code: str | None = None):
        self.code = code
        super().__init__(message)


class SdkRuntimeError(SdkError):
    """The SDK runtime itself reported an error frame."""

    def __init__(self, message: str, code: str | None = None):
        self.code = code
        super().__init__(message)


class RuntimeUnresponsive(AdalConnectionError):
    """The runtime missed an SDK health watchdog and the session was closed."""

    code = "RUNTIME_UNRESPONSIVE"

    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(
            f"AdaL runtime did not respond to {operation!r} "
            f"within {timeout:g} seconds. The session was closed because "
            "protocol synchronization can no longer be guaranteed. "
            "Create a new AdalAgentClient and retry."
        )


class RuntimeUpgradeRequired(SdkError):
    """The installed AdaL runtime does not support a requested SDK capability."""

    code = "RUNTIME_UPGRADE_REQUIRED"

    def __init__(self, capability: str):
        self.capability = capability
        super().__init__(
            f"Installed AdaL runtime does not support {capability!r}. "
            "Update AdaL and try again."
        )
