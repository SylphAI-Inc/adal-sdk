"""AdaL Agent SDK — embed AdaL's agent runtime in your Python applications.

Quick start:
    import asyncio
    from adal_agent_sdk import AdalAgentOptions, query

    async def main():
        async for event in query(
            prompt="Map this workspace: identify the main entry points, test command, and one improvement opportunity.",
            options=AdalAgentOptions(permission_mode="yolo"),
        ):
            print(event)

    asyncio.run(main())
"""

from .client import AdalAgentClient
from .options import AdalAgentOptions
from .query import query
from .permission import PermissionResult, ToolPermissionContext
from .exceptions import (
    SdkError,
    AdalConnectionError,
    QueryError,
    ProtocolError,
    RuntimeUnresponsive,
    RuntimeUpgradeRequired,
    SdkRuntimeError,
)
from .transport import SubprocessTransport

__version__ = "0.1.0"

__all__ = [
    "query",
    "AdalAgentClient",
    "AdalAgentOptions",
    "PermissionResult",
    "ToolPermissionContext",
    "SdkError",
    "AdalConnectionError",
    "QueryError",
    "ProtocolError",
    "RuntimeUnresponsive",
    "RuntimeUpgradeRequired",
    "SdkRuntimeError",
    "SubprocessTransport",
    "__version__",
]
