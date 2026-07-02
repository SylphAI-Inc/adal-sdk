"""Basic AdaL Agent SDK query example.

Run:
    python examples/basic_query.py /path/to/workspace
"""

from __future__ import annotations

import asyncio
import os
import sys

from adal_agent_sdk import AdalAgentOptions, query


async def main() -> None:
    workspace = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    async for event in query(
        prompt="Summarize this workspace in three bullets.",
        options=AdalAgentOptions(
            workspace=workspace,
            # Use yolo for a minimal non-interactive demo. Use "default" when you
            # want can_use_tool callbacks or CLI approval behavior.
            permission_mode="yolo",
        ),
    ):
        print(event)


if __name__ == "__main__":
    asyncio.run(main())
