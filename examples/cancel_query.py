"""Cancel an active SDK query.

Run:
    python examples/cancel_query.py /path/to/workspace
"""

from __future__ import annotations

import os
import sys

import anyio

from adal_agent_sdk import AdalAgentClient, AdalAgentOptions


async def main() -> None:
    workspace = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    async with AdalAgentClient(
        AdalAgentOptions(workspace=workspace, permission_mode="yolo")
    ) as client:
        await client.query("Start a long investigation of this repository.")

        async def cancel_soon() -> None:
            await anyio.sleep(3)
            print("\n[cancel] sending cancel")
            await client.cancel()

        async with anyio.create_task_group() as tg:
            tg.start_soon(cancel_soon)
            async for event in client.receive_events(timeout=None):
                event_type = event.get("type")
                print(f"[event] {event_type}")
                if event_type in ("command.completed", "command.failed"):
                    tg.cancel_scope.cancel()
                    break


if __name__ == "__main__":
    anyio.run(main)
