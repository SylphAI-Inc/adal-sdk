"""Multi-query and resume example.

Run:
    python examples/multi_query_resume.py /path/to/workspace
"""

from __future__ import annotations

import os
import sys

import anyio

from adal_agent_sdk import AdalAgentClient, AdalAgentOptions


async def run_query(client: AdalAgentClient, text: str) -> None:
    await client.query(text)
    async for event in client.receive_events():
        event_type = event.get("type")
        if event_type == "assistant.delta":
            print(event.get("text", ""), end="", flush=True)
        elif event_type in ("command.completed", "command.failed"):
            print(f"\n[{event_type}]")
            return


async def main() -> None:
    workspace = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    async with AdalAgentClient(
        AdalAgentOptions(workspace=workspace, permission_mode="yolo")
    ) as first_client:
        print(f"First session: {first_client.session_id}")
        await run_query(first_client, "Remember this phrase: pink runtime bridge.")
        session_id = first_client.session_id

    if not session_id:
        raise RuntimeError("No session_id returned by SDK runtime")

    async with AdalAgentClient(
        AdalAgentOptions(
            workspace=workspace,
            session_id=session_id,
            permission_mode="yolo",
        )
    ) as resumed_client:
        print(f"Resumed session: {resumed_client.session_id}")
        await run_query(resumed_client, "What phrase did I ask you to remember?")


if __name__ == "__main__":
    anyio.run(main)
