"""Session control example.

Demonstrates startup options plus runtime control requests:
- set_model()
- set_permission_mode()
- thinking_effort at initialization

Run:
    ADAL_MODEL=openai-gpt-5.6-sol python examples/session_controls.py /path/to/workspace
"""

from __future__ import annotations

import os
import sys

import anyio

from adal_agent_sdk import AdalAgentClient, AdalAgentOptions


async def drain_until_done(client: AdalAgentClient) -> None:
    async for event in client.receive_events():
        event_type = event.get("type")
        if event_type == "assistant.delta":
            print(event.get("text", ""), end="", flush=True)
        elif event_type in ("command.completed", "command.failed"):
            print(f"\n[{event_type}]")
            return


async def main() -> None:
    workspace = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    model = os.environ.get("ADAL_MODEL")

    options = AdalAgentOptions(
        workspace=workspace,
        model=model,
        permission_mode="default",

        thinking_effort=os.environ.get("ADAL_THINKING_EFFORT", "low"),
    )

    async with AdalAgentClient(options) as client:
        print(f"Session: {client.session_id}")

        if model:
            response = await client.set_model(model)
            print(f"set_model response: {response}")

        response = await client.set_permission_mode("yolo")
        print(f"set_permission_mode response: {response}")

        await client.query("List the top-level files and explain what this project does.")
        await drain_until_done(client)


if __name__ == "__main__":
    anyio.run(main)
