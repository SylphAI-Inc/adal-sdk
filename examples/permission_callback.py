"""Tool permission callback example.

Shows how SDK users provide can_use_tool to approve, deny, or modify tool input.

Run:
    python examples/permission_callback.py /path/to/workspace
"""

from __future__ import annotations

import os
import sys
from typing import Any

import anyio

from adal_agent_sdk import (
    AdalAgentClient,
    AdalAgentOptions,
    PermissionResult,
    ToolPermissionContext,
)


async def can_use_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    ctx: ToolPermissionContext,
):
    print(f"\n[permission] tool={tool_name} call_id={ctx.tool_call_id} reason={ctx.reason}")
    print(f"[permission] input={tool_input}")

    if tool_name.lower() == "bash":
        command = str(tool_input.get("command", ""))
        if "rm -rf" in command:
            return PermissionResult.deny("Blocked dangerous shell command")

        # Example of editing tool input before allowing it.
        if command.strip() == "pytest":
            return PermissionResult.allow(
                updated_input={**tool_input, "command": "pytest -q"}
            )

    return PermissionResult.allow()


async def main() -> None:
    workspace = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()

    options = AdalAgentOptions(
        workspace=workspace,
        permission_mode="default",
        can_use_tool=can_use_tool,
    )

    async with AdalAgentClient(options) as client:
        await client.query("Inspect the repository and run the smallest useful test.")

        async for event in client.receive_events():
            event_type = event.get("type")
            if event_type == "assistant.delta":
                print(event.get("text", ""), end="", flush=True)
            elif event_type == "tool.started":
                print(f"\n[tool started] {event.get('name')}")
            elif event_type in ("command.completed", "command.failed"):
                print(f"\n[{event_type}]")
                break


if __name__ == "__main__":
    anyio.run(main)
