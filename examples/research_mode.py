"""Run AdaL's managed research agent and render its existing event stream."""

from __future__ import annotations

import anyio

from adal_agent_sdk import AdalAgentClient, AdalAgentOptions


async def main() -> None:
    options = AdalAgentOptions(
        workspace=".",
        agent_mode="research",
        permission_mode="yolo",
    )

    async with AdalAgentClient(options) as client:
        print(f"Active agent mode: {client.active_agent_mode}")
        await client.query(
            "Research durable memory architectures for coding agents and "
            "summarize the tradeoffs."
        )

        async for event in client.receive_events(timeout=None):
            event_type = event.get("type")
            if event_type == "assistant.delta":
                print(event.get("text", ""), end="", flush=True)
            elif event_type == "tool.started":
                print(f"\n[tool started] {event.get('name', 'unknown')}")
            elif event_type == "command.failed":
                print(f"\n[failed] {event.get('error', {})}")


if __name__ == "__main__":
    anyio.run(main)
