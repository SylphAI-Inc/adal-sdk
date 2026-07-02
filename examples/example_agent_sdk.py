import sys
import os
import anyio
from adal_agent_sdk import AdalAgentClient, AdalAgentOptions

async def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    print(f"Workspace: {workspace}")

    options = AdalAgentOptions(
        workspace=workspace,
        permission_mode="yolo",
    )

    async with AdalAgentClient(options) as client:
        print(f"Session: {client.session_id}")
        print("Type your queries. Press Ctrl+C to exit.\n")

        while True:
            try:
                query = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not query:
                continue

            # All input goes to the agent — no client-side command interception.
            # Use Ctrl+C to exit. Use client.set_model() / set_permission_mode()
            # in your own code for session control (matches Claude SDK pattern).

            await client.query(query)

            answer = ""
            async for event in client.receive_events():
                etype = event.get("type", "")
                if etype == "assistant.delta":
                    print(event.get("text", ""), end="", flush=True)
                elif etype == "assistant.message.completed":
                    msg = event.get("message", {})
                    if isinstance(msg, dict):
                        answer = msg.get("content", "")
                elif etype == "tool.started":
                    print(f"\n  [tool: {event.get('name', '?')}]")
                elif etype == "command.completed":
                    break
                elif etype == "command.failed":
                    error = event.get("error", {})
                    print(f"\n  [error: {error.get('message', 'unknown')}]")
                    break

            if answer and not answer.isspace():
                print(answer)
            print()

anyio.run(main)
