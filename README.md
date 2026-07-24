# AdaL Agent SDK for Python

Python SDK for embedding AdaL's agent runtime in your applications.

The SDK provides a Pythonic async client over AdaL's SDK runtime protocol. It keeps a persistent AdaL subprocess open so you can run multiple queries, resume sessions, and handle tool permission callbacks from Python.

## Requirements

- Python 3.10+
- AdaL CLI installed and available as `adal` or `adal-dev`

Install the AdaL CLI runtime with the native installer:

```bash
# macOS, Linux, WSL
curl -fsSL https://adal.sylph.ai/install.sh | bash

# Windows PowerShell
irm https://adal.sylph.ai/install/windows | iex
```

Run `adal` once to complete browser authentication before using the SDK. You can also point the SDK at a custom runtime with `ADAL_RUNTIME_PATH` or `AdalAgentOptions(runtime_path=...)`.

> Note: the SDK requires an AdaL CLI version that supports `--sdk-runtime`. If `adal --help` does not show `--sdk-runtime`, run the installer above to update AdaL.

## Install

From this repository:

```bash
pip install git+https://github.com/SylphAI-Inc/adal-sdk.git
```

For local development:

```bash
git clone https://github.com/SylphAI-Inc/adal-sdk.git
cd adal-sdk
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```python
import asyncio
from adal_agent_sdk import AdalAgentOptions, query

async def main():
    async for event in query(
        prompt="Map this workspace: identify the main entry points, test command, and one improvement opportunity.",
        options=AdalAgentOptions(enabled_default_tools=["Read", "Bash"]),
    ):
        print(event)

asyncio.run(main())
```

More examples are in [`examples/`](examples/).

## Agent modes

Select any main agent mode supported by the installed AdaL runtime. Mode names
are validated by AdaL rather than duplicated in the Python package.

```python
from adal_agent_sdk import AdalAgentClient, AdalAgentOptions

options = AdalAgentOptions(
    workspace=".",
    agent_mode="research",
)

async with AdalAgentClient(options) as client:
    print(client.active_agent_mode)  # "research"

    await client.query("Research durable memory architectures for coding agents")
    async for event in client.receive_events(timeout=None):
        if event["type"] == "assistant.delta":
            print(event.get("text", ""), end="")

    await client.set_agent_mode("coding")
```

Agent-mode operation deadlines are owned by the installed AdaL runtime. A
runtime switch timeout raises `QueryError` with
`code == "AGENT_MODE_SWITCH_TIMEOUT"` and leaves the session usable:

```python
from adal_agent_sdk import QueryError

try:
    await client.set_agent_mode("research")
except QueryError as error:
    if error.code == "AGENT_MODE_SWITCH_TIMEOUT":
        print("AdaL could not finish the switch before its runtime deadline")
```

Separately, the SDK has a generic control-channel health watchdog. If the
runtime never sends any correlated response, the SDK raises
`RuntimeUnresponsive` with `code == "RUNTIME_UNRESPONSIVE"` and closes the
session. Create a new client after this fatal error; continuing on the same
wire stream could consume a late response out of order.

The timing and recovery contract is public:

| Condition | Deadline | Exception/code | Session state |
|---|---:|---|---|
| AdaL cannot complete an agent-mode switch | 60 seconds, owned by the runtime | `QueryError` / `AGENT_MODE_SWITCH_TIMEOUT` | Usable; retry or select another mode |
| No correlated control response arrives | 120 seconds, SDK health watchdog | `RuntimeUnresponsive` / `RUNTIME_UNRESPONSIVE` | Closed; create a new client |
| Runtime initialization never reaches `ready` | 180 seconds, SDK startup watchdog | `RuntimeUnresponsive` / `RUNTIME_UNRESPONSIVE` | Closed; create a new client |

`RuntimeUnresponsive` also exposes the timed-out `operation` and `timeout`
attributes for logs, telemetry, and retry policy.

The SDK uses AdaL's existing UI-neutral event stream for every mode, including
assistant, reasoning, tool, subagent, completion, and failure events. It does
not define a separate research-result format.

The SDK is a client for the installed AdaL runtime; it does not run the agent
independently. Official packaged runtimes provide AdaL's managed web tools.
Source/custom runtime builds remain responsible for their own tool
configuration, and provider failures are reported through the normal SDK event
stream.

## Permission callbacks

```python
from adal_agent_sdk import PermissionResult, ToolPermissionContext

async def can_use_tool(tool_name: str, tool_input: dict, ctx: ToolPermissionContext):
    if tool_name == "bash" and "rm -rf" in tool_input.get("command", ""):
        return PermissionResult.deny("Dangerous command")
    return PermissionResult.allow()

options = AdalAgentOptions(workspace=".", can_use_tool=can_use_tool)
```

## Custom System prompt & Custom tools
https://docs.sylph.ai/features/custom-system-prompt  
https://docs.sylph.ai/features/custom-tools

## SDK Documentation
https://docs.sylph.ai/sdk/quickstart

## License

MIT License. See [`LICENSE`](LICENSE).
