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
        options=AdalAgentOptions(enabled_default_tools=["Read", "Search", "Bash"]),
    ):
        print(event)

asyncio.run(main())
```

More examples are in [`examples/`](examples/).

## Permission callbacks

A `can_use_tool` callback is the policy check for **every** tool call, read-only tools included. It can allow the call, deny it with a reason the model sees (the turn ends), or change its arguments.

```python
from adal_agent_sdk import PermissionResult, ToolPermissionContext

async def can_use_tool(tool_name: str, tool_input: dict, ctx: ToolPermissionContext):
    if not ctx.requires_human:
        return PermissionResult.allow()  # a read-only call nobody would be asked about
    if tool_name == "bash" and "rm -rf" in tool_input.get("command", ""):
        return PermissionResult.deny("Dangerous command")
    return PermissionResult.allow()

options = AdalAgentOptions(workspace=".", can_use_tool=can_use_tool)
```

`ctx.reason` is `"prompt"` when an interactive session would show a confirmation dialog for the call and `"policy"` when nobody would be asked. A callback that takes longer than `can_use_tool_timeout` (30 s by default) denies the call. Full reference: https://docs.sylph.ai/sdk/permissions

## Custom System prompt & Custom tools
https://docs.sylph.ai/features/custom-system-prompt  
https://docs.sylph.ai/features/custom-tools

## SDK Documentation
https://docs.sylph.ai/sdk/quickstart

## License

MIT License. See [`LICENSE`](LICENSE).
