"""Subprocess transport — spawns ``adal --sdk-runtime`` and handles NDJSON I/O."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import anyio

from .exceptions import AdalConnectionError, ProtocolError


class SubprocessTransport:
    """Low-level transport: spawns the TS SDK runtime and pipes NDJSON.

    The transport owns the subprocess lifecycle. It provides:
    - :meth:`send` — write one JSON line to stdin
    - :meth:`receive` — read one JSON line from stdout (blocking)
    - :meth:`close` — graceful shutdown (close stdin → wait → SIGTERM → SIGKILL)

    The transport does NOT interpret message types — that's the client's job.
    """

    def __init__(
        self,
        runtime_path: str | Path | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        stderr=None,
        auth_token: str | None = None,
    ):
        self._runtime_path = str(runtime_path) if runtime_path else None
        self._cwd = str(cwd) if cwd else None
        self._env = env
        self._auth_token = auth_token
        self._stderr = stderr if stderr is not None else sys.stderr
        self._process: anyio.abc.Process | None = None
        self._stdout_buffer = bytearray()

    @property
    def is_alive(self) -> bool:
        """True if the subprocess is still running."""
        return self._process is not None and self._process.returncode is None

    def _resolve_runtime_path(self) -> str:
        """Find the adal CLI binary."""
        # Explicit path — could be a bare command name or a file path
        if self._runtime_path:
            # If it looks like a path (contains /), check the file exists
            if "/" in self._runtime_path or "\\" in self._runtime_path:
                if not os.path.isfile(self._runtime_path):
                    raise AdalConnectionError(f"Runtime not found at: {self._runtime_path}")
                return self._runtime_path
            # Bare command name — do PATH lookup
            found = shutil.which(self._runtime_path)
            if found:
                return found
            raise AdalConnectionError(f"Runtime '{self._runtime_path}' not found in PATH")

        # ADAL_RUNTIME_PATH env var
        env_path = os.environ.get("ADAL_RUNTIME_PATH")
        if env_path:
            if os.path.isfile(env_path):
                return env_path
            found = shutil.which(env_path)
            if found:
                return found

        # Default PATH lookup — prefer adal-dev (has latest features like --sdk-runtime)
        found = shutil.which("adal-dev") or shutil.which("adal")
        if found:
            return found

        raise AdalConnectionError(
            "AdaL CLI not found. Install it with the native installer:\n"
            "  macOS/Linux/WSL: curl -fsSL https://adal.sylph.ai/install.sh | bash\n"
            "  Windows PowerShell: irm https://adal.sylph.ai/install/windows | iex\n"
            "Then run `adal` once to authenticate, or set ADAL_RUNTIME_PATH to the binary location."
        )

    def _runtime_supports_sdk_runtime(self, binary: str) -> bool:
        """Return whether the AdaL CLI advertises SDK runtime support."""
        try:
            result = subprocess.run(
                [binary, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=5,
                encoding="utf-8",
                errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            # Let process startup surface the real error if help probing fails.
            return True

        help_text = f"{result.stdout}\n{result.stderr}"
        return "--sdk-runtime" in help_text

    def _raise_unsupported_runtime(self, binary: str) -> None:
        raise AdalConnectionError(
            f"AdaL CLI at '{binary}' does not support --sdk-runtime.\n"
            "Update AdaL with the native installer:\n"
            "  macOS/Linux/WSL: curl -fsSL https://adal.sylph.ai/install.sh | bash\n"
            "  Windows PowerShell: irm https://adal.sylph.ai/install/windows | iex\n"
            "Or set ADAL_RUNTIME_PATH to a compatible AdaL CLI binary."
        )

    async def start(self) -> None:
        """Spawn the SDK runtime subprocess."""
        binary = self._resolve_runtime_path()
        if not self._runtime_supports_sdk_runtime(binary):
            self._raise_unsupported_runtime(binary)

        spawn_env = os.environ.copy()
        if self._env:
            spawn_env.update(self._env)

        spawn_args = [binary, "--sdk-runtime"]
        if self._auth_token:
            spawn_args += ["--token", self._auth_token]

        try:
            self._process = await anyio.open_process(
                spawn_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                cwd=self._cwd,
                env=spawn_env,
            )
        except FileNotFoundError as e:
            raise AdalConnectionError(f"Failed to spawn runtime: {e}") from e
        except OSError as e:
            raise AdalConnectionError(f"Failed to spawn runtime: {e}") from e

    async def send(self, message: dict[str, Any]) -> None:
        """Write one JSON line to the subprocess stdin."""
        if not self._process or not self._process.stdin:
            raise AdalConnectionError("Transport not started or already closed")

        line = json.dumps(message) + "\n"
        try:
            await self._process.stdin.send(line.encode("utf-8"))
        except (BrokenPipeError, anyio.BrokenResourceError) as e:
            raise AdalConnectionError(f"Failed to write to runtime stdin: {e}") from e

    async def receive(self, timeout: float | None = None) -> dict[str, Any]:
        """Read and parse one JSON line from the subprocess stdout.

        Uses a manual line buffer because anyio's subprocess stdout wrapper
        does not support ``receive_until``.

        Args:
            timeout: Maximum seconds to wait for a complete line. None = no timeout.
        """
        if not self._process or not self._process.stdout:
            raise AdalConnectionError("Transport not started or already closed")

        # Loop to skip empty lines without recursion (avoids stack overflow)
        while True:
            # Check if we already have a complete line in the buffer
            newline_idx = self._stdout_buffer.find(b"\n")
            if newline_idx >= 0:
                line = bytes(self._stdout_buffer[:newline_idx])
                del self._stdout_buffer[:newline_idx + 1]
            else:
                # Need more data from stdout
                try:
                    if timeout is not None:
                        with anyio.fail_after(timeout):
                            chunk = await self._process.stdout.receive(65536)
                    else:
                        chunk = await self._process.stdout.receive(65536)
                except anyio.EndOfStream:
                    # Process closed stdout — flush any remaining buffered data
                    if self._stdout_buffer:
                        line = bytes(self._stdout_buffer)
                        self._stdout_buffer.clear()
                    else:
                        raise AdalConnectionError("Runtime stdout closed unexpectedly")
                except TimeoutError as e:
                    raise AdalConnectionError(f"Timed out waiting for runtime response after {timeout}s") from e
                except Exception as e:
                    raise AdalConnectionError(f"Failed to read from runtime stdout: {e}") from e

                self._stdout_buffer.extend(chunk)
                continue  # Re-check buffer for a complete line

            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                # Skip empty lines — continue the loop instead of recursing
                continue

            try:
                return json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ProtocolError(f"Invalid JSON from runtime: {line_str[:200]}") from e

    async def close(self) -> None:
        """Gracefully shut down the subprocess.

        1. Close stdin (signals the runtime to exit its main loop)
        2. Wait up to 5s for clean exit
        3. SIGTERM if still running
        4. SIGKILL if still running after 3s
        """
        if not self._process:
            return

        # Close stdin
        if self._process.stdin:
            await self._process.stdin.aclose()

        # Wait for clean exit (5s)
        try:
            with anyio.fail_after(5):
                await self._process.wait()
        except TimeoutError:
            pass

        # If still running, SIGTERM
        if self._process.returncode is None:
            self._process.terminate()
            try:
                with anyio.fail_after(3):
                    await self._process.wait()
            except TimeoutError:
                pass

        # If STILL running, SIGKILL
        if self._process.returncode is None:
            self._process.kill()
            await self._process.wait()

        self._process = None
