"""Convenience API for one-shot AdaL Agent SDK queries."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from .client import AdalAgentClient
from .options import AdalAgentOptions


async def query(
    prompt: str,
    options: AdalAgentOptions | None = None,
    **kwargs: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Run a one-shot AdaL query and stream SDK events.

    This is the simplest entry point for scripts and batch jobs. For multi-query
    conversations, cancellation, or direct session lifecycle control, use
    :class:`AdalAgentClient`.

    Args:
        prompt: User prompt to send to AdaL.
        options: Optional pre-built SDK options.
        **kwargs: Convenience options used to construct ``AdalAgentOptions``
            when ``options`` is not provided, for example ``workspace="."`` or
            ``permission_mode="yolo"``.

    Yields:
        Raw SDK event dictionaries from ``AdalAgentClient.receive_events()``.

    Raises:
        ValueError: If both ``options`` and keyword options are provided.
    """
    if options is not None and kwargs:
        raise ValueError("Pass either options or keyword options, not both")

    resolved_options = options if options is not None else AdalAgentOptions(**kwargs)

    async with AdalAgentClient(resolved_options) as client:
        await client.query(prompt)
        async for event in client.receive_events():
            yield event
