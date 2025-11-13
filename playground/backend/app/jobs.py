"""Asynchronous job orchestration for model inference."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, Dict, Optional


class ProgressStreamer:
    """Utility to stream progress updates to listeners."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def publish(self, payload: dict[str, Any]) -> None:
        await self._queue.put(payload)

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._queue.get()
            yield item
            self._queue.task_done()

    def publish_threadsafe(self, loop: asyncio.AbstractEventLoop, payload: dict[str, Any]) -> None:
        try:
            asyncio.run_coroutine_threadsafe(self.publish(payload), loop)
        except RuntimeError:
            # Loop might be closed; swallow silently
            pass


class JobRegistry:
    """Manage active inference jobs and their progress streams."""

    def __init__(self) -> None:
        self._streams: Dict[str, ProgressStreamer] = {}
        self._results: Dict[str, Any] = {}
        self._failures: Dict[str, str] = {}

    def create_job(self) -> tuple[str, ProgressStreamer]:
        task_id = uuid.uuid4().hex
        stream = ProgressStreamer()
        self._streams[task_id] = stream
        return task_id, stream

    def get_stream(self, task_id: str) -> Optional[ProgressStreamer]:
        return self._streams.get(task_id)

    def set_result(self, task_id: str, result: Any) -> None:
        self._results[task_id] = result
        self._failures.pop(task_id, None)

    def get_result(self, task_id: str) -> Any:
        return self._results.get(task_id)

    def set_failure(self, task_id: str, detail: str) -> None:
        self._failures[task_id] = detail
        self._results.pop(task_id, None)

    def get_failure(self, task_id: str) -> Optional[str]:
        return self._failures.get(task_id)

    def clear(self, task_id: str) -> None:
        self._streams.pop(task_id, None)
        self._results.pop(task_id, None)
        self._failures.pop(task_id, None)

    def clear_stream(self, task_id: str) -> None:
        self._streams.pop(task_id, None)


JOB_REGISTRY = JobRegistry()


async def run_inference(
    task_id: str,
    task: Callable[..., Any],
    *args: Any,
    stream: Optional[ProgressStreamer] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    **kwargs: Any,
) -> Any:
    """Execute a blocking inference task in a thread pool and store result."""

    event_loop = loop or asyncio.get_running_loop()

    if stream is not None:
        def progress_callback(payload: dict[str, Any]) -> None:
            stream.publish_threadsafe(event_loop, payload)

        kwargs["progress_callback"] = progress_callback

    try:
        result = await event_loop.run_in_executor(None, lambda: task(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001
        JOB_REGISTRY.set_failure(task_id, str(exc))
        raise
    else:
        JOB_REGISTRY.set_result(task_id, result)
        return result



