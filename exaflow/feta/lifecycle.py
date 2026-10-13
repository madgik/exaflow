import logging
import time
from dataclasses import dataclass
from dataclasses import field
from threading import Event
from threading import RLock
from threading import Thread
from uuid import uuid4

from exaflow.feta.settings import FetaSettings


class ContextClosedError(RuntimeError):
    """The context has expired, been removed, or never existed."""


class ResourceLimitError(RuntimeError):
    """A configured FETA memory or context limit was exceeded."""


@dataclass
class StoredContext:
    deadline: float
    values: dict = field(default_factory=dict)
    bytes_used: int = 0


class ContextStore:
    """Shared dictionary; locks protect lifecycle changes, never computation."""

    def __init__(self, settings=None, clock=time.monotonic):
        self.settings = settings or FetaSettings()
        self.clock = clock
        self.contexts = {}
        self.lock = RLock()

    def create(self, context_id, tensors):
        if not context_id or len(context_id) > 128:
            raise ValueError("Invalid FETA context ID.")
        size = sum(array.nbytes for array, _ in tensors.values())
        if size > self.settings.max_context_bytes:
            raise ResourceLimitError("FETA input exceeds the context memory limit.")
        for array, _ in tensors.values():
            if array.nbytes > self.settings.max_tensor_bytes:
                raise ResourceLimitError("FETA input exceeds the tensor memory limit.")
        with self.lock:
            self.expire()
            if context_id in self.contexts:
                raise ValueError("FETA context already exists.")
            if len(self.contexts) >= self.settings.max_contexts:
                raise ResourceLimitError("FETA context count limit exceeded.")
            ctx = StoredContext(
                self.clock() + self.settings.context_lifetime_minutes * 60
            )
            refs = {}
            for name, (array, metadata) in tensors.items():
                array.setflags(write=False)
                tensor_id = uuid4().hex
                ctx.values[tensor_id] = (array, metadata)
                refs[name] = tensor_id
            ctx.bytes_used = size
            self.contexts[context_id] = ctx
            return refs

    def acquire(self, context_id, tensor_ids):
        with self.lock:
            ctx = self._get(context_id)
            try:
                values = [ctx.values[tensor_id] for tensor_id in tensor_ids]
            except KeyError as exc:
                raise ValueError("Unknown tensor reference in this context.") from exc
            return ctx, values

    def put(self, context_id, expected_context, array, metadata):
        with self.lock:
            ctx = self._get(context_id)
            if ctx is not expected_context:
                raise ContextClosedError("FETA context was replaced during execution.")
            if array.nbytes > self.settings.max_tensor_bytes:
                raise ResourceLimitError("FETA result exceeds the tensor memory limit.")
            if ctx.bytes_used + array.nbytes > self.settings.max_context_bytes:
                raise ResourceLimitError("FETA context memory limit exceeded.")
            array.setflags(write=False)
            tensor_id = uuid4().hex
            ctx.values[tensor_id] = (array, metadata)
            ctx.bytes_used += array.nbytes
            return tensor_id

    def check_active(self, context_id, expected_context):
        with self.lock:
            if self._get(context_id) is not expected_context:
                raise ContextClosedError("FETA context was replaced during execution.")

    def _get(self, context_id):
        ctx = self.contexts.get(context_id)
        if ctx is None:
            raise ContextClosedError("FETA context does not exist or is closed.")
        if self.clock() >= ctx.deadline:
            del self.contexts[context_id]
            ctx.values.clear()
            ctx.bytes_used = 0
            raise ContextClosedError("FETA context expired.")
        return ctx

    def cleanup(self, context_id, expected_context=None):
        with self.lock:
            if (
                expected_context is None
                or self.contexts.get(context_id) is expected_context
            ):
                ctx = self.contexts.pop(context_id, None)
                if ctx is not None:
                    ctx.values.clear()
                    ctx.bytes_used = 0

    def expire(self):
        with self.lock:
            expired = [
                key
                for key, ctx in self.contexts.items()
                if self.clock() >= ctx.deadline
            ]
            for key in expired:
                self.cleanup(key)
            return len(expired)


class CleanupJob:
    def __init__(self, callback, interval, on_failure=None):
        self.callback = callback
        self.interval = interval
        self.on_failure = on_failure
        self.stop_event = Event()
        self.thread = Thread(target=self._run, name="feta-context-cleanup", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join()

    def _run(self):
        try:
            while not self.stop_event.wait(self.interval):
                self.callback()
        except Exception:
            logging.getLogger(__name__).exception("FETA background cleanup failed.")
            if self.on_failure is not None:
                self.on_failure()
            self.stop_event.set()
