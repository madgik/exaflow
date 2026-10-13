import math
import numbers
from dataclasses import dataclass


@dataclass(frozen=True)
class FetaSettings:
    context_lifetime_minutes: float = 60
    cleanup_interval_seconds: float = 30
    max_contexts: int = 32
    max_context_bytes: int = 256 * 1024 * 1024
    max_tensor_bytes: int = 64 * 1024 * 1024

    def __post_init__(self):
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"FETA {name} must be positive.")
            if name.startswith("max_") and not isinstance(value, numbers.Integral):
                raise ValueError(f"FETA {name} must be an integer.")
        if not math.isfinite(self.context_lifetime_minutes * 60):
            raise ValueError("FETA context lifetime in seconds must be finite.")

    @classmethod
    def from_config(cls, config):
        section = getattr(config, "feta", {})
        return cls(**dict(section))

    def aggregation_thread_capacity(self, base_capacity):
        # At most max_contexts blocking fetches; submissions still need RPC threads.
        if not isinstance(base_capacity, numbers.Integral) or base_capacity <= 0:
            raise ValueError("Aggregation gRPC capacity must be a positive integer.")
        return base_capacity + self.max_contexts
