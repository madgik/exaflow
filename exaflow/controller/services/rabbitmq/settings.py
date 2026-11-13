from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


class RabbitMQSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    enabled: bool = False
    url: str = Field(default="", repr=False)
    request_queue: str = "exaflow.analysis.requests"
    dead_letter_queue: str = "exaflow.analysis.dead"
    max_result_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    queue_max_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    reply_ttl_ms: int = Field(default=86400000, ge=1)
    dead_letter_ttl_ms: int = Field(default=604800000, ge=1)
    operation_timeout: float = Field(default=5, gt=0, le=5)
    shutdown_drain_timeout: float = Field(default=30, ge=0, le=30)

    @model_validator(mode="after")
    def validate_connection(self):
        if self.enabled and not self.url.startswith(("amqp://", "amqps://")):
            raise ValueError("Enabled RabbitMQ requires an AMQP URL")
        if not self.request_queue or not self.dead_letter_queue:
            raise ValueError("RabbitMQ queue names must be nonempty")
        if self.request_queue == self.dead_letter_queue:
            raise ValueError("Request and dead-letter queues must differ")
        return self
