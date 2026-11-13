"""Submit one existing analysis JSON payload; print the corresponding AMQP reply."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

import aio_pika


async def submit(args):
    payload = json.loads(Path(args.request).read_text())
    request_id = payload.get("request_id") or uuid4().hex
    payload["request_id"] = request_id
    connection = await aio_pika.connect(os.environ["RABBITMQ_URL"], timeout=10)
    async with connection:
        channel = await connection.channel(
            publisher_confirms=True, on_return_raises=True
        )
        # One durable reply queue per logical request allows retrying this CLI with
        # the same request_id after a client disconnect. The broker expires idle queues.
        replies = await channel.declare_queue(
            f"exaflow.analysis.replies.{request_id}",
            durable=True,
            arguments={
                "x-message-ttl": 86400000,
                "x-expires": 172800000,
                "x-max-length-bytes": 268435456,
                "x-overflow": "reject-publish",
            },
        )
        await channel.default_exchange.publish(
            aio_pika.Message(
                json.dumps(payload).encode(),
                content_type="application/json",
                correlation_id=request_id,
                reply_to=replies.name,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key=args.queue,
            mandatory=True,
            timeout=10,
        )
        async with asyncio.timeout(args.timeout):
            async with replies.iterator() as messages:
                async for message in messages:
                    if message.correlation_id != request_id:
                        raise ValueError("Unexpected reply correlation ID")
                    result = json.loads(message.body)
                    print(json.dumps(result, indent=2))
                    await message.ack()
                    return 0 if result["status"] == "succeeded" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Path to an existing /analysis JSON payload")
    parser.add_argument("--queue", default="exaflow.analysis.requests")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    try:
        return asyncio.run(submit(args))
    except Exception as exc:
        # Connection exception messages can contain URLs or authentication details.
        print(
            f"Analysis submission failed ({type(exc).__name__}). Check configuration and controller health.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
