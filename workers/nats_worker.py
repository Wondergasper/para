"""
APG Python NATS Worker.

Connects to NATS JetStream, subscribes to apg.jobs, runs the pipeline,
and publishes results to apg.results.

Compatible with nats-py >= 2.7.0 (uses push consumer API).

Usage:
    python -m workers.nats_worker

Environment:
    NATS_URL      NATS server URL (default: nats://localhost:4222)
    APG_PROVIDER  LLM provider: ollama|groq|gemini (default: ollama)
    APG_MODEL     Override LLM model name (default: provider default)
"""

import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import replace
from pathlib import Path

import nats
import nats.errors
from nats.aio.client import Client as NATS

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers.pipeline import PipelineConfig, run
from workers.proto import apg_pb2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("apg.worker")


async def main():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    log.info(f"Connecting to NATS at {nats_url}")

    # Graceful shutdown event
    shutdown = asyncio.Event()

    def _signal_handler():
        log.info("Shutdown signal received.")
        shutdown.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows does not support add_signal_handler for SIGTERM
            pass

    try:
        nc = await nats.connect(
            nats_url,
            name="apg-python-worker",
            reconnect_time_wait=2,
            max_reconnect_attempts=10,
        )
    except Exception as e:
        log.error(f"Failed to connect to NATS: {e}")
        log.error("Make sure NATS is running: docker run -p 4222:4222 nats:latest -js")
        return

    js = nc.jetstream()

    # Ensure the stream exists (Go orchestrator creates it, but handle startup ordering)
    try:
        await js.find_stream_name_by_subject("apg.jobs")
        log.info("APG stream found.")
    except Exception:
        log.warning("APG stream not found yet — creating it.")
        try:
            await js.add_stream(name="APG", subjects=["apg.jobs", "apg.results"])
        except Exception as e:
            log.error(f"Could not create APG stream: {e}")

    # Base pipeline config (read from environment)
    base_cfg = PipelineConfig(
        provider=os.getenv("APG_PROVIDER", "ollama"),
        model=os.getenv("APG_MODEL", None),
        enable_tsan=(os.name != "nt"),
        verbose=True,
    )

    log.info(f"Worker ready | provider={base_cfg.provider} | tsan={'on' if base_cfg.enable_tsan else 'off (Windows)'}")

    async def handle_job(msg):
        """Process a single job message."""
        req = apg_pb2.JobRequest()
        try:
            req.ParseFromString(msg.data)
        except Exception as e:
            log.error(f"Failed to parse JobRequest proto: {e}")
            await msg.term()
            return

        job_id = req.id
        log.info(f"[{job_id}] Received job | func={req.func_name} | model_version={req.model_version}")

        # Apply model version routing
        job_cfg = base_cfg
        if req.model_version and req.model_version != "base":
            job_cfg = replace(base_cfg, model_version=req.model_version, job_id=job_id)
        else:
            job_cfg = replace(base_cfg, job_id=job_id)

        try:
            # Run the pipeline in a thread executor so it doesn't block the event loop
            result = await loop.run_in_executor(
                None,
                lambda: run(
                    source_code=req.source,
                    func_name=req.func_name or "func",
                    config=job_cfg,
                ),
            )
        except Exception as e:
            log.error(f"[{job_id}] Pipeline exception: {e}")
            resp = apg_pb2.JobResponse(
                id=job_id,
                success=False,
                score=0.0,
                error=str(e),
            )
            await js.publish("apg.results", resp.SerializeToString())
            await msg.ack()
            return

        # Build response proto
        resp = apg_pb2.JobResponse(
            id=job_id,
            success=result.success,
            score=result.score,
            best_candidate=result.best_candidate,
            annotated_ir_json=json.dumps(result.annotated_ir),
            rounds=result.rounds,
            error=result.error,
        )
        for att in result.attempts:
            resp.attempts.add(
                candidate=att.candidate,
                score=att.score,
                gate=att.gate,
                error=att.error,
            )

        await js.publish("apg.results", resp.SerializeToString())
        await msg.ack()
        log.info(f"[{job_id}] Done | success={result.success} | score={result.score:.2f} | rounds={result.rounds}")

    role = os.getenv("APG_WORKER_ROLE", "all").lower()
    subject = "apg.jobs.*"
    durable_name = "python-worker-all"
    if role == "quick":
        subject = "apg.jobs.quick"
        durable_name = "python-worker-quick"
    elif role == "heavy":
        subject = "apg.jobs.heavy"
        durable_name = "python-worker-heavy"

    log.info(f"Subscribing to subject '{subject}' with durable name '{durable_name}' for role '{role}'")
    # Subscribe with a push consumer (nats-py 2.x API)
    sub = await js.subscribe(
        subject,
        durable=durable_name,
        cb=handle_job,
        manual_ack=True,
    )

    log.info(f"Subscribed to {subject}. Waiting for jobs...")

    # Wait until shutdown signal
    await shutdown.wait()

    log.info("Draining NATS connection...")
    await sub.unsubscribe()
    await nc.drain()
    log.info("Worker shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
