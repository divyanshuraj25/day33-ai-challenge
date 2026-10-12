"""Structured JSON logging for every pipeline step (Python `logging` module).

Each log line is one JSON object with:
  timestamp, session_id, step_name, input_summary, output_summary, latency_ms
(plus `status` = ok | error). Lines go to logs/pipeline.jsonl.
"""
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import LOGS_DIR

_LOGGER_NAME = "rag_pipeline"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        LOGS_DIR.mkdir(exist_ok=True)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = logging.Formatter("%(message)s")  # message is already JSON
        file_handler = logging.FileHandler(LOGS_DIR / "pipeline.jsonl", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        if os.getenv("LOG_TO_CONSOLE") == "1":
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            logger.addHandler(console)
    return logger


def summarize(value, limit: int = 200) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


@contextmanager
def log_step(session_id: str, step_name: str, input_summary):
    """Time a pipeline step and emit one structured JSON log entry.

    Usage:
        with log_step(sid, "retrieve", question) as rec:
            ...
            rec["output_summary"] = "3 chunks"
    """
    rec = {"output_summary": ""}
    status = "ok"
    start = time.perf_counter()
    try:
        yield rec
    except Exception as exc:  # log the failure, then re-raise
        status = "error"
        rec["output_summary"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "step_name": step_name,
            "input_summary": summarize(input_summary),
            "output_summary": summarize(rec["output_summary"]),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "status": status,
        }
        get_logger().info(json.dumps(entry, ensure_ascii=False))
