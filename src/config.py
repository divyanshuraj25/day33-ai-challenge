"""Central configuration.

BASELINE is intentionally flawed (tiny chunks + permissive prompt) so that real,
reproducible failures show up in the eval suite. FIXES are the knobs we turn to
repair them, one at a time, so every fix can be attributed to a failure.
"""
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
DATASET_PATH = ROOT / "eval" / "dataset.json"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

# Scores are on a 0-3 scale. A score BELOW this value counts as a failure.
FAIL_THRESHOLD = 2

BASELINE_PROMPT = """You are a friendly NimbusDesk customer support assistant.
Be concise and helpful. Use the context if it is relevant; otherwise answer from your
general knowledge of customer support software.

Context:
{context}"""

GROUNDED_PROMPT = """You are a NimbusDesk customer support assistant.
Answer ONLY using the context below. Do not use outside knowledge or industry norms.
If the context does not contain the answer, reply exactly:
"I don't have that information in the NimbusDesk documentation."
Quote numbers, durations and limits exactly as written in the context.
Keep the answer to 1-2 sentences.

Context:
{context}"""


@dataclass(frozen=True)
class PipelineConfig:
    name: str = "baseline"
    chunk_size: int = 60            # BUG 1: far too small, splits facts across chunks
    chunk_overlap: int = 0
    top_k: int = 2
    min_score: Optional[float] = None   # retrieval similarity threshold (None = off)
    system_prompt: str = BASELINE_PROMPT  # BUG 2: allows falling back to general knowledge
    chat_model: str = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")


FIXES = {
    "chunking": {"chunk_size": 400, "chunk_overlap": 60},
    "prompt": {"system_prompt": GROUNDED_PROMPT},
    "threshold": {"min_score": 0.25},
}


def build_config(fixes=()) -> PipelineConfig:
    """Return BASELINE with the named fixes applied, e.g. ["chunking", "prompt"]."""
    cfg = PipelineConfig()
    for fix in fixes:
        if fix not in FIXES:
            raise ValueError(f"Unknown fix '{fix}'. Choose from {list(FIXES)}")
        cfg = replace(cfg, **FIXES[fix])
    name = "baseline" if not fixes else "fixed+" + "+".join(fixes)
    return replace(cfg, name=name)
