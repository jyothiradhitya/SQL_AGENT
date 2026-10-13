"""
Fail-fast startup validation, run once before the CLI loop starts.

config.Settings' _require() already raises immediately if OLLAMA_URL or
ANALYTICAL_DB are missing entirely - but a present-but-WRONG value only
used to surface deep inside a request: a typo'd DSN showed up as a raw
SQLAlchemy stack trace on the first query, and a mistyped or never-pulled
model name showed up as an indefinite-looking stall inside an LLM call
with no error at all (this is exactly what happened debugging the
orchestrator rewrite: SQL_MODEL=qwen3:8b looked like valid config but
the model was effectively unusable on that Ollama server, and nothing
checked that ahead of time - it just looked like a hang).

These checks are deliberately about "is the config wired to something
that exists and responds at all", not "is it fast enough" - a real but
slow model still passes; that's a deployment/hardware question, not a
config error.
"""

import logging
import sys

import httpx
from sqlalchemy import create_engine, text

from lms_chatbot.config import settings

logger = logging.getLogger(__name__)


class StartupCheckError(RuntimeError):
    pass


def _check_database() -> None:
    try:
        engine = create_engine(settings.analytical_db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        raise StartupCheckError(
            "Could not connect to ANALYTICAL_DB. Check the DSN in .env is "
            f"correct and the database is reachable.\nUnderlying error: {e}"
        ) from e


def _check_ollama_models() -> None:
    required = {
        "ORCHESTRATOR_MODEL": settings.orchestrator_model,
        "SQL_MODEL": settings.sql_model,
        "NLP_MODEL": settings.nlp_model,
        "EMBED_MODEL": settings.embed_model,
    }

    try:
        response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=10)
        response.raise_for_status()
        available = {m["name"] for m in response.json().get("models", [])}
    except Exception as e:
        raise StartupCheckError(
            f"Could not reach Ollama at {settings.ollama_url}/api/tags. Check "
            f"OLLAMA_URL and that the server is running.\nUnderlying error: {e}"
        ) from e

    # Ollama's /api/tags always includes the ":tag" suffix (e.g.
    # "qwen3:8b", "qwen2.5:3b" -> stored as "...:latest" if untagged), so a
    # configured name given without a tag should still match "<name>:latest".
    def _known(name: str) -> bool:
        return name in available or f"{name}:latest" in available

    missing = {var: model for var, model in required.items() if not _known(model)}

    if missing:
        raise StartupCheckError(
            "The following configured models are not available on this Ollama "
            "server. Pull them (`ollama pull <model>`) or fix the env var:\n"
            + "\n".join(f"  {var}={model}" for var, model in missing.items())
        )


def _check_semantic_kb() -> None:
    if not settings.semantic_kb_path.exists():
        raise StartupCheckError(
            f"SEMANTIC_KB_PATH does not exist: {settings.semantic_kb_path}\n"
            "This is the schema knowledge base the retrieval pipeline reads "
            "from - without it, get_relevant_schema can never return results."
        )


def run_startup_checks() -> None:
    logger.info("Running startup checks (database, Ollama models, schema KB)...")

    for check in (_check_database, _check_ollama_models, _check_semantic_kb):
        check()

    logger.info("Startup checks passed")


if __name__ == "__main__":
    from lms_chatbot.logging_config import setup_logging

    setup_logging()

    try:
        run_startup_checks()
    except StartupCheckError as e:
        print(f"STARTUP CHECK FAILED:\n{e}", file=sys.stderr)
        sys.exit(1)
