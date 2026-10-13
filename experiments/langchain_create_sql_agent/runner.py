"""
Baseline benchmark: point LangChain's stock create_sql_agent() straight
at the production PostgreSQL database and the local Ollama SQL model,
with no semantic KB, no curated join graph, no custom prompt - just the
four tools the SQL toolkit ships (sql_db_list_tables, sql_db_schema,
sql_db_query_checker, sql_db_query) and the default SQL agent prompt.

This is a benchmark, not a service: it opens one read-only DB
connection, runs a fixed list of questions once each, prints/records
what happened, and exits. It intentionally does not import
lms_chatbot.agents.sql_agent, lms_chatbot.retrieval, or SQL_AGENT/ - the
only thing it reuses from the production code is connection/model
*configuration* (lms_chatbot.config.settings), so this stays an
independent measurement of the library's own behavior.

Usage:
    python -m experiments.langchain_create_sql_agent
    python -m experiments.langchain_create_sql_agent --limit 3
    python -m experiments.langchain_create_sql_agent --question "..." --question "..."
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from sqlalchemy import create_engine

from lms_chatbot.agents.llms import get_sql_llm
from lms_chatbot.config import settings
from lms_chatbot.logging_config import setup_logging

from .multi_schema_db import MultiSchemaSQLDatabase
from .questions import DEFAULT_QUESTIONS
from .tracing import SQLAgentTracer

logger = logging.getLogger("experiments.langchain_create_sql_agent")

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _build_read_only_engine():
    # Defense in depth for requirement "do not modify the database":
    # postgresql_readonly rejects any write at the wire protocol level
    # regardless of what SQL the agent decides to generate, and the
    # statement timeout stops a wrong-table guess (e.g. an unintended
    # cross join) from running forever. Neither changes which tables or
    # tools the agent sees, so the agent's own behavior is unaffected.
    return create_engine(
        settings.analytical_db_url,
        pool_pre_ping=True,
        execution_options={"postgresql_readonly": True},
        connect_args={"options": f"-c statement_timeout={settings.statement_timeout_ms}"},
    )


def _build_agent(schemas: list[str] | None, top_k: int, max_iterations: int, verbose: bool):
    engine = _build_read_only_engine()
    db = MultiSchemaSQLDatabase.from_engine(engine, schemas=schemas)

    logger.info(
        "Database ready: %d tables usable=%s",
        len(db.get_usable_table_names()), sorted(db.get_usable_table_names()),
    )

    llm = get_sql_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="tool-calling",
        verbose=verbose,
        top_k=top_k,
        max_iterations=max_iterations,
    )
    return agent_executor, db


def _run_question(agent_executor, question: str) -> dict:
    tracer = SQLAgentTracer(question)
    logger.info("=" * 100)
    logger.info("QUESTION: %s", question)

    start = time.monotonic()
    try:
        agent_executor.invoke({"input": question}, config={"callbacks": [tracer]})
    except Exception as exc:  # noqa: BLE001 - a raw agent's failure IS a result here
        tracer.trace.error = f"{type(exc).__name__}: {exc}"
        logger.warning("Agent invocation raised: %s", tracer.trace.error)
    tracer.trace.agent_duration_s = time.monotonic() - start

    logger.info(
        "RESULT duration=%.2fs llm_calls=%d tool_calls=%d tables_inspected=%s",
        tracer.trace.agent_duration_s,
        len(tracer.trace.llm_calls),
        len(tracer.trace.tool_calls),
        sorted(tracer.trace.tables_inspected),
    )

    return tracer.trace.to_dict()


def run(
    questions: list[str],
    schemas: list[str] | None = None,
    top_k: int = 10,
    max_iterations: int = 15,
    verbose: bool = True,
    output_path: Path | None = None,
) -> list[dict]:
    setup_logging()
    # LOG_LEVEL=DEBUG (as configured in .env) leaves httpx/httpcore at DEBUG,
    # which logs every streamed HTTP chunk from Ollama - useless noise for
    # an agent trajectory trace, so it's silenced here regardless of the
    # configured level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    agent_executor, db = _build_agent(
        schemas=schemas, top_k=top_k, max_iterations=max_iterations, verbose=verbose
    )

    results = []
    for question in questions:
        results.append(_run_question(agent_executor, question))

    _print_summary(results)

    output_path = output_path or (
        RUNS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "sql_model": settings.sql_model,
                "ollama_url": settings.ollama_url,
                "schemas": schemas or sorted({t.split(".")[0] for t in db.get_usable_table_names()}),
                "top_k": top_k,
                "max_iterations": max_iterations,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote full trace to %s", output_path)

    return results


def _print_summary(results: list[dict]) -> None:
    logger.info("=" * 100)
    logger.info("SUMMARY (%d questions)", len(results))
    for r in results:
        status = "ERROR" if r["error"] else "OK"
        logger.info(
            "[%s] %.2fs  llm_calls=%d  tools=%d  tables=%s  q=%r",
            status,
            r["agent_duration_s"] or 0.0,
            r["llm_call_count"],
            len(r["tool_calls"]),
            r["tables_inspected"],
            r["question"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline benchmark for LangChain's create_sql_agent() "
        "against the real database and Ollama, without the semantic KB."
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Ask this question instead of the default set. Repeatable.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N questions."
    )
    parser.add_argument(
        "--schemas",
        type=str,
        default=None,
        help="Comma-separated schema names to reflect (default: auto-discover all).",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument(
        "--quiet", action="store_true", help="Disable AgentExecutor's own verbose printout."
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to write the JSON trace to."
    )
    args = parser.parse_args()

    questions = args.questions or DEFAULT_QUESTIONS
    if args.limit is not None:
        questions = questions[: args.limit]

    schemas = args.schemas.split(",") if args.schemas else None
    output_path = Path(args.output) if args.output else None

    run(
        questions=questions,
        schemas=schemas,
        top_k=args.top_k,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
