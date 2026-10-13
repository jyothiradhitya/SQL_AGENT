"""
Real MCP server (stdio transport) exposing the schema-retrieval and
SQL-execution tools to the SQL agent.

IMPORTANT: this process's stdout IS the MCP protocol channel. Never
print() here, or anything that writes to stdout - it will corrupt the
stream the client is reading. All logging goes to stderr, configured
in logging_config.setup_logging().

Run standalone for a quick manual check:
    uv run python -m lms_chatbot.mcp_server.server
(it will just sit there waiting for a client on stdin/stdout - Ctrl+C
to stop). Normally it is launched as a subprocess by mcp_client.client.
"""

import logging
import time

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from lms_chatbot.config import settings
from lms_chatbot.logging_config import setup_logging
from lms_chatbot.retrieval.pipeline import SchemaRetriever
from lms_chatbot.sql.executor import _engine, execute_sql
from lms_chatbot.sql.formatting import format_query_result
from lms_chatbot.sql.large_objects import LargeObjectResolver
from lms_chatbot.sql.value_check import ValueChecker, build_value_index

setup_logging()
logger = logging.getLogger(__name__)

mcp = FastMCP("LMSChatbotMCP", log_level="ERROR")

# One retriever per server process: it's read-only (schema KB + FK
# graph + precomputed table embeddings), so sharing it across every
# tool call is safe. Contrast with the per-request session state in
# agents/session.py, which exists precisely because SQL execution
# results must NOT be shared across concurrent users.
_retriever = SchemaRetriever()
_table_columns = _retriever.table_columns
_value_checker = ValueChecker(
    _engine,
    _table_columns,
    build_value_index(_retriever.sample_values, _retriever.enum_values, _retriever.identifier_values),
)
_large_objects = LargeObjectResolver(
    _retriever.kb, settings.source_db_urls, settings.large_object_max_chars
)

# Diagnostic-only: caps how much of a SQL string a single log line
# shows. Never affects what is validated or executed.
_SQL_LOG_TRUNCATE_LEN = 2000


def _truncated_sql(sql: str) -> str:
    if sql is None:
        return ""
    if len(sql) <= _SQL_LOG_TRUNCATE_LEN:
        return sql
    return f"{sql[:_SQL_LOG_TRUNCATE_LEN]}... [truncated, {len(sql)} chars total]"


@mcp.tool(
    name="get_relevant_schema",
    description=(
        "Look up the tables, columns and join paths relevant to a natural-language "
        "database question. Call this BEFORE run_sql. If run_sql later fails because "
        "a table you need wasn't included, call this again with a more specific or "
        "differently-worded question to pull in additional tables."
    ),
)
def get_relevant_schema(
    question: str = Field(
        description="The user's database question, or a refined version of it"
    ),
) -> str:
    logger.info("get_relevant_schema(%r)", question)
    logger.info("[SCHEMA] SCHEMA_RETRIEVAL_START question=%r", question)

    start = time.monotonic()
    result = _retriever.get_relevant_schema(question)
    duration = time.monotonic() - start

    if not result.tables:
        logger.warning(
            "[SCHEMA] SCHEMA_RETRIEVAL_EMPTY question=%r duration=%.2fs",
            question, duration,
        )
        return (
            "NO_TABLES_FOUND\n"
            "No relevant tables were found for this question. "
            "Rephrase the question or ask a more specific one."
        )

    logger.info("get_relevant_schema -> %s", result.tables)
    logger.info(
        "[SCHEMA] SCHEMA_RETRIEVAL_SUCCESS tables=%s duration=%.2fs",
        result.tables, duration,
    )

    return result.context


@mcp.tool(
    name="run_sql",
    description=(
        "Validate and execute ONE read-only PostgreSQL SELECT/WITH query. "
        "Only reference tables/columns returned by get_relevant_schema. "
        "Returns row data on success, or a structured SQL_ERROR you should use "
        "to fix and retry."
    ),
)
def run_sql(sql: str = Field(description="One PostgreSQL SELECT or WITH query")) -> str:
    logger.info("run_sql(%r)", sql)
    logger.info("[SQL_EXECUTION] SQL_EXECUTION_START sql=%r", _truncated_sql(sql))

    start = time.monotonic()
    try:
        result = execute_sql(
            sql,
            allowed_tables=_retriever.all_table_names,
            table_columns=_table_columns,
            value_checker=_value_checker,
            large_objects=_large_objects,
        )
    except Exception as e:
        duration = time.monotonic() - start
        logger.warning("run_sql failed: %s", e)
        logger.warning(
            "[SQL_EXECUTION] SQL_EXECUTION_FAILED duration=%.2fs error=%s",
            duration, e,
        )
        return f"SQL_ERROR\nERROR: {e}\nACTION: Fix the SQL and try again."

    duration = time.monotonic() - start
    logger.info(
        "[SQL_EXECUTION] SQL_EXECUTION_SUCCESS rows=%d duration=%.2fs",
        result.get("row_count", 0), duration,
    )

    return format_query_result(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")
