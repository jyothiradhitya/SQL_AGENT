import logging
import time

import sqlglot
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from lms_chatbot.config import settings
from lms_chatbot.sql.empty_result import diagnose, is_empty
from lms_chatbot.sql.validator import SQL_DIALECT, apply_row_limit, validate_sql

logger = logging.getLogger(__name__)

_engine = create_engine(settings.analytical_db_url, pool_pre_ping=True)

# Diagnostic-only: caps how much SQL text a single log line shows.
_SQL_LOG_TRUNCATE_LEN = 2000


def _truncated_sql(sql: str) -> str:
    if sql is None:
        return ""
    if len(sql) <= _SQL_LOG_TRUNCATE_LEN:
        return sql
    return f"{sql[:_SQL_LOG_TRUNCATE_LEN]}... [truncated, {len(sql)} chars total]"


def execute_sql(
    sql: str,
    limit: int | None = None,
    allowed_tables=None,
    table_columns=None,
    value_checker=None,
    large_objects=None,
) -> dict:
    limit = max(1, min(limit or settings.default_row_limit, settings.max_row_limit))

    logger.info("[SQL_VALIDATION] SQL_VALIDATION_START sql=%r", _truncated_sql(sql))
    try:
        validated_sql = validate_sql(
            sql, allowed_tables=allowed_tables, table_columns=table_columns
        )
    except Exception as e:
        logger.warning(
            "[SQL_VALIDATION] SQL_VALIDATION_FAILED reason=%s", e
        )
        raise
    logger.info("[SQL_VALIDATION] SQL_VALIDATION_PASSED")

    tree = sqlglot.parse_one(validated_sql, read=SQL_DIALECT)
    try:
        if value_checker is not None:
            value_checker.check(tree)
    except ValueError as e:
        logger.warning("[SQL_VALIDATION] FILTER_VALUE_NOT_FOUND reason=%s", e)
        raise
    if large_objects is not None:
        try:
            large_objects.check_query(tree)
        except ValueError as e:
            logger.warning("[SQL_VALIDATION] LARGE_OBJECT_MISUSE reason=%s", e)
            raise
    # One row past the cap, so a result the cap actually cut short can be
    # told apart from one that just happens to have exactly `limit` rows -
    # otherwise "list all 145 user trainings" comes back as 100 rows the
    # model has every reason to believe is the complete set.
    tree = apply_row_limit(tree, limit + 1)
    final_sql = tree.sql(dialect=SQL_DIALECT)

    logger.debug("Executing SQL: %s", final_sql)

    logger.info(
        "[SQL_EXECUTION] SQL_DATABASE_EXECUTION_START sql=%r limit=%d",
        _truncated_sql(final_sql), limit,
    )
    db_start = time.monotonic()

    try:
        with _engine.connect() as connection:
            connection.execute(text("SET TRANSACTION READ ONLY"))
            connection.execute(
                text(f"SET LOCAL statement_timeout = {settings.statement_timeout_ms}")
            )

            # text() treats any `:word` as a bind parameter - including one
            # inside a string literal ('Note :draft'), which then fails with
            # "A value is required for bind parameter". Nothing here binds
            # parameters, so every colon is escaped to a literal one
            # (text() unescapes `\:` back to `:`).
            result = connection.execute(text(final_sql.replace(":", r"\:")))

            # fetchmany caps how many rows are ever pulled into memory,
            # even if the LIMIT above were somehow lost.
            rows = result.fetchmany(limit + 1)
            truncated = len(rows) > limit
            rows = rows[:limit]
            columns = list(result.keys())

            db_duration = time.monotonic() - db_start
            logger.info(
                "[SQL_EXECUTION] SQL_DATABASE_EXECUTION_SUCCESS rows=%d duration=%.2fs",
                len(rows), db_duration,
            )

            payload = {
                "sql": final_sql,
                "columns": columns,
                "rows": [list(row) for row in rows],
                "row_count": len(rows),
                "truncated": truncated,
            }
            # An empty result is re-counted with each condition removed -
            # see sql/empty_result.py for why "no rows" alone isn't enough.
            if is_empty(payload):
                payload["empty_check"] = diagnose(tree, connection)

    except SQLAlchemyError as e:
        db_duration = time.monotonic() - db_start
        logger.warning(
            "[SQL_EXECUTION] SQL_DATABASE_EXECUTION_FAILED duration=%.2fs error=%s",
            db_duration, e,
        )
        raise RuntimeError(f"PostgreSQL execution failed: {e}") from e

    # Large-object OIDs -> their real text, from the source databases
    # (see sql/large_objects.py). Outside the analytical connection: it
    # is a separate read-only lookup and must not hold that one open.
    if large_objects is not None:
        payload = large_objects.resolve(tree, payload)
    return payload
