import os
import sys
import json

from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


# ============================================================
# IMPORT SQL FUNCTIONS
# ============================================================

from .validate_sql import (
    validate_sql as _validate_sql,
    execute_sql as _execute_sql,
)


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")

if not OLLAMA_URL:
    raise ValueError(
        "OLLAMA_URL is not defined in .env"
    )


MODEL = "qwen3:8b"

VERBOSE = (
    "-v" in sys.argv
    or "--verbose" in sys.argv
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SNAPSHOT_FILE = (
    BASE_DIR
    / "SQL_AGENT"
    / "Data"
    / "database_snapshot.json"
)


# ============================================================
# LOAD DATABASE SNAPSHOT
# ============================================================

with open(
    SNAPSHOT_FILE,
    "r",
    encoding="utf-8"
) as f:

    database_snapshot = json.load(f)


# ============================================================
# BUILD RELEVANT SCHEMA
#
# Keep your existing FULL_SCHEMA behavior.
# Your retrieval layer can replace/filter this before use.
# ============================================================

schema_parts = []
allowed_tables = []

for schema in database_snapshot.get(
    "schemas",
    []
):

    schema_name = schema.get(
        "schema_name",
        ""
    )

    for table in schema.get(
        "tables",
        []
    ):

        # ----------------------------------------------------
        # Ignore snapshot errors
        # ----------------------------------------------------

        if "error" in table:
            continue

        table_name = table.get(
            "table_name",
            ""
        )

        qualified_name = table.get(
            "qualified_name",
            f"{schema_name}.{table_name}"
        )

        schema_info = table.get(
            "schema_info",
            ""
        )

        if not schema_info:
            continue

        allowed_tables.append(
            qualified_name
        )

        schema_parts.append(
            f"""
TABLE: {qualified_name}

{schema_info}
"""
        )


FULL_SCHEMA = "\n\n".join(
    schema_parts
)


ALLOWED_TABLES = "\n".join(
    f"- {table}"
    for table in allowed_tables
)


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("[SQL AGENT] Creating agent")
print("=" * 70)

print(
    "[SQL AGENT] Model:",
    MODEL
)

print(
    "[SQL AGENT] Verbose:",
    VERBOSE
)


# ============================================================
# SQL TOOL
# ============================================================

@tool
def run_sql_tool(sql: str) -> str:
    """
    Validate and execute a PostgreSQL SELECT query.
    """

    print()
    print("=" * 70)
    print("[SQL AGENT] SQL TOOL")
    print("=" * 70)

    print(
        "[SQL AGENT] GENERATED SQL:"
    )

    print(sql)


    # ========================================================
    # STEP 1 — VALIDATION
    # ========================================================

    print()
    print(
        "[SQL AGENT] STEP 1: VALIDATING SQL"
    )

    try:

        validated_sql = _validate_sql(
            sql,
            allowed_tables=allowed_tables,
        )

    except Exception as e:

        print(
            "[SQL AGENT] VALIDATION FAILED"
        )

        print(
            "[SQL AGENT] ERROR:",
            e
        )

        return (
            "SQL_ERROR\n"
            "ERROR_TYPE: VALIDATION_ERROR\n"
            f"ERROR: {str(e)}\n"
            "ACTION: Fix the SQL and try again."
        )


    print(
        "[SQL AGENT] VALIDATION SUCCESS"
    )

    print(
        "[SQL AGENT] VALIDATED SQL:"
    )

    print(
        validated_sql
    )


    # ========================================================
    # STEP 2 — EXECUTION
    # ========================================================

    print()
    print(
        "[SQL AGENT] STEP 2: EXECUTING SQL"
    )

    try:

        result = _execute_sql(
            validated_sql,
            limit=50,
            allowed_tables=allowed_tables,
        )

    except Exception as e:

        print(
            "[SQL AGENT] EXECUTION FAILED"
        )

        print(
            "[SQL AGENT] ERROR:",
            e
        )

        return (
            "SQL_ERROR\n"
            "ERROR_TYPE: EXECUTION_ERROR\n"
            f"ERROR: {str(e)}\n"
            "ACTION: Fix the SQL and try again."
        )


    # ========================================================
    # RESULT INFORMATION
    # ========================================================

    columns = result["columns"]
    rows = result["rows"]
    row_count = result["row_count"]

    print(
        "[SQL AGENT] EXECUTION SUCCESS"
    )

    print(
        "[SQL AGENT] ROWS RETURNED:",
        row_count
    )


    # ========================================================
    # NO RESULTS
    # ========================================================

    if not rows:

        print(
            "[SQL AGENT] NO RESULTS"
        )

        return (
            "SQL_SUCCESS\n"
            "ROW_COUNT: 0\n"
            "RESULT: NO_RESULTS"
        )


    # ========================================================
    # SENSITIVE COLUMNS
    # ========================================================

    sensitive_columns = {
        "password",
        "password_hash",
        "activation_key",
        "reset_key",
        "access_token",
        "refresh_token",
    }


    safe_columns = []
    safe_indexes = []


    for index, column in enumerate(
        columns
    ):

        if column.lower() in sensitive_columns:
            continue

        safe_columns.append(
            column
        )

        safe_indexes.append(
            index
        )


    # ========================================================
    # FORMAT DATABASE RESULT
    # ========================================================

    output = [
        "SQL_SUCCESS",
        f"ROW_COUNT: {row_count}",
        "COLUMNS: " + ", ".join(
            safe_columns
        ),
        "DATA:"
    ]


    for row in rows:

        values = []

        for index, column in zip(
            safe_indexes,
            safe_columns
        ):

            value = row[index]

            values.append(
                f"{column}={value}"
            )

        output.append(
            " | ".join(values)
        )


    database_result = "\n".join(
        output
    )


    # ========================================================
    # VERBOSE DATABASE OUTPUT
    # ========================================================

    if VERBOSE:

        print()
        print(
            "[SQL AGENT] DATABASE RESULT"
        )

        print(
            "-" * 70
        )

        print(
            database_result
        )

        print(
            "-" * 70
        )


    return database_result


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model=MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
    num_predict=120,
    think=False,
)


# ============================================================
# TOOL-ENABLED LLM
# ============================================================

llm_with_tools = llm.bind_tools(
    [run_sql_tool]
)


# ============================================================
# SQL AGENT PROMPT
#
# IMPORTANT:
# Do NOT tell the model to always JOIN.
# JOIN is conditional.
# ============================================================

SQL_AGENT_PROMPT = f"""
You are a PostgreSQL SQL agent.

Your ONLY job is:
1. understand the user's database question
2. generate ONE correct PostgreSQL SELECT query
3. call run_sql_tool

SCHEMA:
{FULL_SCHEMA}

ALLOWED TABLES:
{ALLOWED_TABLES}

RULES:

- Use ONLY tables and columns present in SCHEMA.
- Use exact schema-qualified table names.
- NEVER invent a column, table, alias, or relationship.
- NEVER use SELECT *.
- PostgreSQL syntax only.
- Only SELECT or WITH queries.

TABLE SELECTION:

- First determine exactly what information the user requested.
- Find the table containing that information.
- Use the smallest number of tables required.
- If ONE table contains everything needed, use ONLY that table.
- DO NOT add a JOIN just because another related table exists.
- DO NOT JOIN merely to make the query look complete.

JOIN RULE:

- Use JOIN ONLY when the requested answer actually requires
  information from another table.
- If another table is required, use the relationship explicitly
  shown in SCHEMA.
- Never guess a relationship.
- JOIN or an equivalent subquery is acceptable.
- Do not add unnecessary JOINs.

SQL CORRECTNESS:

- Every alias must be defined before use.
- Every referenced column must exist in SCHEMA.
- Do not reference an alias that was never declared.
- Do not use a column from one table through another table's alias.
- Do not invent foreign keys.
- Do not add unnecessary GROUP BY.
- Do not aggregate unless the question requires aggregation.
- Do not use DISTINCT unless required.
- Do not use ORDER BY unless useful or requested.
- Do not use LIMIT unless requested or needed for safety.

POSTGRESQL:

- Use PostgreSQL syntax.
- Do NOT use MySQL functions.
- Do NOT use GROUP_CONCAT.
- Use STRING_AGG only when string aggregation is actually required.

QUESTION SCOPE:

- If the user asks about one specific user, filter to that user.
- If the user asks about all users, return all matching users.
- Do not silently broaden or narrow the requested scope.

IMPORTANT:

- Do not explain your reasoning.
- Do not write a plan.
- Do not answer the user before SQL execution.
- Do not generate multiple SQL queries.
- Call run_sql_tool immediately with ONE query.
- If SQL execution fails, fix the specific error and retry.
"""


# ============================================================
# STARTUP
# ============================================================

print(
    "[SQL AGENT] Tool bound"
)

print(
    "[SQL AGENT] Ready"
)

print("=" * 70)


# ============================================================
# FORMAT FINAL RESULT
#
# No second Qwen call.
# This prevents:
# - duplicated rows
# - hallucinated values
# - unnecessary CPU inference
# ============================================================

def format_database_result(
    database_result: str
) -> str:

    if not database_result:
        return (
            "I could not retrieve the requested "
            "database information."
        )


    if "RESULT: NO_RESULTS" in database_result:

        return (
            "No matching records were found."
        )


    lines = database_result.splitlines()


    # --------------------------------------------------------
    # Extract row count
    # --------------------------------------------------------

    row_count = None

    for line in lines:

        if line.startswith(
            "ROW_COUNT:"
        ):

            try:

                row_count = int(
                    line.split(
                        ":",
                        1
                    )[1].strip()
                )

            except Exception:
                pass

            break


    # --------------------------------------------------------
    # Find DATA section
    # --------------------------------------------------------

    data_index = None

    for index, line in enumerate(
        lines
    ):

        if line.strip() == "DATA:":
            data_index = index
            break


    if data_index is None:
        return database_result


    data_lines = [
        line.strip()
        for line in lines[
            data_index + 1:
        ]
        if line.strip()
    ]


    if not data_lines:
        return "No matching records were found."


    # --------------------------------------------------------
    # Parse rows
    # --------------------------------------------------------

    parsed_rows = []

    for line in data_lines:

        row = {}

        parts = [
            part.strip()
            for part in line.split("|")
        ]

        for part in parts:

            if "=" not in part:
                continue

            key, value = part.split(
                "=",
                1
            )

            row[key.strip()] = value.strip()


        if row:
            parsed_rows.append(row)


    if not parsed_rows:
        return database_result


    # ========================================================
    # SINGLE COLUMN RESULT
    #
    # Example:
    #
    # login
    # user
    # admin1
    # admin2
    # ========================================================

    if all(
        len(row) == 1
        for row in parsed_rows
    ):

        values = []

        for row in parsed_rows:

            value = next(
                iter(row.values())
            )

            values.append(
                value
            )


        return "\n".join(
            values
        )


    # ========================================================
    # MULTI-COLUMN RESULT
    # ========================================================

    columns = list(
        parsed_rows[0].keys()
    )


    output = []

    for row in parsed_rows:

        values = []

        for column in columns:

            values.append(
                str(
                    row.get(
                        column,
                        ""
                    )
                )
            )

        output.append(
            " | ".join(values)
        )


    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    header = " | ".join(
        columns
    )

    result = [
        header,
        "-" * len(header),
        *output
    ]


    return "\n".join(
        result
    )


# ============================================================
# RUN SQL AGENT
# ============================================================

def run_sql_agent(
    question: str
) -> str:

    print()
    print("=" * 70)
    print("[SQL AGENT] START")
    print("=" * 70)

    print(
        "[SQL AGENT] USER QUESTION:"
    )

    print(
        question
    )


    # ========================================================
    # MESSAGE HISTORY
    # ========================================================

    messages = [
        {
            "role": "system",
            "content": SQL_AGENT_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Generate the SQL required to answer this "
                "database question. Use the minimum required "
                "tables. Use a JOIN only if another table is "
                "actually required.\n\n"
                f"QUESTION: {question}"
            ),
        },
    ]


    # ========================================================
    # RETRY CONFIGURATION
    # ========================================================

    max_attempts = 4

    sql_success = False

    database_result = None


    # ========================================================
    # SQL LOOP
    # ========================================================

    for attempt in range(
        1,
        max_attempts + 1
    ):

        print()
        print("=" * 70)

        print(
            f"[SQL AGENT] SQL ATTEMPT "
            f"{attempt}/{max_attempts}"
        )

        print("=" * 70)


        # ----------------------------------------------------
        # QWEN
        # ----------------------------------------------------

        print(
            "[SQL AGENT] Calling Qwen for SQL"
        )


        response = llm_with_tools.invoke(
            messages
        )


        # ----------------------------------------------------
        # TOOL CALLS
        # ----------------------------------------------------

        tool_calls = getattr(
            response,
            "tool_calls",
            []
        )


        # ====================================================
        # NO TOOL CALL
        # ====================================================

        if not tool_calls:

            print()
            print(
                "[SQL AGENT] Qwen did not call SQL tool"
            )


            messages.append(
                response
            )


            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Call run_sql_tool now. "
                        "Return ONE PostgreSQL SELECT query. "
                        "Do not explain."
                    ),
                }
            )

            continue


        # ----------------------------------------------------
        # STORE ASSISTANT RESPONSE
        # ----------------------------------------------------

        messages.append(
            response
        )


        # ====================================================
        # PROCESS TOOL CALLS
        # ====================================================

        for tool_call in tool_calls:

            tool_name = tool_call.get(
                "name"
            )

            tool_args = tool_call.get(
                "args",
                {}
            )

            tool_call_id = tool_call.get(
                "id"
            )


            print()
            print(
                "[SQL AGENT] TOOL:",
                tool_name
            )

            print(
                "[SQL AGENT] ARGS:",
                tool_args
            )


            # ------------------------------------------------
            # UNKNOWN TOOL
            # ------------------------------------------------

            if tool_name != "run_sql_tool":

                tool_result = (
                    "SQL_ERROR\n"
                    "ERROR_TYPE: UNKNOWN_TOOL\n"
                    f"Unknown tool: {tool_name}\n"
                    "ACTION: Use run_sql_tool."
                )


            else:

                sql = tool_args.get("sql")

                if isinstance(sql, dict):
                    sql = sql.get("value")


                # --------------------------------------------
                # MISSING SQL
                # --------------------------------------------

                if not sql:

                    tool_result = (
                        "SQL_ERROR\n"
                        "ERROR_TYPE: MISSING_SQL\n"
                        "No SQL was supplied.\n"
                        "ACTION: Generate one SELECT query."
                    )

                else:

                    # ----------------------------------------
                    # EXECUTE
                    # ----------------------------------------

                    tool_result = (
                        run_sql_tool.invoke(
                            {
                                "sql": sql
                            }
                        )
                    )


            tool_result = str(
                tool_result
            )


            print()
            print(
                "[SQL AGENT] TOOL RESULT RECEIVED"
            )


            # =================================================
            # SUCCESS
            # =================================================

            if tool_result.startswith(
                "SQL_SUCCESS"
            ):

                sql_success = True

                database_result = (
                    tool_result
                )


                print(
                    "[SQL AGENT] SQL EXECUTION SUCCESS"
                )


                # IMPORTANT:
                #
                # Do NOT call Qwen again.
                # Do NOT continue the retry loop.
                #

                break


            # =================================================
            # FAILURE
            # =================================================

            print(
                "[SQL AGENT] SQL EXECUTION FAILED"
            )

            print(
                tool_result
            )


            # ------------------------------------------------
            # SEND ERROR TO QWEN
            #
            # Compact correction prompt.
            # This is intentionally much shorter.
            # ------------------------------------------------

            messages.append(
                ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )


            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The SQL failed. Fix ONLY the SQL error. "
                        "Use only columns/tables in the schema. "
                        "Do not add unnecessary JOINs. "
                        "Use PostgreSQL syntax. "
                        "Call run_sql_tool immediately with "
                        "one corrected SELECT query. "
                        "Do not explain."
                    ),
                }
            )


        # ====================================================
        # SUCCESS → EXIT
        # ====================================================

        if sql_success:

            print()
            print("=" * 70)

            print(
                "[SQL AGENT] SQL SUCCESS"
            )

            print(
                "[SQL AGENT] Leaving SQL retry loop"
            )

            print("=" * 70)

            break


        # ====================================================
        # FAILURE → RETRY
        # ====================================================

        print()
        print(
            "[SQL AGENT] SQL FAILED"
        )


        if attempt < max_attempts:

            print(
                "[SQL AGENT] Retrying SQL generation"
            )

        else:

            print(
                "[SQL AGENT] Maximum SQL attempts reached"
            )


    # ========================================================
    # SQL NEVER SUCCEEDED
    # ========================================================

    if not sql_success:

        print()
        print("=" * 70)

        print(
            "[SQL AGENT] DATABASE QUERY FAILED"
        )

        print("=" * 70)

        return (
            "I could not retrieve the requested "
            "database information."
        )


    # ========================================================
    # DATABASE RESULT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "[SQL AGENT] DATABASE RESULT RECEIVED"
    )

    print("=" * 70)


    if VERBOSE:

        print(
            database_result
        )


    # ========================================================
    # FINAL ANSWER
    #
    # NO SECOND QWEN CALL.
    # ========================================================

    print()
    print(
        "[SQL AGENT] Formatting database result"
    )


    final_answer = format_database_result(
        database_result
    )


    # ========================================================
    # FINAL ANSWER
    # ========================================================

    print()
    print("=" * 70)

    print(
        "[SQL AGENT] FINAL ANSWER"
    )

    print("=" * 70)

    print(
        final_answer
    )

    print("=" * 70)


    return final_answer