from lms_chatbot.sql.validator import BLOCKED_COLUMN_NAMES

# This used to be its own hand-maintained set, independent of
# sql/validator.py's blocklist - and had already drifted out of sync
# with the real schema (missing auth_string, token, session_id, email,
# mobile_number, birthdate, profile_picture, reset_remarks, ip; see
# validator.py for how that list was built). Reusing
# BLOCKED_COLUMN_NAMES makes this one source of truth instead of two
# that can silently disagree.
#
# In practice sql/executor.py always runs validate_sql() before a query
# reaches format_query_result, so a query naming any of these columns
# never executes in the first place - this redaction never actually
# fires on the live path. It's kept anyway as a second, independent
# layer for any future caller of format_query_result that doesn't go
# through execute_sql/validate_sql first. `password`/`access_token`/
# `refresh_token` aren't in this DB's schema but are kept as generic
# defensive names in case a differently-named credential column shows
# up later.
#
# Deliberately NOT including BLOCKED_QUALIFIED_COLUMNS (e.g. the
# sys_user.image_url special case) - this redaction only ever sees bare
# result column names with no table context, so it can't tell
# sys_user.image_url (blocked) apart from product_store.image_url
# (a store logo, not personal data) the way validator.py can.
SENSITIVE_COLUMNS = BLOCKED_COLUMN_NAMES | {
    "password",
    "access_token",
    "refresh_token",
}


def format_query_result(result: dict) -> str:
    columns = result["columns"]
    rows = result["rows"]
    row_count = result["row_count"]

    # Facts about an empty/zero result (sql/empty_result.py).
    check = result.get("empty_check") or []
    check_lines = (["EMPTY_RESULT_CHECK:"] + [f"  {line}" for line in check]) if check else []

    if not rows:
        return "\n".join(["SQL_SUCCESS", "ROW_COUNT: 0", "RESULT: NO_RESULTS"] + check_lines)

    safe_columns = []
    safe_indexes = []

    for index, column in enumerate(columns):
        if column.lower() in SENSITIVE_COLUMNS:
            continue
        safe_columns.append(column)
        safe_indexes.append(index)

    output = [
        "SQL_SUCCESS",
        f"ROW_COUNT: {row_count}",
    ]
    if result.get("truncated"):
        output.append(
            f"TRUNCATED: true - more rows exist beyond these {row_count}; this is "
            "NOT the full result, so the real total is unknown from this data"
        )
    output += [
        "COLUMNS: " + ", ".join(safe_columns),
        "DATA:",
    ]

    for row in rows:
        values = [
            f"{column}={_one_line(row[index])}" for index, column in zip(safe_indexes, safe_columns)
        ]
        output.append(" | ".join(values))

    # sql/large_objects.py replaced reference numbers with the stored text
    resolved = [c for c in result.get("large_object_columns") or [] if c in safe_columns]
    if resolved:
        output.append(
            f"NOTE: {', '.join(resolved)} hold the real stored text (fetched from the "
            "source system; rich-text formatting removed). Use this text as the source "
            "of truth: quote or summarise it in the answer, never a reference number."
        )
    files = [c for c in result.get("file_columns") or [] if c in safe_columns]
    if files:
        output.append(
            f"NOTE: {', '.join(files)} describe attached files (type and size); their "
            "content is not text and is not available to read."
        )

    return "\n".join(output + check_lines)


def _one_line(value) -> str:
    """Multi-line stored text kept on its row's line ("a / b")."""
    return " / ".join(str(value).splitlines()) if isinstance(value, str) else str(value)
