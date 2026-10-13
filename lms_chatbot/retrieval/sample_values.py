"""
Real lookup-table sample rows, loaded from the DB snapshot the old
SQL_AGENT pipeline captured (SQL_AGENT/Data/database_snapshot.json).

semantic_kb.json has per-column enrichment fields (business_description,
example_values, synonyms, ...) in its schema, but none of this KB's 535
columns actually have them populated - the KB-generation step never
filled them in. For lookup tables specifically that gap matters: an
id/name pair like (2, "APPROVED") *is* the "active" business meaning
("Active" is literally the description text on that row) and nothing
else in the KB tells the SQL agent that - it would otherwise have to
guess. Since lookup tables are small (attach_lookups only ever pulls in
the direct FK targets classify.is_lookup_table() flags, and those cap
out around 20 rows in the snapshot), it's cheap to render every
captured row rather than try to pick which ones matter.

actual_data is a Python repr() of the query result (tuples, including
datetime.datetime(...) calls) rather than JSON, so ast.literal_eval
can't parse it as-is - datetime calls are blanked out via regex first
since only the primary-key and label columns are ever kept anyway.
Only tables classify.is_lookup_table() flags as lookup tables are
covered - not "whatever role a table happened to get assigned for one
question's retrieval" (retrieval/pipeline.py can pull a genuine lookup
table in as a bridge on an FK path, and it's still a lookup table).
Anchor tables are deliberately excluded even when they happen to have
a name/description column: they carry PII (party.email,
party.mobile_number, ...) that has no reason to be copied into every
prompt.

load_enum_like_values() below covers a different, narrower gap on
those excluded anchor tables: a short free-text column that's
genuinely just a small fixed set of values in the real data (e.g.
masters.party.nature_of_intro is only ever 'New' or 'Existing'), but
isn't normalized into its own lookup table with an FK, so it gets none
of the VALUES treatment above and the KB has nothing for it either
(same empty-enrichment gap). Found by scanning real column
cardinality, not by naming "nature_of_intro" specifically - a
per-column hardcoded list would only ever cover whatever example
someone thought to write down (see
feedback_avoid_hardcoded_prompt_examples in project memory).
"""

import json
import logging
import re
from ast import literal_eval
from pathlib import Path
from typing import Any

from sqlalchemy import text

from lms_chatbot.retrieval.classify import is_lookup_table
from lms_chatbot.sql.validator import BLOCKED_COLUMN_NAMES

logger = logging.getLogger(__name__)

# Matches datetime.datetime(...), datetime.date(...), and
# datetime.time(...) - repr() calls psycopg2/SQLAlchemy can emit for
# any of these three column types. Only datetime.datetime(...) was
# covered here originally; load_sample_values() never hit the gap
# because it only ever scans lookup tables, none of which happened to
# have a bare `date`-typed sample column - load_enum_like_values()
# below scans anchor tables too and hit it immediately (lms.tm,
# lms.training_tm, masters.party_job_code all have one).
_DATETIME_CALL = re.compile(r"datetime\.(?:datetime|date|time)\([^)]*\)")

_LABEL_COLUMN_PRIORITY = ("name", "description", "label", "title", "code")
_MAX_LABEL_COLUMNS = 2
# A lookup's description can be a paragraph; a row only has to be
# identifiable in the prompt, not reproduced in full.
_MAX_LABEL_CHARS = 120
_MAX_LIVE_LOOKUP_ROWS = 300
_MAX_IDENTIFIER_VALUES = 20000

# A real enum-like operational column (status/type/intro-nature flags,
# Y/N codes, ...) almost always holds a handful of distinct values
# regardless of table size, while genuine free text (notes, remarks,
# names, logins) accumulates many more as the table grows - this is
# what separates the two. Deliberately generous enough to catch a
# small fixed set, tight enough that a real free-text column won't
# sneak through even on a small dev snapshot.
_MAX_ENUM_DISTINCT_VALUES = 8
_ENUM_ELIGIBLE_TYPE_PREFIXES = ("varchar", "char", "text")

# Excluded by structural naming convention, not by business term (see
# module docstring on why per-term hardcoding is avoided elsewhere) -
# these are column-name *shapes* this codebase already treats
# specially, not one org's specific vocabulary:
#   - created_by/last_modified_by: a system/service account recording
#     who touched the ROW, already called out the same way in
#     agents/sql_agent.py's SQL_AGENT_SYSTEM_PROMPT - never a fixed
#     business category, however few distinct usernames a small
#     snapshot happens to show.
#   - first_name/last_name/display_name: per-person identifiers, not a
#     business enum, that only look small because a dev snapshot has
#     few people in it.
#   - comments/remarks/notes/value/message/info: free-text/narrative
#     fields that grow without bound in real usage - a 20-row snapshot
#     just hasn't accumulated enough rows yet to show that.
_ENUM_EXCLUDED_COLUMN_NAMES = {
    "created_by",
    "last_modified_by",
    "first_name",
    "last_name",
    "display_name",
    "comments",
    "remarks",
    "notes",
    "value",
    "message",
    "info",
}


def _label_column_names(column_names: list[str]) -> list[str]:
    return [c for c in _LABEL_COLUMN_PRIORITY if c in column_names][:_MAX_LABEL_COLUMNS]


def load_sample_values(
    snapshot_path: Path, kb_lookup: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """qualified_name -> list of {pk_column: value, label_column: value, ...},
    for every table present in both the snapshot and kb_lookup that has a
    known primary key and at least one label-ish column. Missing snapshot
    file, unparseable rows, or a table with no usable columns all degrade
    to "no sample values for this table" (logged, not raised) - callers
    already treat this data as an optional prompt enrichment, not a
    correctness requirement.

    Snapshot fallback only - the retriever prefers
    load_sample_values_live() whenever the database is reachable."""

    if not snapshot_path.exists():
        logger.warning(
            "No database snapshot found at %s - lookup tables will render "
            "without real sample values.",
            snapshot_path,
        )
        return {}

    result: dict[str, list[dict[str, Any]]] = {}
    for name, rows in _snapshot_tables(snapshot_path):
        kb_entry = kb_lookup.get(name.lower())
        if not kb_entry or not is_lookup_table(kb_entry):
            continue
        column_names = [c.get("column_name") for c in kb_entry.get("columns") or []]
        rendered_rows = _lookup_rows(kb_entry, column_names, rows)
        if rendered_rows:
            result[name] = rendered_rows

    return result


def _lookup_rows(kb_entry: dict[str, Any], column_names: list[str], rows) -> list[dict[str, Any]]:
    """A lookup table's pk + label columns from `rows` (each aligned to
    `column_names`) - shared by the snapshot and live loaders so both
    render identically."""
    pk_columns = [pk.get("column") for pk in kb_entry.get("primary_keys") or []]
    label_columns = _label_column_names(column_names)
    if not pk_columns or not label_columns:
        return []

    kept_columns = [c for c in pk_columns[:1] + label_columns if c in column_names]
    indices = [column_names.index(c) for c in kept_columns]
    return [
        {column_names[i]: _clip(row[i]) for i in indices}
        for row in rows
        if len(row) == len(column_names)
    ]


def _clip(value: Any) -> Any:
    if isinstance(value, str) and len(value) > _MAX_LABEL_CHARS:
        return value[:_MAX_LABEL_CHARS] + "..."
    return value


def _snapshot_tables(snapshot_path: Path):
    """(qualified_name, rows) for every snapshot table whose actual_data parses."""
    with open(snapshot_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    for schema in snapshot.get("schemas", []):
        for table in schema.get("tables", []):
            name = table.get("qualified_name")
            raw = table.get("actual_data")
            if not name or not raw:
                continue
            try:
                yield name, literal_eval(_DATETIME_CALL.sub("None", raw))
            except (ValueError, SyntaxError) as e:
                logger.warning("Could not parse sample rows for %s: %s", name, e)


def load_enum_like_values(
    snapshot_path: Path, kb_lookup: dict[str, dict[str, Any]]
) -> dict[str, dict[str, list[str]]]:
    """qualified_name -> {column_name: [distinct non-null values]}, for
    every short free-text column across ALL tables (unlike
    load_sample_values, this deliberately includes anchor tables - see
    module docstring) whose real data turns out to hold only a small
    fixed set of distinct values.

    Security note: BLOCKED_COLUMN_NAMES (sql/validator.py) is excluded
    unconditionally, regardless of how few distinct values a column
    would show - the same floor that already applies to what a query
    is allowed to return applies here too, so a credential/PII column
    can never end up rendered into a prompt just because it happened
    to look "enum-like" in this snapshot.

    Lookup tables (classify.is_lookup_table) are skipped entirely -
    load_sample_values() above already renders their real id/label rows
    in full, so scanning them again here would only re-show the same
    values through a second, differently-shaped block.

    Snapshot fallback only - see load_enum_like_values_live().
    """
    if not snapshot_path.exists():
        return {}

    result: dict[str, dict[str, list[str]]] = {}

    for name, rows in _snapshot_tables(snapshot_path):
        kb_entry = kb_lookup.get(name.lower())
        if not kb_entry or is_lookup_table(kb_entry):
            continue

        columns = kb_entry.get("columns") or []
        column_names = [c.get("column_name") for c in columns]
        column_types = {
            c.get("column_name"): (c.get("data_type") or "").lower()
            for c in columns if c.get("column_name")
        }

        table_result: dict[str, list[str]] = {}
        for column in _enum_eligible_columns(column_types, kb_entry):
            index = column_names.index(column)
            distinct = {
                row[index]
                for row in rows
                if len(row) == len(column_names) and row[index] not in (None, "")
            }
            if 2 <= len(distinct) <= _MAX_ENUM_DISTINCT_VALUES:
                table_result[column] = sorted(str(v) for v in distinct)

        if table_result:
            result[name] = table_result

    return result


def _enum_eligible_columns(column_types: dict[str, str], kb_entry: dict[str, Any] | None = None) -> list[str]:
    """Short free-text columns that could hold a small fixed value set:
    never a blocked credential/PII column, never a structurally-excluded
    name shape, never CDC bookkeeping (__op, __deleted), never a
    large-object reference column (its values are OID numbers - see
    sql/large_objects.py)."""
    large_objects = {
        c.get("column_name") for c in (kb_entry or {}).get("columns") or [] if c.get("large_object")
    }
    eligible = []
    for column, dtype in column_types.items():
        if column in large_objects:
            continue
        col_lower = column.lower()
        if col_lower in BLOCKED_COLUMN_NAMES or col_lower in _ENUM_EXCLUDED_COLUMN_NAMES:
            continue
        if col_lower.startswith("__"):
            continue
        if dtype.startswith(_ENUM_ELIGIBLE_TYPE_PREFIXES):
            eligible.append(column)
    return eligible


# ---------------------------------------------------------------------
# Live-database loaders (preferred over the snapshot)
# ---------------------------------------------------------------------
# The offline snapshot caps every table at 20 captured rows and is only
# as fresh as the last extraction run. On the real DB that silently hid
# 17 of lms.status's 37 rows (every status past row 20, e.g.
# UT_EXAM_LIMIT, could never be matched by the SQL agent however a
# question was phrased), 33 of masters.job_code's 53, and still showed
# a lms.status_category row that no longer exists. Reading the same
# values from the database at server start has neither problem, for
# every table at once, with no per-table knowledge. Both queries are
# bounded (LIMIT) so a lookup or enum column that turns out to be large
# can't blow up startup time or the prompt.

def _live_columns(conn, qualified_name: str) -> dict[str, str]:
    schema, _, table = qualified_name.partition(".")
    rows = conn.execute(
        text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = :s AND table_name = :t ORDER BY ordinal_position"
        ),
        {"s": schema, "t": table},
    ).fetchall()
    return {r[0]: r[1].lower() for r in rows}


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_table(qualified_name: str) -> str:
    return ".".join(_quote(part) for part in qualified_name.split("."))


def _table_key(kb_entry: dict[str, Any], fallback: str) -> str:
    return kb_entry.get("physical_qualified_name") or kb_entry.get("qualified_name") or fallback


def _read_only(conn) -> None:
    conn.execute(text("SET TRANSACTION READ ONLY"))


def load_sample_values_live(engine, kb_lookup: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Same output as load_sample_values(), read from the live database:
    every row (up to _MAX_LIVE_LOOKUP_ROWS) of every lookup table, using
    the table's real columns. Raises if the database can't be reached at
    all, so the caller can fall back to the snapshot."""
    result: dict[str, list[dict[str, Any]]] = {}
    with engine.connect() as conn:
        _read_only(conn)
        for name, kb_entry in kb_lookup.items():
            if not is_lookup_table(kb_entry):
                continue
            live = _live_columns(conn, name)
            pk_columns = [pk.get("column") for pk in kb_entry.get("primary_keys") or []]
            column_names = [c for c in pk_columns[:1] + _label_column_names(list(live)) if c in live]
            if len(column_names) < 2:
                continue
            try:
                rows = conn.execute(text(
                    f"SELECT {', '.join(_quote(c) for c in column_names)} "
                    f"FROM {_quote_table(name)} ORDER BY 1 LIMIT {_MAX_LIVE_LOOKUP_ROWS}"
                )).fetchall()
            except Exception as e:  # noqa: BLE001 - one bad table must not kill startup
                logger.warning("Live lookup values unavailable for %s: %s", name, e)
                conn.rollback()
                _read_only(conn)
                continue
            rendered = _lookup_rows(kb_entry, column_names, [tuple(r) for r in rows])
            if rendered:
                result[_table_key(kb_entry, name)] = rendered
    return result


def load_identifier_values_live(engine, kb_lookup: dict[str, dict[str, Any]], is_identity) -> dict[str, dict[str, list[str]]]:
    """qualified_name -> {column: [values]} for every identity table's
    identifier columns - a text column whose values are unique per row
    (a login, an employee code), found by real cardinality rather than
    by name. Used only to tell the SQL agent where a misplaced literal
    actually belongs (sql/value_check.py) - never rendered into a prompt.
    Blocked credential/PII columns are excluded as everywhere else."""
    result: dict[str, dict[str, list[str]]] = {}
    with engine.connect() as conn:
        _read_only(conn)
        for name, kb_entry in kb_lookup.items():
            if not is_identity(kb_entry):
                continue
            columns: dict[str, list[str]] = {}
            for column in _enum_eligible_columns(_live_columns(conn, name), kb_entry):
                try:
                    total, distinct = conn.execute(text(
                        f"SELECT COUNT({_quote(column)}), COUNT(DISTINCT {_quote(column)}) "
                        f"FROM {_quote_table(name)}")).first()
                    if not total or total != distinct or total > _MAX_IDENTIFIER_VALUES:
                        continue
                    columns[column] = [str(r[0]) for r in conn.execute(text(
                        f"SELECT {_quote(column)} FROM {_quote_table(name)} WHERE {_quote(column)} IS NOT NULL"
                    )).fetchall()]
                except Exception as e:  # noqa: BLE001
                    logger.warning("Identifier scan failed for %s.%s: %s", name, column, e)
                    conn.rollback()
                    _read_only(conn)
            if columns:
                result[_table_key(kb_entry, name)] = columns
    return result


def attach_parent_labels_live(engine, sample_values: dict[str, list[dict[str, Any]]], graph, kb_lookup) -> None:
    """For every FK from one lookup table to ANOTHER lookup table, add the
    parent row's label to each VALUES row, in place.

    A lookup table shared across entities only carries the information
    that tells its rows apart through such an FK: lms.status holds the
    statuses of trainings, user trainings, exams, sessions and session
    registrations all together, and which entity each one belongs to is
    only on lms.status.category_id -> lms.status_category. Without it a
    status like E_FAILED is just a name, and was observed being applied
    to the session-register table instead of lms.exam. Found for every
    lookup->lookup edge in the FK graph at once, not per table.

    The added key is spelled as the join itself
    ("category_id→status_category.name") so it can't be mistaken for a
    real column of the table - the same trap the rendered `acronym`
    column fell into."""
    with engine.connect() as conn:
        _read_only(conn)
        for name, rows in sample_values.items():
            kb_entry = kb_lookup.get(name.lower()) or {}
            pk_columns = [pk.get("column") for pk in kb_entry.get("primary_keys") or []]
            if not pk_columns or not rows or pk_columns[0] not in rows[0]:
                continue
            pk = pk_columns[0]
            for edge in graph.get_neighbors(name)["outgoing"]:
                parent = edge.target_table
                if parent not in sample_values or parent == name:
                    continue
                labels = _label_column_names(list(_live_columns(conn, parent)))
                if not labels:
                    continue
                key = f"{edge.source_column}→{parent.split('.')[-1]}.{labels[0]}"
                try:
                    mapping = dict(conn.execute(text(
                        f"SELECT c.{_quote(pk)}, p.{_quote(labels[0])} "
                        f"FROM {_quote_table(name)} c JOIN {_quote_table(parent)} p "
                        f"ON c.{_quote(edge.source_column)} = p.{_quote(edge.target_column)}"
                    )).fetchall())
                except Exception as e:  # noqa: BLE001
                    logger.warning("Parent labels unavailable for %s -> %s: %s", name, parent, e)
                    conn.rollback()
                    _read_only(conn)
                    continue
                if not mapping:
                    continue
                for row in rows:
                    row[key] = _clip(mapping.get(row.get(pk)) or "")


def load_enum_like_values_live(engine, kb_lookup: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Same output as load_enum_like_values(), read from the live
    database. Each eligible column fetches at most
    _MAX_ENUM_DISTINCT_VALUES + 1 distinct values - enough to tell a
    small fixed set apart from free text without scanning further."""
    result: dict[str, dict[str, list[str]]] = {}
    with engine.connect() as conn:
        _read_only(conn)
        for name, kb_entry in kb_lookup.items():
            if is_lookup_table(kb_entry):
                continue
            table_result: dict[str, list[str]] = {}
            for column in _enum_eligible_columns(_live_columns(conn, name), kb_entry):
                try:
                    values = [r[0] for r in conn.execute(text(
                        f"SELECT {_quote(column)} FROM {_quote_table(name)} "
                        f"WHERE {_quote(column)} IS NOT NULL AND {_quote(column)}::text <> '' "
                        f"GROUP BY 1 ORDER BY 1 LIMIT {_MAX_ENUM_DISTINCT_VALUES + 1}"
                    )).fetchall()]
                except Exception as e:  # noqa: BLE001
                    logger.warning("Live enum scan failed for %s.%s: %s", name, column, e)
                    conn.rollback()
                    _read_only(conn)
                    continue
                if 2 <= len(values) <= _MAX_ENUM_DISTINCT_VALUES:
                    table_result[column] = [str(v) for v in values]
            if table_result:
                result[_table_key(kb_entry, name)] = table_result
    return result
