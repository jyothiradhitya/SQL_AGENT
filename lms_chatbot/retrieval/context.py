"""
Condensed per-table prompt rendering.

The old pipeline injected the full semantic_kb.json blob (or a full
information_schema dump) per table - long English sentences for every
field, repeated for every table. This renders one compact block per
table instead, roughly 5-10x fewer tokens each, so the SQL agent's
per-question table budget (see retrieval.pipeline) buys more tables
for the same prompt size.

JOIN lines come from the deterministic fk_graph (retrieval/fk_graph.py,
built offline by retrieval/graph_builder.py), not from re-parsing each
table's raw foreign_keys[] here. That KB field is frequently wrong
(references_table recorded as "public.<table>" regardless of the
target's real schema) and sometimes genuinely ambiguous (the same bare
name exists in more than one schema) - graph_builder.py already did
the work of resolving what's resolvable and dropping what isn't, so
this only ever renders a JOIN line for an edge that's actually real.

VALUES lines (lookup tables only) come from retrieval/sample_values.py -
real id/label rows captured from the DB, since semantic_kb.json's
per-column business_description/example_values fields exist in its
schema but were never actually populated for any column in this KB.

COLUMN VALUES lines are the same idea for a non-lookup table's own
short free-text columns that turn out to hold only a small fixed set
of real values (retrieval/sample_values.py's load_enum_like_values,
e.g. masters.party.nature_of_intro is only ever 'New'/'Existing') -
mutually exclusive with the VALUES block above for any given table,
since load_enum_like_values skips genuine lookup tables entirely.
"""

import re
from typing import Any

from lms_chatbot.retrieval.fk_graph import FKGraph

_ACRONYM_STOPWORDS = {"and", "of", "the", "for", "in", "on", "to", "a", "an", "&"}


def _acronym(label: str) -> str | None:
    """A short, deterministic "initials" form of a label, so a user's
    shorthand ("QA") can match a real VALUES row's text ("Quality
    Assurance") the same way any other column value already can - by
    literal comparison - instead of asking the model to recognize the
    abbreviation from its own training data. Computed once per row
    here rather than hand-written per term in a prompt (see git
    history on sql_agent.py's SQL_AGENT_SYSTEM_PROMPT for the version
    that tried that - it only ever covered whatever examples someone
    thought to write down), so it covers whatever labels this org's
    actual data has, for any lookup table, automatically.

    None for single-word labels ("Production") - a one-letter acronym
    isn't a meaningful shorthand, and a single word is already covered
    by ordinary substring matching on the label itself.
    """
    words = [w for w in re.split(r"[\s\-/]+", label) if w]
    significant = [w for w in words if w.lower() not in _ACRONYM_STOPWORDS]
    if len(significant) < 2:
        return None

    # A word that's already a multi-letter all-caps token ("OQ" in
    # "OQ-Department") IS the informal shorthand for this label already -
    # reducing it to a single initial ("O") would destroy the very term
    # a user is likely to type. Prefer it standalone over combining
    # initials across every word in the label.
    existing = next(
        (w for w in significant if w.isalpha() and w.isupper() and len(w) >= 2), None
    )
    if existing:
        return existing

    letters = "".join(w[0] for w in significant if w[0].isalpha())
    return letters.upper() if len(letters) >= 2 else None


def _columns_line(table: dict[str, Any]) -> str:
    # Every column is prefixed with its own table's bare name
    # (bare_name.column, not the full alias syntax) even though it's
    # already grouped under this table's own heading - observed
    # failure mode otherwise: scanning several tables' COLUMNS lines
    # in the same prompt, the model picks up a column that belongs to
    # a DIFFERENT table just above/below (e.g. wrote party.login when
    # only sys_user has a login column) because ownership was implied
    # only by position, not stated on the column itself.
    bare_name = (table.get("table_name") or "").strip()
    parts = []

    for column in table.get("columns") or []:
        name = column.get("column_name")
        if not name:
            continue

        dtype = column.get("data_type", "")
        # Large-object reference columns (sql/large_objects.py): the DB
        # holds only an OID; the real content is filled into results.
        if column.get("large_object") == "text":
            dtype += " [stored text: select it plainly, real text is filled in; cannot filter/search/sort on it]"
        elif column.get("large_object") == "file":
            dtype += " [attached file reference, not text]"
        qualified = f"{bare_name}.{name}" if bare_name else name
        parts.append(f"{qualified} {dtype}".strip())

    return ", ".join(parts)


def _joins_lines(table_ref: str, graph: FKGraph) -> list[str]:
    return [
        f"  {edge.source_column} -> {edge.target_table}.{edge.target_column}"
        for edge in graph.get_neighbors(table_ref)["outgoing"]
    ]


def _values_lines(name: str, sample_values: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Real id/label rows for a lookup table (retrieval/sample_values.py) -
    e.g. what "active" actually resolves to. sample_values only ever
    contains genuine lookup tables (see that module), regardless of the
    role - anchor/bridge/lookup - a table was selected under for this
    particular question.

    An "acronym" column is appended when at least one row's label
    yields one (see _acronym) - purely a rendering-time addition, not
    a real database column, so a user's shorthand term still only ever
    has to be matched against literal text actually shown here."""
    rows = sample_values.get(name)
    if not rows:
        return []

    header = list(rows[0].keys())
    label_key = next((k for k in ("name", "title", "label") if k in header), None)

    acronyms = [_acronym(str(row.get(label_key, ""))) for row in rows] if label_key else []
    if not any(acronyms):
        acronyms = []

    display_header = header + ["acronym"] if acronyms else header
    lines = [f"VALUES ({' | '.join(display_header)}):"]

    for i, row in enumerate(rows):
        values = [str(row.get(col, "")) for col in header]
        if acronyms:
            values.append(acronyms[i] or "")
        lines.append("  " + " | ".join(values))

    return lines


def _enum_values_lines(name: str, enum_values: dict[str, dict[str, list[str]]]) -> list[str]:
    """Real distinct values for a short free-text column that isn't a
    genuine lookup table but still only takes a handful of values in
    practice (retrieval/sample_values.py's load_enum_like_values) -
    e.g. what masters.party.nature_of_intro actually contains, so the
    SQL agent doesn't have to guess a filter value it was never shown."""
    columns = enum_values.get(name)
    if not columns:
        return []

    lines = ["COLUMN VALUES:"]
    for column, values in columns.items():
        lines.append(f"  {column}: {', '.join(repr(v) for v in values)}")
    return lines


def render_table(
    table: dict[str, Any],
    role: str,
    graph: FKGraph,
    sample_values: dict[str, list[dict[str, Any]]] | None = None,
    enum_values: dict[str, dict[str, list[str]]] | None = None,
) -> str:
    name = table.get("physical_qualified_name") or table.get("qualified_name")

    purpose = table.get("business_meaning") or table.get("business_purpose") or ""

    lines = [f"TABLE {name} ({role})"]

    if purpose:
        lines.append(f"PURPOSE: {purpose}")

    columns_line = _columns_line(table)
    if columns_line:
        lines.append(f"COLUMNS: {columns_line}")

    if name and sample_values:
        lines.extend(_values_lines(name, sample_values))

    if name and enum_values:
        lines.extend(_enum_values_lines(name, enum_values))

    join_lines = _joins_lines(name, graph) if name else []
    if join_lines:
        lines.append("JOINS:")
        lines.extend(join_lines)

    return "\n".join(lines)


def render_context(
    tables_with_roles: list[tuple[dict[str, Any], str]],
    graph: FKGraph,
    sample_values: dict[str, list[dict[str, Any]]] | None = None,
    enum_values: dict[str, dict[str, list[str]]] | None = None,
) -> str:
    return "\n\n".join(
        render_table(table, role, graph, sample_values, enum_values)
        for table, role in tables_with_roles
    )
