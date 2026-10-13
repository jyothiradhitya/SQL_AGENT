"""
Rejects an equality filter whose literal doesn't exist in that column.

In the 1002-question bulk evaluation (experiments/bulk_eval) this was
the single largest failure class, in several shapes that all end the
same way - a valid query that returns 0 rows, which the bot then
reports as a confident "none":
  - a login matched against the wrong column:
      user_training.supervisor = 'approver1002', party.display_name = 'hra'
  - a guessed label: status.name = 'Completed' (real: 'T_COMPLETED'),
      training_category.name = 'Quick Training' (real: 'QUICK_TRAINING')
  - a name mangled into pieces: training.name = 'OQQT-002' AND <online
      filter>, when the training is literally named 'OQQT-002 ONLINE'
Nothing about such a result looks wrong from the inside.

So every `column = 'text'` / `column IN ('text', ...)` against a real
text column is checked against the LIVE database: if no row anywhere in
that column holds the value, the filter can only ever match nothing,
and the query is refused before it runs - with the column's closest
real values, and any other column where that exact value does exist
(e.g. "'approver1002' is a value of pc.sys_user.login"), so the agent
corrects from facts instead of guessing again. Live rather than a
startup snapshot, so a user or training created after startup is never
wrongly refused. PostgreSQL's `=` is exact and case-sensitive, so an
exact miss really does match nothing; LIKE/ILIKE patterns and
function-wrapped columns (lower(name) = ...) are left alone.

If the question genuinely names something that doesn't exist, the
error says so explicitly - "no such user" is then the right answer, not
a retry.
"""

import difflib
import logging

from sqlalchemy import text
from sqlglot import exp

from lms_chatbot.sql.validator import BLOCKED_COLUMN_NAMES

logger = logging.getLogger(__name__)

_TEXT_TYPES = ("text", "character varying", "character", "varchar", "char")
_MAX_CANDIDATES_SCANNED = 5000
_MAX_SUGGESTIONS = 8
_LIST_ALL_IF_AT_MOST = 40


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_table(qualified: str) -> str:
    return ".".join(_quote(p) for p in qualified.split("."))


class ValueChecker:
    """Holds the per-process caches (column types, and an index of where
    small known value sets live) - the existence checks themselves are
    always live."""

    def __init__(self, engine, table_columns: dict[str, set[str]], value_index: dict[str, set[str]] | None = None):
        self._engine = engine
        self._table_columns = {k.lower(): v for k, v in (table_columns or {}).items()}
        self._value_index = value_index or {}
        self._types: dict[str, dict[str, str]] = {}

    # -- resolution -----------------------------------------------------
    def _qualify(self, table: exp.Table) -> str | None:
        if table.db:
            q = f"{table.db}.{table.name}".lower()
            return q if q in self._table_columns else None
        matches = [q for q in self._table_columns if q.split(".")[-1] == table.name.lower()]
        return matches[0] if len(matches) == 1 else None  # ambiguous across schemas: skip

    def _column_type(self, conn, qualified: str, column: str) -> str:
        if qualified not in self._types:
            schema, _, table = qualified.partition(".")
            rows = conn.execute(text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = :s AND table_name = :t"), {"s": schema, "t": table}).fetchall()
            self._types[qualified] = {r[0].lower(): r[1].lower() for r in rows}
        return self._types[qualified].get(column.lower(), "")

    def _filters(self, tree: exp.Expression):
        alias_to_table: dict[str, str] = {}
        for table in tree.find_all(exp.Table):
            if isinstance(table.this, exp.Identifier):
                q = self._qualify(table)
                if q:
                    alias_to_table[table.alias_or_name.lower()] = q
                    alias_to_table[table.name.lower()] = q
        in_query = set(alias_to_table.values())

        def owner(column: exp.Column) -> str | None:
            if column.table:
                return alias_to_table.get(column.table.lower())
            owners = [t for t in in_query if column.name.lower() in self._table_columns.get(t, set())]
            return owners[0] if len(owners) == 1 else None

        for node in tree.find_all(exp.EQ, exp.In):
            if not _conjunctive(node):
                continue
            if isinstance(node, exp.EQ):
                pairs = [(node.this, [node.expression]), (node.expression, [node.this])]
            else:
                pairs = [(node.this, node.expressions)]
            for column, literals in pairs:
                if not isinstance(column, exp.Column):
                    continue
                table = owner(column)
                if not table:
                    continue
                for literal in literals:
                    if isinstance(literal, exp.Literal) and literal.is_string:
                        yield table, column.name, literal.this

    # -- the check ------------------------------------------------------
    def check(self, tree: exp.Expression) -> None:
        filters = list(dict.fromkeys(self._filters(tree)))
        if not filters:
            return
        problems = []
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SET TRANSACTION READ ONLY"))
                for table, column, literal in filters:
                    if not self._column_type(conn, table, column).startswith(_TEXT_TYPES):
                        continue
                    exists = conn.execute(text(
                        f"SELECT 1 FROM {_quote_table(table)} WHERE {_quote(column)} = :v LIMIT 1"),
                        {"v": literal}).first()
                    if exists:
                        continue
                    candidates = [r[0] for r in conn.execute(text(
                        f"SELECT DISTINCT {_quote(column)} FROM {_quote_table(table)} "
                        f"WHERE {_quote(column)} IS NOT NULL LIMIT {_MAX_CANDIDATES_SCANNED}")).fetchall()]
                    problems.append((table, column, literal, candidates,
                                     self._found_in_query_tables(conn, tree, table, column, literal)))
        except Exception as e:  # noqa: BLE001 - a check that can't run must never block a query
            logger.warning("[VALUE_CHECK] skipped: %s", e)
            return

        if problems:
            raise ValueError(self._message(problems))

    def _found_in_query_tables(self, conn, tree, table, column, literal) -> set[str]:
        """Other text columns, of tables already in this query, that hold
        `literal` exactly - observed: `training_schedule.name = 'OQQT-006
        ONLINE'` when that value is lms.training.name, a table the same
        query already joins. Blocked credential/PII columns are never
        probed, so this can't become an oracle for whether some email or
        token exists."""
        found = set()
        tables = {self._qualify(t) for t in tree.find_all(exp.Table) if isinstance(t.this, exp.Identifier)}
        for other in sorted(t for t in tables if t):
            self._column_type(conn, other, "")  # populate the type cache
            for col, dtype in self._types.get(other, {}).items():
                if (other, col) == (table, column.lower()) or col.startswith("__") or col in BLOCKED_COLUMN_NAMES:
                    continue
                if not dtype.startswith(_TEXT_TYPES):
                    continue
                if conn.execute(text(
                        f"SELECT 1 FROM {_quote_table(other)} WHERE {_quote(col)} = :v LIMIT 1"),
                        {"v": literal}).first():
                    found.add(f"{other}.{col}")
        return found

    def _message(self, problems) -> str:
        # The closing guidance depends on what the facts support. Offering
        # "or say it doesn't exist" when the value DOES exist in another
        # column was observed being taken anyway: told "'approver1002' is
        # a value of pc.sys_user.login", the agent still answered "no
        # trainings are assigned to approver1002".
        parts, any_elsewhere = [], False
        for table, column, literal, candidates, in_query in problems:
            cands = [str(c) for c in candidates]
            if len(cands) <= _LIST_ALL_IF_AT_MOST:
                shown, label = sorted(cands), "Its real values are"
            else:
                folded = [c for c in cands if c.lower() == literal.lower()]
                close = difflib.get_close_matches(literal, cands, n=_MAX_SUGGESTIONS, cutoff=0.4)
                contains = [c for c in cands if literal.lower() in c.lower() or c.lower() in literal.lower()]
                shown = list(dict.fromkeys(folded + close + contains))[:_MAX_SUGGESTIONS]
                label = "Its closest real values are"
            msg = f"No row of {table}.{column} equals {literal!r}, so this filter matches nothing."
            if cands:
                msg += f" {label}: " + (", ".join(repr(c) for c in shown) if shown else "(none similar)") + "."
            elsewhere = sorted((self._value_index.get(literal.lower(), set()) | in_query) - {f"{table}.{column}".lower()})
            if elsewhere:
                any_elsewhere = True
                msg += (
                    f" {literal!r} DOES exist - as a value of {', '.join(elsewhere)}: that is what the "
                    f"question refers to, so filter on that column instead, joining through to it."
                )
            parts.append(msg)
        if any_elsewhere:
            closing = (
                "Rewrite the query to filter on the column where the value really exists. Do not "
                "report that it doesn't exist - it does."
            )
        else:
            closing = (
                "If the question means one of the real values listed, rewrite the query to use it. "
                "If none of them is what the question names, it genuinely does not exist in the "
                "database: stop and say plainly that it does not exist - that is the answer."
            )
        return "Filter value not found. " + " ".join(parts) + " " + closing


_CONDITION_ROOTS = (exp.Where, exp.Having, exp.Join, exp.If)


def _conjunctive(node: exp.Expression) -> bool:
    """True when `node` is joined to its enclosing condition (WHERE,
    HAVING, JOIN ... ON, CASE WHEN) by AND only. Only then does a
    comparison that can never be true make the whole condition match
    nothing: under an OR another branch may still match, and under a NOT
    an impossible comparison makes the condition match everything."""
    parent = node.parent
    while parent is not None and not isinstance(parent, _CONDITION_ROOTS):
        if isinstance(parent, (exp.Or, exp.Not)):
            return False
        if not isinstance(parent, (exp.And, exp.Paren)):
            return False  # inside some other expression (a function argument, ...)
        parent = parent.parent
    return parent is not None


def build_value_index(sample_values: dict, enum_values: dict, identifier_values: dict) -> dict[str, set[str]]:
    """lowercased value -> {"schema.table.column", ...} over the small,
    known value sets: lookup labels, enum-like columns, and identity
    tables' identifier columns (logins). Lets a refusal say where a
    misplaced literal actually belongs. Rendering-only keys (parent
    labels like "category_id→status_category.name") are skipped."""
    index: dict[str, set[str]] = {}

    def add(table, column, value):
        if isinstance(value, str) and value and "→" not in column:
            index.setdefault(value.lower(), set()).add(f"{table}.{column}".lower())

    for table, rows in (sample_values or {}).items():
        for row in rows:
            for column, value in row.items():
                add(table, column, value)
    for source in (enum_values, identifier_values):
        for table, columns in (source or {}).items():
            for column, values in columns.items():
                for value in values:
                    add(table, column, value)
    return index
