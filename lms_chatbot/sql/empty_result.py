"""
Explains an empty result instead of letting it pass as "none".

A query that returns no rows (or a lone all-zero aggregate like
COUNT = 0) is reported by the bot as a confident "there are none" - and
in the bulk evaluation (experiments/bulk_eval) that was often wrong for
a reason invisible in the result itself: a condition or join the
question never asked for emptied it. "List all departments" ran
`JOIN product_store_type ... WHERE pst.name = 'DEPARTMENT'`: the value
is real, but 18 of 19 departments have no type set, so the join alone
removed them and the answer was "there are no departments".

So when a result is empty, the same query is re-counted with each of
its AND-ed conditions removed in turn (and the FROM table counted on
its own), and those numbers travel with the result. That's evidence,
not a verdict: a genuine zero ("oquser3 failed no exams") also shows a
non-zero count once its status filter is removed - the question asked
for that filter, so the zero stands. Deciding which it is stays with
the model (orchestrator.py's ANSWERING rules); this only supplies the
facts it can't otherwise see. Bounded to _MAX_PROBES cheap COUNT(*)
queries, only on empty results, and any failure just skips the check.
"""

import logging

from sqlalchemy import text
from sqlglot import exp

logger = logging.getLogger(__name__)

_MAX_PROBES = 6


def is_empty(result: dict) -> bool:
    if result["row_count"] == 0:
        return True
    if result["row_count"] == 1:
        values = result["rows"][0]
        return bool(values) and all(v is None or (isinstance(v, (int, float)) and v == 0) for v in values)
    return False


def _conjuncts(condition: exp.Expression | None) -> list[exp.Expression]:
    if condition is None:
        return []
    if isinstance(condition, exp.Paren):
        return _conjuncts(condition.this)
    if isinstance(condition, exp.And):
        return _conjuncts(condition.left) + _conjuncts(condition.right)
    return [condition]


def _count_query(select: exp.Select, where: list[exp.Expression]) -> str:
    probe = select.copy()
    for arg in ("group", "having", "order", "limit", "offset", "distinct"):
        probe.set(arg, None)
    probe.set("expressions", [exp.Count(this=exp.Star())])
    probe.set("where", exp.Where(this=exp.and_(*where)) if where else None)
    return probe.sql(dialect="postgres")


def diagnose(tree: exp.Expression, connection) -> list[str]:
    """Human-readable facts about why `tree` returned nothing, or [] when
    the query's shape isn't one this can safely take apart."""
    if not isinstance(tree, exp.Select) or tree.args.get("with") or not tree.args.get("from_"):
        return []
    where = tree.args.get("where")
    conditions = _conjuncts(where.this if where else None)[:_MAX_PROBES]

    lines = []
    try:
        base = tree.args["from_"].this
        if isinstance(base, exp.Table) and isinstance(base.this, exp.Identifier):
            table_sql = exp.Table(this=base.this.copy(), db=base.args.get("db")).sql(dialect="postgres")
            n = connection.execute(text(f"SELECT COUNT(*) FROM {table_sql}".replace(":", r"\:"))).scalar()
            lines.append(f"{table_sql} has {n} rows in total.")
        if tree.args.get("joins"):
            n = connection.execute(text(_count_query(tree, []).replace(":", r"\:"))).scalar()
            lines.append(f"With its JOINs but no WHERE conditions, this query matches {n} rows.")
        for i, condition in enumerate(conditions):
            rest = conditions[:i] + conditions[i + 1:]
            n = connection.execute(text(_count_query(tree, rest).replace(":", r"\:"))).scalar()
            lines.append(f"Without the condition `{condition.sql(dialect='postgres')}` it matches {n} rows.")
    except Exception as e:  # noqa: BLE001 - diagnosis is best-effort, never fails the query
        logger.warning("[EMPTY_RESULT] diagnosis skipped: %s", e)
        return []
    return lines
