"""
Deterministic edge-case probes for lms_chatbot/sql/validator.py and
sql/executor.py - no LLM involved. Each case is a SQL string the SQL
agent *could* emit (or be manipulated into emitting) plus what SHOULD
happen. A row marked FAIL means the code's behaviour disagrees with the
expectation, i.e. a real bug/hole.

Run:  uv run python experiments/edge_case_validator_tests.py
"""

import sqlglot

from lms_chatbot.retrieval.pipeline import SchemaRetriever
from lms_chatbot.sql.executor import execute_sql
from lms_chatbot.sql.validator import SQL_DIALECT, apply_row_limit, validate_sql

_R = SchemaRetriever()
ALLOWED = _R.all_table_names
COLS = _R.table_columns
from lms_chatbot.sql.executor import _engine  # noqa: E402
from lms_chatbot.sql.value_check import ValueChecker, build_value_index  # noqa: E402
CHECKER = ValueChecker(_engine, COLS, build_value_index(_R.sample_values, _R.enum_values, _R.identifier_values))

# (id, description, sql, expect)  expect: "block" | "allow" | "run"
#   block = validator must reject
#   allow = validator must accept (legitimate query)
#   run   = must validate AND execute successfully against the real DB
CASES = [
    # --- sensitive-data bypasses: blocked columns never named directly ---
    ("S1", "whole-row reference returns every column incl. password_hash",
     "SELECT u FROM pc.sys_user u", "block"),
    ("S2", "row_to_json(alias) dumps every column",
     "SELECT row_to_json(u) FROM pc.sys_user u", "block"),
    ("S3", "to_jsonb(alias.*) - star hidden inside a function arg",
     "SELECT to_jsonb(p.*) FROM masters.party p", "block"),
    ("S4", "json_agg(alias) aggregates every column",
     "SELECT json_agg(p) FROM masters.party p", "block"),
    ("S5", "query_to_xml runs an arbitrary SQL *string* - validator never parses it",
     "SELECT query_to_xml('select password_hash from pc.sys_user', true, false, '')", "block"),
    ("S6", "table_to_xml dumps a whole table by name",
     "SELECT table_to_xml('pc.sys_user', true, false, '')", "block"),
    ("S7", "pg_sleep_for (sleep variant not in blocklist)",
     "SELECT pg_sleep_for('5 seconds')", "block"),
    ("S8", "schema-qualified blocked function",
     "SELECT pg_catalog.pg_sleep(1)", "block"),
    ("S9", "pg_ls_logdir / pg_ls_waldir (fs listing, not in blocklist)",
     "SELECT pg_ls_waldir()", "block"),
    ("S10", "information_schema probing",
     "SELECT column_name FROM information_schema.columns", "block"),
    ("S11", "blocked column via quoted identifier",
     'SELECT "EMAIL" FROM masters.party', "block"),
    ("S12", "direct email (baseline - should already block)",
     "SELECT email FROM masters.party", "block"),
    ("S13", "sys_user.image_url via subquery alias",
     "SELECT x.image_url FROM (SELECT image_url FROM pc.sys_user) x", "block"),
    ("S14", "hstore/record-to-text cast of whole row",
     "SELECT u::text FROM pc.sys_user u", "block"),
    ("S15", "current_user / version() info disclosure",
     "SELECT version(), current_user", "block"),
    ("S16", "writable CTE (baseline - should already block)",
     "WITH d AS (DELETE FROM lms.exam RETURNING id) SELECT id FROM d", "block"),
    ("S17", "two statements (baseline)",
     "SELECT 1; SELECT 2", "block"),
    ("S18", "SELECT INTO creates a table",
     "SELECT id INTO TEMP t FROM lms.exam", "block"),
    ("S19", "FOR UPDATE takes row locks on a 'read-only' query",
     "SELECT id FROM lms.exam FOR UPDATE", "block"),

    # --- legitimate queries that must NOT be rejected ---
    ("L1", "plain CTE (CTE name is not a real table)",
     "WITH t AS (SELECT id FROM lms.exam) SELECT COUNT(*) FROM t", "run"),
    ("L2", "recursive CTE",
     "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<3) SELECT i FROM n", "run"),
    ("L3", "COUNT(*) (star inside aggregate)",
     "SELECT COUNT(*) FROM masters.party", "run"),
    ("L4", "string literal containing ' :word' (SQLAlchemy text() bind-param trap)",
     "SELECT COUNT(*) FROM masters.party WHERE first_name <> 'Note :draft'", "run"),
    ("L5", "FETCH FIRST n ROWS ONLY (+ executor adds LIMIT)",
     "SELECT id FROM lms.exam ORDER BY id FETCH FIRST 5 ROWS ONLY", "run"),
    ("L6", "UNION with ORDER BY/LIMIT",
     "SELECT name FROM lms.status UNION SELECT name FROM masters.status ORDER BY 1 LIMIT 3", "run"),
    ("L7", "generate_series in FROM (set-returning function, not a table)",
     "SELECT d FROM generate_series(1,3) AS d", "run"),
    ("L8", "unnest in FROM",
     "SELECT x FROM unnest(ARRAY[1,2]) AS x", "run"),
    ("L9", "lateral join to a subquery",
     "SELECT p.id, s.c FROM masters.party p CROSS JOIN LATERAL (SELECT COUNT(*) c FROM lms.user_training ut WHERE ut.party_id = p.id) s", "run"),
    ("L10", "column literally named like a blocked one but different (email_verified-ish) - n/a check: 'ip' substring",
     "SELECT name AS ship_name FROM lms.status", "run"),
    ("L11", "product_store.image_url (store logo - explicitly allowed)",
     "SELECT image_url FROM masters.product_store", "run"),
    ("L12", "bare 'image_url' in a join where sys_user is NOT the owner",
     "SELECT ps.name, ps.image_url FROM masters.product_store ps JOIN masters.product_store_user_group g ON g.product_store_id = ps.id", "run"),
    ("L13", "EXISTS subquery",
     "SELECT COUNT(*) FROM masters.party p WHERE EXISTS (SELECT 1 FROM lms.user_training ut WHERE ut.party_id = p.id)", "run"),
    ("L14", "window function",
     "SELECT id, ROW_NUMBER() OVER (ORDER BY id) FROM lms.exam", "run"),
    ("L15", "markdown-fenced SQL",
     "```sql\nSELECT COUNT(*) FROM lms.exam\n```", "run"),
    ("L16", "trailing semicolon",
     "SELECT COUNT(*) FROM lms.exam;", "run"),
    ("L17", "BIGINT epoch conversion recommended by the prompt",
     "SELECT COUNT(*) FROM masters.party WHERE to_timestamp(created_date/1000000.0) >= now() - interval '10 years'", "run"),

    # --- regressions for the fixes (real column sharing a table alias's name) ---
    ("R1", "masters.job_code.status is a real column while masters.status is joined",
     "SELECT job_code.name, status FROM masters.job_code JOIN masters.status ON masters.status.id = job_code.status_id", "allow"),
    ("R2", "lms.tm_job_code.job_code is a real column while masters.job_code is joined",
     "SELECT job_code FROM lms.tm_job_code JOIN masters.job_code ON masters.job_code.id = lms.tm_job_code.job_code_id", "allow"),
    ("R3", "count(alias.*) still refused (expands columns)",
     "SELECT COUNT(p.*) FROM masters.party p", "block"),
    ("R4", "query_to_xml hidden in FROM clause",
     "SELECT x FROM query_to_xml('select 1', true, false, '') AS x", "block"),
    ("R5", "now()/CURRENT_DATE stay allowed",
     "SELECT CURRENT_DATE, now()", "run"),
    ("R6", "string literal with colons + LIKE percent",
     "SELECT COUNT(*) FROM masters.party WHERE first_name LIKE '%OQ%' OR first_name = 'a :b 12:30'", "run"),

    # --- guessed filter values (real SQL the bot produced in the bulk baseline) ---
    ("V1", "real-but-sparse label ('DEPARTMENT' exists; 18/19 rows have NULL type_id) - valid SQL, data gap",
     "SELECT COUNT(*) FROM masters.product_store ps JOIN masters.product_store_type pst ON ps.type_id = pst.id WHERE pst.name = 'DEPARTMENT'", "run"),
    ("V11", "masters.status value used against lms.status (same bare name, other schema)",
     "SELECT COUNT(*) FROM lms.exam e JOIN lms.status s ON s.id = e.status_id WHERE s.name = 'ACTIVE_INITIATED'", "block"),
    ("V12", "masters.status value against masters.status (must pass)",
     "SELECT COUNT(*) FROM masters.party p JOIN masters.status s ON s.id = p.status_id WHERE s.name = 'PARTY_ACTIVE'", "run"),
    ("V2", "guessed party_type label ('0 employees')",
     "SELECT COUNT(*) FROM masters.party p JOIN masters.party_type pt ON p.party_type_id = pt.id WHERE pt.name = 'Employee'", "block"),
    ("V3", "wrong-case category label (= is case-sensitive, matches nothing)",
     "SELECT COUNT(*) FROM lms.training t JOIN lms.training_category c ON c.id = t.category_id WHERE c.name = 'Induction'", "block"),
    ("V4", "IN list with one guessed status",
     "SELECT COUNT(*) FROM lms.exam e JOIN lms.status s ON s.id = e.status_id WHERE s.name IN ('E_PASSED', 'PASSED')", "block"),
    ("V5", "guessed enum-like column value",
     "SELECT COUNT(*) FROM masters.party WHERE nature_of_intro = 'Old'", "block"),
    ("V6", "real status label (must pass)",
     "SELECT COUNT(*) FROM lms.user_training ut JOIN lms.status s ON s.id = ut.status_id WHERE s.name = 'UT_COMPLETED'", "run"),
    ("V7", "real department + real IN list (must pass)",
     "SELECT COUNT(*) FROM masters.party p JOIN masters.product_store ps ON ps.id = p.department_id WHERE ps.name = 'Quality Assurance' AND p.nature_of_intro IN ('New','Existing')", "run"),
    ("V8", "status that only exists past the old 20-row snapshot cap (must pass)",
     "SELECT COUNT(*) FROM lms.user_training ut JOIN lms.status s ON s.id = ut.status_id WHERE s.name = 'UT_EXAM_LIMIT'", "run"),
    ("V9", "ILIKE pattern is left alone",
     "SELECT COUNT(*) FROM masters.product_store WHERE name ILIKE '%quality%'", "run"),
    ("V10", "login that doesn't exist -> refused with 'does not exist' guidance",
     "SELECT COUNT(*) FROM pc.sys_user WHERE login = 'nosuchuser'", "block"),
    ("V13", "login matched against the wrong column (baseline Q0213)",
     "SELECT COUNT(*) AS training_count FROM lms.user_training WHERE supervisor = 'approver1002'", "block"),
    ("V14", "training name mangled into pieces (baseline Q0624)",
     "SELECT COUNT(DISTINCT ut.party_id) FROM lms.user_training ut JOIN lms.training t ON ut.training_id = t.id WHERE t.name = 'OQQT-002'", "block"),
    ("V15", "login in display_name (baseline Q0240)",
     "SELECT COUNT(*) FROM masters.party_job_code pj JOIN masters.party p ON pj.party_id = p.id WHERE p.display_name = 'hra'", "block"),
    ("V16", "real login via the identity join (must pass)",
     "SELECT COUNT(*) FROM lms.user_training ut JOIN masters.party p ON ut.party_id = p.id JOIN pc.sys_user u ON p.user_id = u.id WHERE u.login = 'approver1002'", "run"),
    ("V17", "real full training name (must pass)",
     "SELECT COUNT(*) FROM lms.user_training ut JOIN lms.training t ON ut.training_id = t.id WHERE t.name = 'OQQT-002 ONLINE'", "run"),
    ("V19", "impossible value under NOT is left alone (it matches everything)",
     "SELECT COUNT(*) FROM pc.sys_user WHERE NOT (login = 'nosuchuser')", "run"),
    ("V20", "guessed status inside CASE WHEN (silently counts 0) is refused",
     "SELECT SUM(CASE WHEN s.name = 'Failed' THEN 1 ELSE 0 END) FROM lms.exam e JOIN lms.status s ON s.id = e.status_id", "block"),
    ("V21", "guessed value in a JOIN ... ON condition is refused",
     "SELECT COUNT(*) FROM lms.exam e JOIN lms.status s ON s.id = e.status_id AND s.name = 'Passed'", "block"),
    ("J1", "disconnected join -> cartesian product (baseline Q0493, '57 completed')",
     "SELECT COUNT(*) FROM lms.user_training ut JOIN pc.sys_user su ON su.login = 'oquser8' JOIN masters.party p ON p.user_id = su.id JOIN lms.status s ON s.id = ut.status_id WHERE s.name = 'UT_COMPLETED'", "block"),
    ("J2", "correctly linked 4-table join (must pass)",
     "SELECT COUNT(*) FROM lms.user_training ut JOIN masters.party p ON ut.party_id = p.id JOIN pc.sys_user su ON p.user_id = su.id JOIN lms.status s ON s.id = ut.status_id WHERE su.login = 'oquser8' AND s.name = 'UT_COMPLETED'", "run"),
    ("J3", "link supplied in WHERE instead of ON (must pass)",
     "SELECT COUNT(*) FROM lms.user_training ut JOIN pc.sys_user su ON su.login = 'oquser8' JOIN masters.party p ON p.user_id = su.id WHERE ut.party_id = p.id", "run"),
    ("J4", "explicit CROSS JOIN is intentional (must pass)",
     "SELECT COUNT(*) FROM lms.exam_type et CROSS JOIN lms.training_type tt", "run"),
    ("J5", "correlated EXISTS does not confuse the check (must pass)",
     "SELECT COUNT(*) FROM masters.party p JOIN pc.sys_user u ON p.user_id = u.id WHERE EXISTS (SELECT 1 FROM lms.exam e WHERE e.party_id = p.id)", "run"),
    ("J6", "join to a subquery, linked (must pass)",
     "SELECT p.id, x.n FROM masters.party p JOIN (SELECT party_id, COUNT(*) n FROM lms.exam GROUP BY party_id) x ON x.party_id = p.id", "run"),
    ("V22", "value filtered on the wrong table's name, right table already joined (Q0653)",
     "SELECT tt.name FROM lms.training_schedule ts JOIN lms.training t ON ts.training_id = t.id JOIN lms.training_type tt ON t.type_id = tt.id WHERE ts.name = 'OQQT-006 ONLINE'", "block"),
    ("V18", "non-text column compared to a string is left alone",
     "SELECT COUNT(*) FROM lms.user_training WHERE status_id = '5'", "run"),
]

# Exact row counts expected from "run" cases, where the count itself is the bug.
EXPECTED_ROWS = {"L5": 5}


def check(case):
    cid, desc, sql, expect = case
    try:
        validated_sql = validate_sql(sql, allowed_tables=ALLOWED, table_columns=COLS)
        CHECKER.check(sqlglot.parse_one(validated_sql, read=SQL_DIALECT))
        validated = True
        verr = None
    except Exception as e:
        validated = False
        verr = str(e).splitlines()[0][:140]

    if expect == "block":
        ok = not validated
        detail = "blocked: " + verr if not validated else "PASSED VALIDATION (hole)"
        return ok, detail

    if not validated:
        return False, "wrongly rejected: " + verr

    if expect == "allow":
        return True, "accepted"

    try:
        res = execute_sql(sql, allowed_tables=ALLOWED, table_columns=COLS, value_checker=CHECKER)
        want = EXPECTED_ROWS.get(cid)
        if want is not None and res["row_count"] != want:
            return False, f"ran but returned {res['row_count']} rows, expected {want}"
        return True, f"ran, {res['row_count']} rows"
    except Exception as e:
        return False, "validated but EXECUTION failed: " + str(e).splitlines()[0][:160]


def row_cap_probe():
    # The executor silently caps every result at DEFAULT_ROW_LIMIT, and the
    # formatted result gives the model no signal that it was truncated.
    from lms_chatbot.sql.formatting import format_query_result
    over = execute_sql("SELECT id FROM lms.user_training", allowed_tables=ALLOWED, table_columns=COLS)
    exact = execute_sql("SELECT id FROM lms.user_training LIMIT 100", allowed_tables=ALLOWED, table_columns=COLS)
    return (over["row_count"], "TRUNCATED" in format_query_result(over),
            exact["row_count"], "TRUNCATED" in format_query_result(exact))


def limit_rewrite_probe():
    out = []
    for sql in ["SELECT id FROM lms.exam LIMIT ALL",
                "SELECT id FROM lms.exam LIMIT 5 OFFSET 2",
                "SELECT id FROM lms.exam FETCH FIRST 5 ROWS ONLY"]:
        t = apply_row_limit(sqlglot.parse_one(sql, read=SQL_DIALECT), 100)
        out.append((sql, t.sql(dialect=SQL_DIALECT)))
    return out


if __name__ == "__main__":
    fails = 0
    for case in CASES:
        ok, detail = check(case)
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {case[0]:<4} {case[1][:70]:<70} -> {detail}")
    print()
    print(f"{fails} / {len(CASES)} cases FAILED")
    print()
    n1, t1, n2, t2 = row_cap_probe()
    print(f"Row-cap probe (lms.user_training = 145 rows): unfiltered -> {n1} rows, TRUNCATED marker={t1}"
          f" {'PASS' if t1 else 'FAIL'};  LIMIT 100 -> {n2} rows, marker={t2} {'PASS' if not t2 else 'FAIL'}")
    print()
    for before, after in limit_rewrite_probe():
        print(f"LIMIT rewrite: {before!r}\n            -> {after!r}")
