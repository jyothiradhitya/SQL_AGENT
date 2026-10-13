"""
Generates a large, auto-gradable question bank for lms_chatbot from the
REAL analytical_db contents: every question is paired with ground truth
computed by a hand-written reference SQL, so a bot answer can be graded
without a human reading it.

Grading modes (see grade.py):
  num    - every value in `expect` (list of numbers) must appear in the answer;
           an `any_of` list of alternative expectations may be given instead
  zero   - answer must convey "none/0" (no rows match)
  names  - answer must mention >= 80% of `expect` (list of strings)
  text   - answer must mention at least one string in `expect`
  nodb   - non-database question: answer must be non-empty and not an error

Run:  PYTHONPATH=. uv run python experiments/bulk_eval/generate_questions.py
Writes experiments/bulk_eval/questions.json
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from sqlalchemy import create_engine, text

from lms_chatbot.config import settings

OUT = Path(__file__).with_name("questions.json")
engine = create_engine(settings.analytical_db_url)
conn = engine.connect()


def q(sql, **params):
    return conn.execute(text(sql), params).fetchall()


def scalar(sql, **params):
    return q(sql, **params)[0][0]


def words(code: str) -> str:
    """UT_IN_PROGRESS -> 'in progress' (test phrasing only)."""
    parts = code.lower().split("_")
    if len(parts) > 1 and len(parts[0]) <= 2:
        parts = parts[1:]
    return " ".join(parts)


def typo(s: str) -> str:
    """Deterministic realistic typo: swap two adjacent letters in the longest word."""
    ws = s.split(" ")
    i = max(range(len(ws)), key=lambda k: len(ws[k]))
    w = ws[i]
    if len(w) < 5:
        return s
    j = len(w) // 2
    ws[i] = w[:j - 1] + w[j] + w[j - 1] + w[j + 1:]
    return " ".join(ws)


QUESTIONS = []
_seen = set()


def add(family, question, mode, expect=None, **extra):
    key = question.strip().lower()
    if key in _seen:
        return
    _seen.add(key)
    QUESTIONS.append({
        "id": f"Q{len(QUESTIONS) + 1:04d}",
        "family": family,
        "question": question,
        "mode": mode,
        "expect": expect,
        **extra,
    })


def num_or_zero(family, question, n, **extra):
    if n in (0, None):
        add(family, question, "zero", **extra)
    else:
        add(family, question, "num", [float(n)], **extra)


# ---------------------------------------------------------------------
# 1. Global counts per entity (several phrasings each)
# ---------------------------------------------------------------------
GLOBAL = [
    ("masters.party", ["How many users are there?", "How many employees are registered?", "Total number of people in the system?", "count all users"], None),
    ("lms.training", ["How many trainings are there?", "Total number of trainings", "how many training programs exist", "count of trainings defined in the LMS"], None),
    ("lms.user_training", ["How many user training records are there?", "How many training assignments exist in total?", "total trainings assigned to users"], None),
    ("lms.exam", ["How many exams are there?", "Total number of exams taken", "how many exam attempts exist"], None),
    ("lms.training_session", ["How many training sessions are there?", "total number of sessions conducted", "count training sessions"], None),
    ("lms.training_session_register", ["How many session registrations are there?", "How many trainees registered for sessions in total?"], None),
    ("lms.training_category", ["How many training categories are there?", "number of training categories"], None),
    ("lms.training_type", ["How many training types exist?", "number of training types"], None),
    ("masters.product_store", ["How many departments are there?", "total number of departments", "how many departments does the organization have"], None),
    ("masters.job_code", ["How many job codes are there?", "total job codes defined"], None),
    ("masters.request", ["How many requests are there?", "total number of requests raised"], None),
    ("masters.security_group", ["How many security groups are there?", "number of user groups for security"], None),
    ("masters.permission", ["How many permissions exist?", "total number of permissions in the system"], None),
    ("lms.tm", ["How many training materials are there?", "count of training modules (TM)"], None),
    ("lms.time_logs", ["How many time log entries are there?"], None),
    ("masters.leave", ["How many leave records are there?"], None),
    ("pc.sys_user", ["How many login accounts are there?", "how many system user accounts exist"], None),
    ("pc.sys_authority", ["How many authorities/roles are defined?"], None),
    ("lms.exam_type", ["How many exam types are there?"], None),
    ("masters.request_type", ["How many request types are there?"], None),
]
for table, phrasings, _ in GLOBAL:
    n = scalar(f"SELECT COUNT(*) FROM {table}")
    for p in phrasings:
        extra = {}
        if table == "masters.party":
            # "users" is genuinely ambiguous here: 25 parties vs 26 logins.
            extra = {"any_of": [[25.0], [26.0]]}
        num_or_zero("global_count", p, n, **extra)

# ---------------------------------------------------------------------
# 2. Lookup-value listings
# ---------------------------------------------------------------------
LISTS = [
    ("SELECT name FROM lms.training_category", ["List all training categories", "what training categories exist?", "show me the training categories"]),
    ("SELECT name FROM lms.training_type", ["List all training types", "what are the different training types?"]),
    ("SELECT name FROM masters.product_store", ["List all departments", "what departments are there?", "show department names"]),
    ("SELECT name FROM lms.exam_type", ["List the exam types"]),
    ("SELECT name FROM masters.request_type", ["List all request types"]),
    ("SELECT name FROM masters.mod", ["List all modules", "which modules exist in the system?"]),
    ("SELECT name FROM masters.security_group", ["List all security groups"]),
    ("SELECT DISTINCT name FROM lms.training", ["List all trainings", "show the names of every training"]),
    ("SELECT login FROM pc.sys_user", ["List all user logins", "show all usernames"]),
]
for sql, phrasings in LISTS:
    names = sorted({r[0] for r in q(sql) if r[0]})
    for p in phrasings:
        add("list_lookup", p, "names", names)

# ---------------------------------------------------------------------
# 3. Per-department questions (all 19 departments, incl. empty ones)
# ---------------------------------------------------------------------
depts = q("SELECT id, name FROM masters.product_store ORDER BY id")
for dept_id, dept in depts:
    users = scalar("SELECT COUNT(*) FROM masters.party WHERE department_id = :d", d=dept_id)
    jobs = scalar("SELECT COUNT(*) FROM masters.job_code WHERE dept_id = :d", d=dept_id)
    names = [f"{a} {b}".strip() for a, b in q(
        "SELECT first_name, COALESCE(last_name,'') FROM masters.party WHERE department_id = :d", d=dept_id)]
    variants = [dept, dept.lower()]
    if typo(dept) != dept:
        variants.append(typo(dept))
    for v in variants:
        num_or_zero("dept_user_count", f"How many users are in the {v} department?", users, subject=dept)
        num_or_zero("dept_job_codes", f"How many job codes does {v} have?", jobs, subject=dept)
    num_or_zero("dept_user_count", f"number of employees in {dept}", users, subject=dept)
    if names and len(names) <= 20:
        add("dept_user_list", f"List the users in the {dept} department", "names",
            sorted({n.split()[0] for n in names}), subject=dept)
    ut_completed = scalar(
        """SELECT COUNT(*) FROM lms.user_training ut JOIN masters.party p ON p.id = ut.party_id
           JOIN lms.status s ON s.id = ut.status_id
           WHERE p.department_id = :d AND s.name = 'UT_COMPLETED'""", d=dept_id)
    num_or_zero("dept_completed_trainings",
                f"How many trainings have users in {dept} completed?", ut_completed, subject=dept)

# ---------------------------------------------------------------------
# 4. Per-login questions (all 26 logins)
# ---------------------------------------------------------------------
logins = q("SELECT u.id, u.login, p.id, p.first_name, ps.name FROM pc.sys_user u "
           "LEFT JOIN masters.party p ON p.user_id = u.id "
           "LEFT JOIN masters.product_store ps ON ps.id = p.department_id ORDER BY u.login")


def ut_count(party_id, status=None):
    if party_id is None:
        return 0
    if status:
        return scalar("""SELECT COUNT(*) FROM lms.user_training ut JOIN lms.status s ON s.id = ut.status_id
                         WHERE ut.party_id = :p AND s.name = :s""", p=party_id, s=status)
    return scalar("SELECT COUNT(*) FROM lms.user_training WHERE party_id = :p", p=party_id)


def exam_count(party_id, status=None):
    if party_id is None:
        return 0
    if status:
        return scalar("""SELECT COUNT(*) FROM lms.exam x JOIN lms.status s ON s.id = x.status_id
                         WHERE x.party_id = :p AND s.name = :s""", p=party_id, s=status)
    return scalar("SELECT COUNT(*) FROM lms.exam WHERE party_id = :p", p=party_id)


for uid, login, pid, first, dept in logins:
    total = ut_count(pid)
    comp = ut_count(pid, "UT_COMPLETED")
    comp_any = comp + ut_count(pid, "UT_TM_COMPLETED")
    num_or_zero("login_trainings", f"How many trainings are assigned to {login}?", total, subject=login)
    num_or_zero("login_trainings", f"how many trainings does user {login} have", total, subject=login)
    extra = {"any_of": [[float(comp)], [float(comp_any)]]} if comp != comp_any else {}
    num_or_zero("login_completed", f"How many trainings has {login} completed?", comp, subject=login, **extra)
    num_or_zero("login_completed", f"completed trainings count for {login}", comp, subject=login, **extra)
    num_or_zero("login_failed", f"How many trainings did {login} fail?", ut_count(pid, "UT_FAILED"), subject=login)
    num_or_zero("login_cancelled", f"How many of {login}'s trainings were cancelled?", ut_count(pid, "UT_CANCELLED"), subject=login)
    num_or_zero("login_pending", f"How many trainings are still pending (not started) for {login}?", ut_count(pid, "UT_CREATED"), subject=login)
    num_or_zero("login_exams", f"How many exams has {login} taken?", exam_count(pid), subject=login)
    num_or_zero("login_exams_passed", f"How many exams did {login} pass?", exam_count(pid, "E_PASSED"), subject=login)
    num_or_zero("login_exams_failed", f"How many exams did {login} fail?", exam_count(pid, "E_FAILED"), subject=login)
    if pid is not None:
        regs = scalar("SELECT COUNT(*) FROM lms.training_session_register WHERE trainee_id = :p", p=pid)
        num_or_zero("login_sessions", f"How many training sessions is {login} registered for?", regs, subject=login)
        jc = scalar("SELECT COUNT(*) FROM masters.party_job_code WHERE party_id = :p", p=pid)
        num_or_zero("login_job_codes", f"How many job codes are assigned to {login}?", jc, subject=login)
        avg = scalar("SELECT AVG(percentage) FROM lms.exam WHERE party_id = :p", p=pid)
        if avg is not None:
            add("login_avg_exam", f"What is the average exam percentage of {login}?", "num",
                [round(float(avg), 2)], tolerance=0.6, subject=login)
        if dept:
            add("login_dept", f"Which department does {login} belong to?", "text", [dept], subject=login)
            add("login_dept", f"what dept is user {login} in", "text", [dept], subject=login)
        tnames = sorted({r[0] for r in q(
            "SELECT DISTINCT t.name FROM lms.user_training ut JOIN lms.training t ON t.id = ut.training_id WHERE ut.party_id = :p",
            p=pid)})
        if tnames and len(tnames) <= 25:
            add("login_training_list", f"List the trainings assigned to {login}", "names", tnames, subject=login)
    if first:
        add("login_name", f"What is the first name of user {login}?", "text", [first], subject=login)

# Logins that do not exist - the answer must say none/not found.
for i, fake in enumerate(["oquser99", "qauser1", "john.doe", "hruser7", "testuser", "oquser0",
                          "admin9", "qahod2", "stc5", "manager1", "trainee1", "xyzuser",
                          "oq_user1", "user2", "approver9", "initiator9", "hrhod", "itadmin"]):
    add("fake_login", f"How many trainings are assigned to {fake}?", "zero", subject=fake)
    add("fake_login", f"How many exams has {fake} passed?", "zero", subject=fake)

# ---------------------------------------------------------------------
# 5. Per-training questions (distinct names; duplicates aggregated)
# ---------------------------------------------------------------------
tnames = sorted({r[0] for r in q("SELECT name FROM lms.training")})
for tn in tnames:
    assigned = scalar("SELECT COUNT(*) FROM lms.user_training ut JOIN lms.training t ON t.id = ut.training_id WHERE t.name = :n", n=tn)
    completed = scalar("""SELECT COUNT(*) FROM lms.user_training ut JOIN lms.training t ON t.id = ut.training_id
                          JOIN lms.status s ON s.id = ut.status_id WHERE t.name = :n AND s.name = 'UT_COMPLETED'""", n=tn)
    sessions = scalar("SELECT COUNT(*) FROM lms.training_session ts JOIN lms.training t ON t.id = ts.training_id WHERE t.name = :n", n=tn)
    cats = sorted({r[0] for r in q("""SELECT c.name FROM lms.training t JOIN lms.training_category c ON c.id = t.category_id
                                       WHERE t.name = :n""", n=tn) if r[0]})
    types = sorted({r[0] for r in q("""SELECT c.name FROM lms.training t JOIN lms.training_type c ON c.id = t.type_id
                                        WHERE t.name = :n""", n=tn) if r[0]})
    num_or_zero("training_assigned", f"How many users are assigned the training {tn}?", assigned, subject=tn)
    num_or_zero("training_completed", f"How many users completed training {tn}?", completed, subject=tn)
    num_or_zero("training_sessions", f"How many sessions were held for training {tn}?", sessions, subject=tn)
    if cats:
        add("training_category_of", f"What category is the training {tn} in?", "text", cats, subject=tn)
    if types:
        add("training_type_of", f"What is the training type of {tn}?", "text", types, subject=tn)

# ---------------------------------------------------------------------
# 6. Category / type / status breakdowns
# ---------------------------------------------------------------------
for (cat,) in q("SELECT name FROM lms.training_category"):
    n = scalar("SELECT COUNT(*) FROM lms.training t JOIN lms.training_category c ON c.id = t.category_id WHERE c.name = :c", c=cat)
    for v in {cat, words(cat) if "_" in cat else cat.lower(), words(cat)}:
        num_or_zero("category_training_count", f"How many trainings are in the {v} category?", n, subject=cat)
    names = sorted({r[0] for r in q("SELECT t.name FROM lms.training t JOIN lms.training_category c ON c.id = t.category_id WHERE c.name = :c", c=cat)})
    if names:
        add("category_training_list", f"List the {words(cat)} trainings", "names", names, subject=cat)
    ut = scalar("""SELECT COUNT(*) FROM lms.user_training ut JOIN lms.training t ON t.id = ut.training_id
                   JOIN lms.training_category c ON c.id = t.category_id WHERE c.name = :c""", c=cat)
    num_or_zero("category_assignments", f"How many training assignments are there for {words(cat)} trainings?", ut, subject=cat)

for (tt,) in q("SELECT name FROM lms.training_type"):
    n = scalar("SELECT COUNT(*) FROM lms.training t JOIN lms.training_type c ON c.id = t.type_id WHERE c.name = :c", c=tt)
    for v in {tt, words(tt)}:
        num_or_zero("type_training_count", f"How many trainings are of type {v}?", n, subject=tt)

for (st,) in q("SELECT DISTINCT s.name FROM lms.status s WHERE s.name LIKE 'UT\\_%'"):
    n = scalar("SELECT COUNT(*) FROM lms.user_training ut JOIN lms.status s ON s.id = ut.status_id WHERE s.name = :s", s=st)
    num_or_zero("ut_status_count", f"How many user trainings are {words(st)}?", n, subject=st)
    num_or_zero("ut_status_count", f"count of training assignments with status {st}", n, subject=st)
for (st,) in q("SELECT DISTINCT s.name FROM lms.status s WHERE s.name LIKE 'T\\_%'"):
    n = scalar("SELECT COUNT(*) FROM lms.training t JOIN lms.status s ON s.id = t.status_id WHERE s.name = :s", s=st)
    num_or_zero("training_status_count", f"How many trainings are in {words(st)} status?", n, subject=st)
for (st,) in q("SELECT DISTINCT s.name FROM lms.status s WHERE s.name LIKE 'E\\_%'"):
    n = scalar("SELECT COUNT(*) FROM lms.exam x JOIN lms.status s ON s.id = x.status_id WHERE s.name = :s", s=st)
    num_or_zero("exam_status_count", f"How many exams are {words(st)}?", n, subject=st)
for (et,) in q("SELECT name FROM lms.exam_type"):
    n = scalar("SELECT COUNT(*) FROM lms.exam x JOIN lms.exam_type t ON t.id = x.type_id WHERE t.name = :t", t=et)
    num_or_zero("exam_type_count", f"How many {et.lower()} exams were taken?", n, subject=et)
for (ps,) in q("SELECT DISTINCT name FROM masters.status"):
    n = scalar("SELECT COUNT(*) FROM masters.party p JOIN masters.status s ON s.id = p.status_id WHERE s.name = :s", s=ps)
    if ps.startswith("PARTY_"):
        num_or_zero("party_status_count", f"How many users are {words(ps)}?", n, subject=ps)
    r = scalar("SELECT COUNT(*) FROM masters.request r JOIN masters.status s ON s.id = r.status_id WHERE s.name = :s", s=ps)
    if r:
        num_or_zero("request_status_count", f"How many requests are in {words(ps)} status?", r, subject=ps)
for (rt,) in q("SELECT name FROM masters.request_type"):
    n = scalar("SELECT COUNT(*) FROM masters.request r JOIN masters.request_type t ON t.id = r.request_type_id WHERE t.name = :t", t=rt)
    num_or_zero("request_type_count", f"How many {words(rt) if '_' in rt else rt.lower()} requests are there?", n, subject=rt)
for (m,) in q("SELECT name FROM masters.mod"):
    n = scalar("SELECT COUNT(*) FROM masters.permission p JOIN masters.mod m ON m.id = p.mod_id WHERE m.name = :m", m=m)
    num_or_zero("module_permissions", f"How many permissions belong to the {m} module?", n, subject=m)
for nat in ("New", "Existing"):
    n = scalar("SELECT COUNT(*) FROM masters.party WHERE nature_of_intro = :n", n=nat)
    num_or_zero("nature_of_intro", f"How many {nat.lower()} users are there?", n, subject=nat)

# ---------------------------------------------------------------------
# 7. Dates (BIGINT epoch-microsecond columns, relative + absolute)
# ---------------------------------------------------------------------
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DATED = [
    ("lms.user_training", "created_date", "training assignments were created"),
    ("lms.exam", "created_date", "exams were taken"),
    ("lms.training_session", "from_date", "training sessions started"),
    ("masters.party", "date_of_joining", "users joined"),
    ("masters.request", "created_date", "requests were raised"),
]
for table, col, phrase in DATED:
    years = [int(r[0]) for r in q(f"SELECT DISTINCT EXTRACT(YEAR FROM to_timestamp({col}/1000000.0)) FROM {table} WHERE {col} IS NOT NULL")]
    for y in sorted(set(years) | {2023}):
        n = scalar(f"SELECT COUNT(*) FROM {table} WHERE EXTRACT(YEAR FROM to_timestamp({col}/1000000.0)) = :y", y=y)
        num_or_zero("date_year", f"How many {phrase} in {y}?", n, subject=f"{table}.{col}")
    for mi, mname in enumerate(MONTHS, start=1):
        n = scalar(f"""SELECT COUNT(*) FROM {table} WHERE EXTRACT(YEAR FROM to_timestamp({col}/1000000.0)) = 2026
                       AND EXTRACT(MONTH FROM to_timestamp({col}/1000000.0)) = :m""", m=mi)
        num_or_zero("date_month", f"How many {phrase} in {mname} 2026?", n, subject=f"{table}.{col}")

# ---------------------------------------------------------------------
# 8. Rankings / top-N / aggregates
# ---------------------------------------------------------------------
def top_text(family, question, sql):
    rows = q(sql)
    if not rows:
        return
    best = rows[0][1]
    winners = [r[0] for r in rows if r[1] == best and r[0]]
    add(family, question, "text", winners)


top_text("ranking", "Which user has the most trainings assigned?",
         "SELECT u.login, COUNT(*) FROM lms.user_training ut JOIN masters.party p ON p.id = ut.party_id JOIN pc.sys_user u ON u.id = p.user_id GROUP BY 1 ORDER BY 2 DESC")
top_text("ranking", "Which user has completed the most trainings?",
         "SELECT u.login, COUNT(*) FROM lms.user_training ut JOIN masters.party p ON p.id = ut.party_id JOIN pc.sys_user u ON u.id = p.user_id JOIN lms.status s ON s.id = ut.status_id WHERE s.name = 'UT_COMPLETED' GROUP BY 1 ORDER BY 2 DESC")
top_text("ranking", "Who failed the most exams?",
         "SELECT u.login, COUNT(*) FROM lms.exam x JOIN masters.party p ON p.id = x.party_id JOIN pc.sys_user u ON u.id = p.user_id JOIN lms.status s ON s.id = x.status_id WHERE s.name = 'E_FAILED' GROUP BY 1 ORDER BY 2 DESC")
top_text("ranking", "Who passed the most exams?",
         "SELECT u.login, COUNT(*) FROM lms.exam x JOIN masters.party p ON p.id = x.party_id JOIN pc.sys_user u ON u.id = p.user_id JOIN lms.status s ON s.id = x.status_id WHERE s.name = 'E_PASSED' GROUP BY 1 ORDER BY 2 DESC")
top_text("ranking", "Which department has the most users?",
         "SELECT ps.name, COUNT(*) FROM masters.party p JOIN masters.product_store ps ON ps.id = p.department_id GROUP BY 1 ORDER BY 2 DESC")
top_text("ranking", "Which department has the most job codes?",
         "SELECT ps.name, COUNT(*) FROM masters.job_code j JOIN masters.product_store ps ON ps.id = j.dept_id GROUP BY 1 ORDER BY 2 DESC")
top_text("ranking", "Which training has the most users assigned?",
         "SELECT t.name, COUNT(*) FROM lms.user_training ut JOIN lms.training t ON t.id = ut.training_id GROUP BY 1 ORDER BY 2 DESC")
top_text("ranking", "Which training category has the most trainings?",
         "SELECT c.name, COUNT(*) FROM lms.training t JOIN lms.training_category c ON c.id = t.category_id GROUP BY 1 ORDER BY 2 DESC")
top_text("ranking", "Which training has the most sessions?",
         "SELECT t.name, COUNT(*) FROM lms.training_session ts JOIN lms.training t ON t.id = ts.training_id GROUP BY 1 ORDER BY 2 DESC")
top_text("ranking", "Which user has the highest average exam percentage?",
         "SELECT u.login, ROUND(AVG(x.percentage)::numeric, 4) FROM lms.exam x JOIN masters.party p ON p.id = x.party_id JOIN pc.sys_user u ON u.id = p.user_id GROUP BY 1 ORDER BY 2 DESC NULLS LAST")
top_text("ranking", "Which training category has the fewest trainings?",
         "SELECT c.name, COUNT(*) FROM lms.training t JOIN lms.training_category c ON c.id = t.category_id GROUP BY 1 ORDER BY 2 ASC")
top_text("ranking", "Which department has the most completed trainings?",
         "SELECT ps.name, COUNT(*) FROM lms.user_training ut JOIN masters.party p ON p.id = ut.party_id JOIN masters.product_store ps ON ps.id = p.department_id JOIN lms.status s ON s.id = ut.status_id WHERE s.name = 'UT_COMPLETED' GROUP BY 1 ORDER BY 2 DESC")

AGG = [
    ("What is the average exam percentage overall?", "SELECT AVG(percentage) FROM lms.exam", 0.6),
    ("What is the highest exam percentage ever scored?", "SELECT MAX(percentage) FROM lms.exam", 0.01),
    ("What is the lowest exam percentage?", "SELECT MIN(percentage) FROM lms.exam", 0.01),
    ("What is the average marks in exams?", "SELECT AVG(marks) FROM lms.exam", 0.6),
    ("What is the exam pass rate?", "SELECT 100.0 * SUM(CASE WHEN s.name='E_PASSED' THEN 1 ELSE 0 END) / COUNT(*) FROM lms.exam x JOIN lms.status s ON s.id = x.status_id", 0.6),
    ("What percentage of user trainings are completed?", "SELECT 100.0 * SUM(CASE WHEN s.name='UT_COMPLETED' THEN 1 ELSE 0 END) / COUNT(*) FROM lms.user_training ut JOIN lms.status s ON s.id = ut.status_id", 0.6),
    ("How many distinct users have been assigned at least one training?", "SELECT COUNT(DISTINCT party_id) FROM lms.user_training", 0),
    ("How many distinct users have taken an exam?", "SELECT COUNT(DISTINCT party_id) FROM lms.exam", 0),
    ("How many users have never been assigned a training?", "SELECT COUNT(*) FROM masters.party p WHERE NOT EXISTS (SELECT 1 FROM lms.user_training ut WHERE ut.party_id = p.id)", 0),
    ("How many users have never taken an exam?", "SELECT COUNT(*) FROM masters.party p WHERE NOT EXISTS (SELECT 1 FROM lms.exam x WHERE x.party_id = p.id)", 0),
    ("How many trainings have never been assigned to anyone?", "SELECT COUNT(*) FROM lms.training t WHERE NOT EXISTS (SELECT 1 FROM lms.user_training ut WHERE ut.training_id = t.id)", 0),
    ("How many trainings require an exam?", "SELECT COUNT(*) FROM lms.training WHERE require_exam", 0),
    ("How many sessions were attended?", "SELECT COUNT(*) FROM lms.training_session_register WHERE attended", 0),
    ("How many online training sessions were there?", "SELECT COUNT(*) FROM lms.training_session WHERE is_online", 0),
    ("What is the average number of trainings per user?", "SELECT COUNT(*)::float / COUNT(DISTINCT party_id) FROM lms.user_training", 0.06),
    ("How many exams were passed and how many failed?", None, None),
]
for question, sql, tol in AGG:
    if sql is None:
        continue
    v = scalar(sql)
    v = 0 if v is None else float(v)
    if v == 0:
        add("aggregate", question, "zero")
    else:
        add("aggregate", question, "num", [round(v, 2)], tolerance=tol)
passed = scalar("SELECT COUNT(*) FROM lms.exam x JOIN lms.status s ON s.id = x.status_id WHERE s.name='E_PASSED'")
failed = scalar("SELECT COUNT(*) FROM lms.exam x JOIN lms.status s ON s.id = x.status_id WHERE s.name='E_FAILED'")
add("aggregate", "How many exams were passed and how many failed?", "num", [float(passed), float(failed)])

# Per-group breakdowns: every group's count must be present.
BREAKDOWNS = [
    ("How many users are there in each department?", "SELECT ps.name, COUNT(*) FROM masters.party p JOIN masters.product_store ps ON ps.id = p.department_id GROUP BY 1"),
    ("Show the number of trainings per category", "SELECT c.name, COUNT(*) FROM lms.training t JOIN lms.training_category c ON c.id = t.category_id GROUP BY 1"),
    ("Break down user trainings by status", "SELECT s.name, COUNT(*) FROM lms.user_training ut JOIN lms.status s ON s.id = ut.status_id GROUP BY 1"),
    ("How many exams per exam type?", "SELECT t.name, COUNT(*) FROM lms.exam x JOIN lms.exam_type t ON t.id = x.type_id GROUP BY 1"),
    ("Count job codes per department", "SELECT ps.name, COUNT(*) FROM masters.job_code j JOIN masters.product_store ps ON ps.id = j.dept_id GROUP BY 1"),
    ("How many trainings of each type are there?", "SELECT c.name, COUNT(*) FROM lms.training t JOIN lms.training_type c ON c.id = t.type_id GROUP BY 1"),
    ("How many requests per status?", "SELECT s.name, COUNT(*) FROM masters.request r JOIN masters.status s ON s.id = r.status_id GROUP BY 1"),
]
for question, sql in BREAKDOWNS:
    rows = q(sql)
    add("breakdown", question, "num", sorted({float(r[1]) for r in rows}))

# ---------------------------------------------------------------------
# 8b. Harder shapes: acronyms, casing, multi-condition filters
# ---------------------------------------------------------------------
for dept_id, dept in depts:
    initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", dept) if w.lower() not in {"and", "of", "the"}).upper()
    users = scalar("SELECT COUNT(*) FROM masters.party WHERE department_id = :d", d=dept_id)
    if len(initials) >= 2 and initials.lower() != dept.lower():
        num_or_zero("dept_acronym", f"How many users are in {initials}?", users, subject=dept)
    passed_d = scalar("""SELECT COUNT(*) FROM lms.exam x JOIN masters.party p ON p.id = x.party_id
                         JOIN lms.status s ON s.id = x.status_id WHERE p.department_id = :d AND s.name = 'E_PASSED'""", d=dept_id)
    num_or_zero("dept_exams_passed", f"How many exams did users in {dept} pass?", passed_d, subject=dept)

for uid, login, pid, first, dept in logins:
    num_or_zero("login_upper", f"How many trainings are assigned to {login.upper()}?", ut_count(pid), subject=login)
    if pid is None:
        continue
    row = q("""SELECT c.name, COUNT(*) FROM lms.user_training ut JOIN lms.training t ON t.id = ut.training_id
               JOIN lms.training_category c ON c.id = t.category_id WHERE ut.party_id = :p
               GROUP BY 1 ORDER BY 2 DESC LIMIT 1""", p=pid)
    if row:
        cat, n = row[0]
        num_or_zero("login_category", f"How many {words(cat)} trainings are assigned to {login}?", n, subject=login)

for mi in (3, 4, 5, 6):
    for st, label in (("E_PASSED", "passed"), ("E_FAILED", "failed")):
        n = scalar("""SELECT COUNT(*) FROM lms.exam x JOIN lms.status s ON s.id = x.status_id WHERE s.name = :s
                      AND EXTRACT(YEAR FROM to_timestamp(x.created_date/1000000.0)) = 2026
                      AND EXTRACT(MONTH FROM to_timestamp(x.created_date/1000000.0)) = :m""", s=st, m=mi)
        num_or_zero("month_status", f"How many exams were {label} in {MONTHS[mi - 1]} 2026?", n)

# ---------------------------------------------------------------------
# 9. Non-database questions (must answer directly, not crash/blank)
# ---------------------------------------------------------------------
for p in ["hello", "hi there!", "good morning", "thanks, that helps", "who are you?",
          "what can you help me with?", "what is a training needs analysis?",
          "what does GMP stand for?", "what is the capital of France?", "tell me a fun fact",
          "how are you today?", "what is 15 multiplied by 12?", "explain what an SOP is",
          "what is the difference between online and offline training?",
          "bye", "ok", "can you speak Hindi?", "what is an LMS?",
          "summarize what you do in one sentence", "what is induction training?"]:
    add("nodb", p, "nodb")

OUT.write_text(json.dumps(QUESTIONS, indent=1), encoding="utf-8")
fam = defaultdict(int)
for x in QUESTIONS:
    fam[x["family"]] += 1
print(f"{len(QUESTIONS)} questions -> {OUT}")
for k, v in sorted(fam.items(), key=lambda kv: -kv[1]):
    print(f"  {v:4d}  {k}")
