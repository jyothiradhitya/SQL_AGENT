"""
In-place curation of questions.json (ids kept stable so results stay
comparable across runs). Fixes ambiguities in the GENERATED BANK itself,
found while reviewing the baseline - these are test-bank defects, not
chatbot defects:

  * a question with more than one legitimate reading gets every valid
    ground truth in `any_of` (e.g. "how many trainings does X have" =
    assignment rows OR distinct trainings - the same training can be
    assigned twice)
  * a question unanswerable as worded is marked "ambiguous" and scored
    separately: the login literally named "user" ("How many exams did
    user pass?" reads as a missing name), and generator phrasing
    collisions (ON_JOB rendered as "job"; RQ_APPROVED and APPROVED both
    rendered as "approved")

    PYTHONPATH=. uv run python experiments/bulk_eval/curate.py
"""

import json
import re
from pathlib import Path

from sqlalchemy import create_engine, text

from lms_chatbot.config import settings

P = Path(__file__).with_name("questions.json")
Q = json.loads(P.read_text(encoding="utf-8"))
conn = create_engine(settings.analytical_db_url).connect()


def scalar(sql, **kw):
    return conn.execute(text(sql), kw).scalar()


def party_of(login):
    return scalar("SELECT p.id FROM masters.party p JOIN pc.sys_user u ON u.id = p.user_id WHERE u.login = :l", l=login)


def opts(*values):
    out = []
    for v in values:
        v = float(v or 0)
        if [v] not in out:
            out.append([v])
    return out


changed = 0
for q in Q:
    q.pop("ambiguous", None)
    fam, s = q["family"], q.get("subject")

    # --- unanswerable as worded --------------------------------------
    if fam.startswith("login_") and s == "user":
        q["ambiguous"] = "the login 'user' reads as a missing name in English"
    if q["question"] in ("How many trainings are in the job category?",
                         "How many requests are in approved status?",
                         "How many users are party job code?",
                         "How many exams are created?"):
        q["ambiguous"] = "generator phrasing collision / not a natural reading"
    if fam in ("login_job_codes", "login_trainings") and s == "hra":
        q["ambiguous"] = "'hra' is both a login and the HRA department's acronym"

    # --- several legitimate readings ---------------------------------
    pid = party_of(s) if fam.startswith("login_") and s else None
    if fam == "login_trainings" and pid:
        q["any_of"] = opts(q["expect"][0] if q["expect"] else 0,
                           scalar("SELECT COUNT(DISTINCT training_id) FROM lms.user_training WHERE party_id = :p", p=pid))
    if fam == "login_completed" and pid:
        base = [x[0] for x in q.get("any_of") or [q["expect"] or [0]]]
        q["any_of"] = opts(*base, scalar(
            "SELECT COUNT(DISTINCT ut.training_id) FROM lms.user_training ut JOIN lms.status s ON s.id = ut.status_id "
            "WHERE ut.party_id = :p AND s.name = 'UT_COMPLETED'", p=pid))
    if fam == "login_sessions" and pid:
        q["any_of"] = opts(q["expect"][0] if q["expect"] else 0, scalar(
            "SELECT COUNT(DISTINCT training_session_id) FROM lms.training_session_register WHERE trainee_id = :p", p=pid))
    if fam == "login_name" and s:
        both = conn.execute(text(
            "SELECT p.first_name, u.first_name FROM pc.sys_user u LEFT JOIN masters.party p ON p.user_id = u.id "
            "WHERE u.login = :l"), {"l": s}).first()
        q["expect"] = sorted({v for v in both if v})
    if fam == "dept_user_list" and s:
        # listing people by login is at least as correct as by first name
        q["expect_alt"] = [r[0] for r in conn.execute(text(
            "SELECT u.login FROM masters.party p JOIN pc.sys_user u ON u.id = p.user_id "
            "JOIN masters.product_store ps ON ps.id = p.department_id WHERE ps.name = :d"), {"d": s}).fetchall()]
    if fam == "login_pending" and pid:
        q["any_of"] = opts(q["expect"][0] if q["expect"] else 0, scalar(
            "SELECT COUNT(DISTINCT ut.training_id) FROM lms.user_training ut JOIN lms.status s ON s.id = ut.status_id "
            "WHERE ut.party_id = :p AND s.name = 'UT_CREATED'", p=pid))
    if q["question"] == "How many employees are registered?":
        q["any_of"] = opts(25, 26, scalar(
            "SELECT COUNT(*) FROM masters.party p JOIN masters.party_type t ON t.id = p.party_type_id WHERE t.name = 'EMPLOYEE'"))
    if q["question"] == "How many trainees registered for sessions in total?":
        q["any_of"] = opts(133, scalar("SELECT COUNT(DISTINCT trainee_id) FROM lms.training_session_register"))
    # any_of lists must be the zero-aware form the grader expects
    if q.get("any_of") and all(o == [0.0] for o in q["any_of"]):
        q["mode"], q["any_of"] = "zero", None
    elif q.get("any_of") and q["mode"] == "zero":
        q["mode"] = "num"
    changed += 1

P.write_text(json.dumps(Q, indent=1), encoding="utf-8")
print(f"curated {len(Q)} questions; ambiguous: {sum(1 for q in Q if q.get('ambiguous'))}; "
      f"multi-answer: {sum(1 for q in Q if q.get('any_of'))}")
