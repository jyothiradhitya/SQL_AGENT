"""
Auto-grader for bulk_eval results: compares each chatbot answer against
the ground truth generate_questions.py computed from the real DB.

Verdicts:
  PASS        answer matches ground truth
  WRONG       answered, but the value(s) don't match
  NO_ANSWER   pipeline gave up / said it couldn't retrieve the data
  ERROR       exception, timeout, or empty answer
"""

import re

_GIVE_UP = (
    "could not retrieve the requested database information",
    "i could not answer that question",
    "something went wrong",
    "could not be retrieved",
    "unable to retrieve",
    "i couldn't retrieve",
    "i could not retrieve",
)
_NONE_PHRASES = (
    "no ", "none", "not found", "zero", "doesn't have any", "does not have any",
    "don't have any", "do not have any", "hasn't", "has not", "haven't", "have not",
    "there are no", "there were no", "no record", "not exist", "doesn't exist",
    "does not exist", "not assigned", "not registered", "isn't", "is not",
    "did not", "didn't", "never", "not any",
    "couldn't find", "could not find", "can't find", "cannot find", "no user", "no such",
    "unable to find", "unable to locate", "not able to find", "not able to locate",
)


def norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("‑", "-").replace(" ", " ").replace(" ", " ")
    s = re.sub(r"[_\-*`]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_WORD_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_WORD_NUMBERS.update({"a single": 1, "single": 1, "only one": 1})


def numbers(s: str) -> list[float]:
    s = (s or "").replace(",", "")
    found = [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", s)]
    low = s.lower()
    found += [float(v) for w, v in _WORD_NUMBERS.items() if re.search(rf"\b{w}\b", low)]
    return found


def _num_match(expected: list[float], found: list[float], tol: float) -> bool:
    def ok(e):
        t = tol if tol else (0.0 if float(e).is_integer() else 0.01)
        # a percentage answer may be phrased as a fraction or vice versa
        return any(abs(f - e) <= t or (0 < e <= 1 and abs(f - e * 100) <= max(t * 100, 0.6)) or
                   (e > 1 and abs(f / 100 - e) <= t) for f in found)
    return all(ok(e) for e in expected)


def grade(item: dict, result: dict) -> str:
    if result.get("exception") or result.get("timeout"):
        return "ERROR"
    answer = result.get("answer") or ""
    if not answer.strip():
        return "ERROR"
    a = norm(answer)
    if any(p in a for p in _GIVE_UP):
        return "NO_ANSWER"

    mode = item["mode"]
    options = item.get("any_of")
    if mode == "num" and options and [0.0] in options:
        found0 = numbers(answer)
        if 0.0 in found0 or any(p in a for p in _NONE_PHRASES):
            return "PASS"
    if mode == "nodb":
        return "PASS"

    found = numbers(answer)
    if mode == "zero":
        if 0.0 in found or any(p in a for p in _NONE_PHRASES):
            return "PASS"
        return "WRONG"

    if mode == "num":
        tol = item.get("tolerance") or 0.0
        options = item.get("any_of") or [item["expect"]]
        return "PASS" if any(_num_match(o, found, tol) for o in options) else "WRONG"

    if mode == "names":
        for alternative in [item["expect"]] + ([item["expect_alt"]] if item.get("expect_alt") else []):
            expected = [norm(x) for x in alternative]
            hit = sum(1 for x in expected if x in a)
            if hit >= max(1, round(0.8 * len(expected))):
                return "PASS"
        return "WRONG"

    if mode == "text":
        return "PASS" if any(norm(x) in a for x in item["expect"]) else "WRONG"

    raise ValueError(mode)
