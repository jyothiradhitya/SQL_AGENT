"""
Shortcut (abbreviation) resolution, learned from the schema and data.

Users type shorthand ("how many CCs are open", "UOM list") that matches
no table description literally, so embedding search misses the right
tables. Rather than a hand-written glossary (which only ever covers the
terms someone thought to write down - see context._acronym for the same
reasoning on lookup values), every shortcut here is derived from what
the database itself contains:

- initials of each table's business name, aliases and business terms
  ("Change Control" -> CC, "Unit of Measure" -> UOM)
- table-name prefixes shared by several tables in one schema
  (qms.cc_action, qms.cc_type, ... -> "cc")
- underscore prefixes of real lookup values (CC_INIT, CC_QA_ASSESSED
  in qms.status -> "cc")

At question time, short codes in the question are looked up in that
index. A code that isn't found whole is split into a known prefix plus
the remaining letters (CCF -> CC + F) so the agent still learns the
part that is known. Whatever stays unresolved is reported as unknown,
so the agent asks the user instead of inventing a meaning.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from lms_chatbot.retrieval.context import _ACRONYM_STOPWORDS

# Ordinary words a lowercase token must not be mistaken for. Uppercase
# tokens in the question are always treated as possible shortcuts.
_COMMON_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "has", "have", "how", "i", "id", "if", "in", "is", "it", "me",
    "my", "no", "not", "of", "on", "or", "so", "the", "to", "up", "us",
    "was", "we", "what", "who", "why", "all", "any", "per", "via", "vs",
}

_TOKEN = re.compile(r"\b[A-Za-z]{2,6}s?\b")


@dataclass
class Shortcut:
    expansions: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, phrase: str, table: str) -> None:
        self.expansions[phrase.strip().lower()].add(table)


@dataclass
class Resolution:
    token: str
    meanings: list[tuple[str, list[str]]]  # (expansion, tables)
    unresolved: str = ""  # letters left over after a known prefix


def _texts(table: dict[str, Any]) -> list[str]:
    texts = [table.get("business_name") or "", table.get("business_meaning") or ""]
    for key in ("aliases", "business_terms", "related_business_concepts", "likely_user_phrases"):
        texts += table.get(key) or []
    return [t for t in texts if isinstance(t, str) and t.strip()]


def _value_texts(rows: list[dict[str, Any]]) -> list[str]:
    """Multi-word labels from real lookup rows ("Quality Assurance")."""
    return [
        v for row in rows for v in row.values()
        if isinstance(v, str) and " " in v.strip()
    ]


def _codes(words: list[str]) -> set[str]:
    """Initials of a phrase, with and without small words
    ("unit of measure" -> UOM and UM)."""
    full = "".join(w[0] for w in words)
    short = "".join(w[0] for w in words if w not in _ACRONYM_STOPWORDS)
    return {c for c in (full, short) if 2 <= len(c) <= 6}


def _singular(word: str) -> str:
    """Fold plurals so "change controls" counts as "change control"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _mine_phrases(text: str) -> set[tuple[str, str]]:
    """(code, phrase) for every 2-5 word run in `text` that doesn't
    start or end on a small word."""
    words = re.findall(r"[a-z]+", text.lower())
    found = set()
    for size in range(2, 6):
        for i in range(len(words) - size + 1):
            run = words[i:i + size]
            if run[0] in _ACRONYM_STOPWORDS or run[-1] in _ACRONYM_STOPWORDS:
                continue
            phrase = " ".join(run[:-1] + [_singular(run[-1])])
            for code in _codes(run):
                found.add((code, phrase))
    return found


def build_shortcut_index(
    kb: list[dict[str, Any]],
    sample_values: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Shortcut]:
    index: dict[str, Shortcut] = defaultdict(Shortcut)

    # 1. Phrases from each table's descriptions and its real values,
    # keyed by their initials. A meaning's weight is how many tables
    # use it, so "change control" (all qms.cc_* tables) outranks a
    # one-off "communication channel".
    for table in kb:
        name = table.get("qualified_name")
        if not name:
            continue
        texts = _texts(table) + _value_texts((sample_values or {}).get(name, []))
        for text in texts:
            for code, phrase in _mine_phrases(text):
                index[code.lower()].add(phrase, name)

    # 2. Table-name prefixes shared by 2+ tables in one schema
    # (qms.cc_action, qms.cc_type -> "cc"). A prefix that is itself a
    # word in those tables' names ("item") is not a shortcut.
    by_prefix: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for table in kb:
        parts = (table.get("table_name") or "").split("_")
        if len(parts) > 1 and 2 <= len(parts[0]) <= 5:
            by_prefix[(table.get("schema_name"), parts[0].lower())].append(table)

    for (_schema, prefix), tables in by_prefix.items():
        if len(tables) < 2:
            continue
        names = " ".join((t.get("business_name") or "").lower() for t in tables)
        if re.search(rf"\b{prefix}\b", names):
            continue
        # Link the prefix tables to the meanings they themselves mention
        own = {t["qualified_name"] for t in tables}
        for phrase, users in index.get(prefix, Shortcut()).expansions.items():
            if users & own:
                users |= own

    return dict(index)


def _candidate_tokens(question: str, index: dict[str, Shortcut]) -> list[str]:
    seen, tokens = set(), []
    for raw in _TOKEN.findall(question):
        token = raw
        # Plural shortcut: "CCs" -> "CC"
        if len(raw) > 2 and raw.endswith("s") and raw[:-1].isupper():
            token = raw[:-1]
        key = token.lower()
        if key in seen or key in _COMMON_WORDS:
            continue
        if token.isupper() or key in index:
            seen.add(key)
            tokens.append(token)
    return tokens


def resolve_shortcuts(question: str, index: dict[str, Shortcut]) -> list[Resolution]:
    results = []
    for token in _candidate_tokens(question, index):
        key = token.lower()
        if key in index:
            results.append(Resolution(token, _meanings(index[key])))
            continue
        # Longest known prefix: CCF -> CC + "F"
        for cut in range(len(key) - 1, 1, -1):
            if key[:cut] in index:
                results.append(Resolution(token, _meanings(index[key[:cut]]), token[cut:]))
                break
        else:
            results.append(Resolution(token, []))
    return results


def _meanings(shortcut: Shortcut) -> list[tuple[str, list[str]]]:
    """Top meanings by how many tables use them; a meaning used by far
    fewer tables than the best one is noise, not an alternative."""
    ranked = sorted(shortcut.expansions.items(), key=lambda kv: (-len(kv[1]), len(kv[0]), kv[0]))
    if not ranked:
        return []
    best = len(ranked[0][1])
    return [
        (phrase, sorted(tables)) for phrase, tables in ranked[:3]
        if len(tables) * 3 >= best
    ]


def expanded_question(question: str, resolutions: list[Resolution]) -> str:
    """Question text with known meanings appended, for embedding search."""
    extra = [phrase for r in resolutions for phrase, _ in r.meanings[:2]]
    return f"{question} ({'; '.join(extra)})" if extra else question


def render_shortcuts(resolutions: list[Resolution]) -> str:
    if not resolutions:
        return ""
    lines = ["SHORTCUTS IN QUESTION (derived from this database's own names/values):"]
    for r in resolutions:
        if not r.meanings:
            lines.append(
                f"  {r.token}: not found in any table name, description or value - "
                "if the answer depends on it, ask the user what it means instead of guessing."
            )
            continue
        meanings = "; ".join(
            f"{phrase} [{', '.join(tables[:3])}]" for phrase, tables in r.meanings
        )
        if r.unresolved:
            prefix = r.token[: len(r.token) - len(r.unresolved)]
            lines.append(
                f"  {r.token}: starts with {prefix} = {meanings}; the trailing "
                f"'{r.unresolved}' is not defined anywhere in the database - ask the "
                "user if it changes the answer."
            )
        else:
            lines.append(f"  {r.token} = {meanings}")
    return "\n".join(lines)
