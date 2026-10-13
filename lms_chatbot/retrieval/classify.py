"""
Anchor vs. lookup table classification.

The schema splits into two very different kinds of table, and treating
them the same is what made the old FK-graph BFS explode:

- Anchor ("fact") tables are real business entities (training, tm,
  user_training, exam...). These are what a user's question is
  actually about, and what embedding search should match against.

- Lookup ("reference") tables are small id -> label tables (status,
  category, type, role_type...). Nobody asks "what's in the status
  table" by name; they say "active trainings". These tables only
  matter as one-hop JOIN targets to resolve an *_id column into
  something readable, and they never need further expansion
  themselves - they're leaves in the join graph for our purposes.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_kb_tables(path: Path) -> list[dict[str, Any]]:
    """Table records from semantic_kb.json. SQL_AGENT/Retrivel/kb.py
    writes {"tables": [...]}; older KBs were a bare list - accept both."""
    with open(path, "r", encoding="utf-8") as f:
        kb = json.load(f)
    return kb.get("tables", []) if isinstance(kb, dict) else kb

_LOOKUP_ROLE_MARKERS = (
    "reference table",
    "lookup table",
    "lookup/reference",
    "reference/lookup",
    "type table",
    "status table",
)

# Fallback heuristic for tables whose table_role wasn't populated by the
# KB-generation step: a table with no outgoing FKs, at least one table
# pointing INTO it, and a handful of columns is almost always a lookup
# table (id, name, description, active_flag, ...).
_LOOKUP_MAX_COLUMNS = 6

# Small, explicit allowlist (same pattern as _IDENTITY_TABLES below) for
# tables that are genuinely id/name/description lookup tables but whose
# KB-generated table_role text happens not to contain any string in
# _LOOKUP_ROLE_MARKERS - the KB-generation step produces free text, not
# a controlled vocabulary, so "Classification" or "Classification and
# categorization of organizational units" describes exactly the same
# role as "reference table" without ever matching it textually.
#
# This matters beyond taxonomy: sample_values.py only renders real
# id -> name rows (retrieval/sample_values.py's actual_data VALUES
# list) for tables is_lookup_table() flags, and the SQL agent is told
# to copy filter values verbatim from that list rather than guess. A
# table stuck on the anchor side of this line never gets that list, so
# the agent is left guessing literal text for values it has never
# seen - confirmed via masters.product_store (org departments):
# real rows are 'Engineering Services', 'Quality Assurance', etc, but
# with no VALUES shown the agent filtered on the user's raw shorthand
# ('QA', 'OQ') verbatim and got zero rows every time, even with a
# wildcarded ILIKE, since e.g. "QA" isn't literally a substring of
# "Quality Assurance" - no amount of pattern matching fixes that, the
# agent has to actually see the real value once.
_FORCE_LOOKUP_TABLES = {
    "masters.product_store",
    "lms.training_category",
    "lms.training_type",
    "lms.status_category",
}

# The mirror-image bug: a genuine anchor/bridge table (multiple
# outgoing FKs, temporal from_date/thru_date columns, represents a
# membership/assignment EVENT - not a small id -> label table) gets
# wrongly demoted to lookup because its KB-generated table_role text
# happens to contain a _LOOKUP_ROLE_MARKERS phrase despite not fitting
# what that marker is meant to catch. Confirmed via
# masters.user_security_group: role text is "Reference table for user
# security group memberships" - the substring "reference table"
# matches, but it has 4 outgoing FKs (party_id, security_group_id,
# user_group_id, status_id) and is structurally identical to
# masters.product_store_user_group (correctly an anchor, since its own
# role text "Linking table" doesn't happen to match a marker). Being
# marked lookup means it never surfaces via anchor-table embedding
# search (see this module's docstring), so a "security group for user
# X" question never saw the one table that actually answers it, and
# the SQL agent fell back to the wrong bridge table
# (product_store_user_group) and a fabricated join, silently returning
# zero rows instead of failing loudly.
_FORCE_ANCHOR_TABLES = {
    "masters.user_security_group",
    # Same bug, same shape: "Reference table for managing access
    # control permissions at the group level" matches the "reference
    # table" marker, but this is a group_id/permission_id membership
    # table with from_date/thru_date tracking, not an id -> label table.
    "masters.security_group_permission",
}


def table_name(table: dict[str, Any]) -> str | None:
    return table.get("physical_qualified_name") or table.get("qualified_name")


def is_lookup_table(table: dict[str, Any]) -> bool:
    name = table_name(table)

    if name in _FORCE_ANCHOR_TABLES:
        return False

    if name in _FORCE_LOOKUP_TABLES:
        return True

    role = (table.get("table_role") or "").strip().lower()

    if role:
        return any(marker in role for marker in _LOOKUP_ROLE_MARKERS)

    columns = table.get("columns") or []
    outgoing_fks = table.get("foreign_keys") or []
    incoming_fks = table.get("incoming_foreign_keys") or []

    return (
        not outgoing_fks
        and bool(incoming_fks)
        and len(columns) <= _LOOKUP_MAX_COLUMNS
    )


def is_anchor_table(table: dict[str, Any]) -> bool:
    return not is_lookup_table(table)


# The "who is this row about" dimension for this schema. Too wide
# (40+ columns each) to pass the generic is_lookup_table heuristic, so
# they're real anchor tables by that test - but almost every
# person-related question ("for <name>", "<user>'s trainings", ...)
# needs one of them, one hop away from a fact table like
# user_training/exam, to turn a username into the party_id/emp_id
# those fact tables actually store. Relying on the SQL agent's LLM to
# notice that gap and re-call get_relevant_schema with a better-worded
# question is unreliable in practice (observed: it just repeats the
# same question) - so these are always attached, like a lookup table,
# whenever reachable in one hop from anything already selected. Small,
# explicit allowlist rather than a generic rule because "the identity
# table(s) for this domain" isn't something that can be inferred from
# column/table shape the way lookup tables can.
_IDENTITY_TABLES = {"masters.party", "pc.sys_user"}


def is_identity_table(table: dict[str, Any]) -> bool:
    name = table_name(table)
    return bool(name) and name.lower() in _IDENTITY_TABLES


class NameIndex:
    """Resolves a foreign-key target name to the KB's own entries.

    This KB has a real data-quality issue: a table's own entry is keyed
    by a logical schema prefix (e.g. "lms.training_category"), but other
    tables' foreign_keys[].references_table often point at it using a
    different prefix (observed: "public.training_category"). Exact
    match then fails silently, which broke both bridge-path-finding and
    lookup-attachment before this existed.

    Falls back to matching on the bare table name (the part after the
    last dot). Ambiguous bare names - e.g. "status" exists as both
    masters.status and lms.status - resolve to every match rather than
    guessing one: over-including a same-named lookup table is cheap,
    silently dropping a needed join is not.
    """

    def __init__(self, kb: list[dict[str, Any]]):
        self._by_qualified: dict[str, str] = {}
        self._by_bare: dict[str, list[str]] = defaultdict(list)

        for table in kb:
            name = table_name(table)
            if not name:
                continue

            self._by_qualified[name.lower()] = name
            self._by_bare[name.split(".")[-1].lower()].append(name)

    def resolve(self, raw_name: str | None) -> list[str]:
        if not raw_name:
            return []

        key = raw_name.lower()

        exact = self._by_qualified.get(key)
        if exact:
            return [exact]

        return list(self._by_bare.get(key.split(".")[-1], []))


def split_tables(
    kb: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (anchor_tables, lookup_tables)."""

    anchors: list[dict[str, Any]] = []
    lookups: list[dict[str, Any]] = []

    for table in kb:
        if not table_name(table):
            continue

        (lookups if is_lookup_table(table) else anchors).append(table)

    return anchors, lookups
