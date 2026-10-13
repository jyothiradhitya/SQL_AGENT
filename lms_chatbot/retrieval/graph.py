"""
Foreign-key graph over the schema, used for two narrow purposes:

1. bridge_tables() - find the tables that actually sit BETWEEN two
   anchor tables selected by embedding search, via shortest path, so a
   question spanning two entities ("trainings a user has completed")
   still gets the connecting table without pulling in the whole schema.

2. attach_lookups() - for a set of selected tables, find the small
   id -> label tables one hop away that are needed to resolve their
   foreign-key columns into something a human (or the SQL agent)
   can filter on by name instead of guessing an integer id.

Deliberately NOT a general-purpose "explore everything reachable within
depth N" BFS - that's what blew the old pipeline's prompt up to ~35
tables from a single seed. Shortest-path-between-selected-anchors and
one-hop-lookup-attachment are both bounded by construction.
"""

import logging
from collections import defaultdict, deque
from typing import Any

from lms_chatbot.retrieval.classify import NameIndex, table_name

logger = logging.getLogger(__name__)


class Edge:
    __slots__ = ("from_table", "from_column", "to_table", "to_column")

    def __init__(self, from_table: str, from_column: str, to_table: str, to_column: str):
        self.from_table = from_table
        self.from_column = from_column
        self.to_table = to_table
        self.to_column = to_column


Graph = dict[str, list[Edge]]


def build_fk_graph(kb: list[dict[str, Any]], name_index: NameIndex | None = None) -> Graph:
    name_index = name_index or NameIndex(kb)
    graph: Graph = defaultdict(list)

    for table in kb:
        name = table_name(table)
        if not name:
            continue

        for fk in table.get("foreign_keys") or []:
            fk_column = fk.get("column")
            raw_ref_table = fk.get("references_table")
            ref_column = fk.get("references_column")

            if not fk_column or not raw_ref_table or not ref_column:
                continue

            resolved_targets = name_index.resolve(raw_ref_table)
            if not resolved_targets:
                logger.debug(
                    "FK %s.%s -> %s.%s: target table not found in KB (even via "
                    "bare-name fallback), skipping edge",
                    name, fk_column, raw_ref_table, ref_column,
                )

            for ref_table in resolved_targets:
                graph[name.lower()].append(
                    Edge(name, fk_column, ref_table, ref_column)
                )
                graph[ref_table.lower()].append(
                    Edge(ref_table, ref_column, name, fk_column)
                )

    return graph


def shortest_path(graph: Graph, start: str, goal: str, max_hops: int) -> list[str] | None:
    """BFS shortest path between two tables, undirected, capped at max_hops."""

    start_key, goal_key = start.lower(), goal.lower()

    if start_key == goal_key:
        return [start]

    visited = {start_key}
    queue: deque[tuple[str, list[str]]] = deque([(start_key, [start])])

    while queue:
        current, path = queue.popleft()

        if len(path) - 1 >= max_hops:
            continue

        for edge in graph.get(current, []):
            next_key = edge.to_table.lower()

            if next_key in visited:
                continue

            next_path = path + [edge.to_table]

            if next_key == goal_key:
                return next_path

            visited.add(next_key)
            queue.append((next_key, next_path))

    return None


def bridge_tables(
    graph: Graph,
    anchors: list[str],
    max_hops: int,
) -> set[str]:
    """Tables strictly between pairs of anchors on their shortest connecting path."""

    bridges: set[str] = set()
    anchor_keys = {a.lower() for a in anchors}

    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            path = shortest_path(graph, a, b, max_hops)

            if path is None:
                logger.debug("No FK path within %d hops between %s and %s", max_hops, a, b)
                continue

            for table in path:
                if table.lower() not in anchor_keys:
                    bridges.add(table)

    return bridges


def attach_lookups(
    kb_lookup: dict[str, dict[str, Any]],
    name_index: NameIndex,
    tables: list[str],
    is_lookup_fn,
) -> set[str]:
    """One-hop outgoing FK targets of `tables` that are lookup tables."""

    lookups: set[str] = set()

    for name in tables:
        entry = kb_lookup.get(name.lower())
        if not entry:
            continue

        for fk in entry.get("foreign_keys") or []:
            raw_ref_table = fk.get("references_table")
            if not raw_ref_table:
                continue

            for ref_table in name_index.resolve(raw_ref_table):
                ref_entry = kb_lookup.get(ref_table.lower())
                if ref_entry and is_lookup_fn(ref_entry):
                    lookups.add(ref_table)

    return lookups
