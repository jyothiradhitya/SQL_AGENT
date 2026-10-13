"""
Deterministic FK graph data model + traversal.

This module only defines the *shape* of the graph artifact
(graph_builder.py builds one from semantic_kb.json and writes it to
fk_graph.json) and pure, LLM-free traversal over it:

    get_neighbors(table)                 - one-hop outgoing/incoming FKs
    find_paths(source, target, max_depth) - shortest FK path(s), with
                                             every hop's exact join
                                             columns preserved

Nothing here talks to Ollama, embeds anything, or executes SQL. It is
the deterministic "ground truth" layer that a later semantic-retrieval
step (question -> embedding -> anchor tables) will use to expand an
anchor table into its real join paths, replacing the ad-hoc BFS in
retrieval/graph.py. That wiring is a separate, later step - this
module is only the data structure and its traversal.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EDGE_TYPE_FOREIGN_KEY = "foreign_key"


@dataclass(frozen=True)
class FKEdge:
    """One real FK relationship, at column granularity.

    source_table.source_column -> target_table.target_column
    is the literal JOIN condition - never reduced to just table names.
    """

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    edge_type: str = EDGE_TYPE_FOREIGN_KEY

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FKEdge":
        return cls(
            source_table=data["source_table"],
            source_column=data["source_column"],
            target_table=data["target_table"],
            target_column=data["target_column"],
            edge_type=data.get("edge_type", EDGE_TYPE_FOREIGN_KEY),
        )

    def sort_key(self) -> tuple[str, str, str, str]:
        return (self.source_table, self.source_column, self.target_table, self.target_column)


@dataclass
class GraphNode:
    """A table, stripped to what the graph itself needs.

    The rich semantic fields (business_meaning, likely_user_phrases,
    ...) deliberately stay in semantic_kb.json only - this node is a
    lightweight pointer plus what's needed to validate FK columns,
    not a copy of the KB record.
    """

    reference: str
    schema: str
    table: str
    columns: list[str] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphNode":
        return cls(
            reference=data["reference"],
            schema=data["schema"],
            table=data["table"],
            columns=list(data.get("columns") or []),
            primary_keys=list(data.get("primary_keys") or []),
        )


@dataclass
class FKGraph:
    version: int
    nodes: dict[str, GraphNode]
    edges: list[FKEdge]
    unresolved_fks: list[dict[str, Any]]
    ambiguous_fks: list[dict[str, Any]]
    # FKs that were ambiguous by bare name but resolved deterministically
    # because exactly one candidate shared the source table's schema -
    # see graph_builder.resolve_fk_edges. Already included in `edges`;
    # kept here separately only so the resolution method stays auditable.
    schema_tiebreak_fks: list[dict[str, Any]] = field(default_factory=list)

    # Adjacency, keyed by lowercase table reference. Built once at
    # construction time so get_neighbors()/find_paths() don't rescan
    # the flat edge list on every call.
    _outgoing: dict[str, list[FKEdge]] = field(default_factory=dict, repr=False)
    _incoming: dict[str, list[FKEdge]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self._outgoing or self._incoming:
            return

        outgoing: dict[str, list[FKEdge]] = {}
        incoming: dict[str, list[FKEdge]] = {}

        for edge in self.edges:
            outgoing.setdefault(edge.source_table.lower(), []).append(edge)
            incoming.setdefault(edge.target_table.lower(), []).append(edge)

        self._outgoing = outgoing
        self._incoming = incoming

    def node(self, table: str) -> GraphNode | None:
        return self.nodes.get(table.lower())

    def get_neighbors(self, table: str) -> dict[str, list[FKEdge]]:
        """One-hop outgoing and incoming FK edges for `table`."""
        key = table.lower()
        return {
            "outgoing": list(self._outgoing.get(key, [])),
            "incoming": list(self._incoming.get(key, [])),
        }

    def find_paths(
        self, source: str, target: str, max_depth: int = 4
    ) -> list[list[dict[str, Any]]]:
        """Shortest FK path(s) between `source` and `target`.

        Traverses edges in EITHER direction (a question may need to
        walk an FK backwards, e.g. "users in this group" walks
        group <- user_group.group_id), capped at max_depth hops.
        Every hop keeps its real edge (source_table, source_column,
        target_table, target_column) plus which way it was walked
        relative to the path, so the caller always has the exact JOIN
        condition - never just "table A relates to table B".

        Returns every shortest path found (there can be more than one
        at the same minimal depth); empty list if none exists within
        max_depth.
        """
        source_key, target_key = source.lower(), target.lower()

        if source_key not in self.nodes:
            return []
        if source_key == target_key:
            return [[]]

        # BFS layer by layer so we can collect every path tied for
        # shortest, not just the first one found.
        visited = {source_key}
        frontier: list[tuple[str, list[dict[str, Any]]]] = [(source_key, [])]
        depth = 0

        while frontier and depth < max_depth:
            depth += 1
            next_frontier: list[tuple[str, list[dict[str, Any]]]] = []
            found: list[list[dict[str, Any]]] = []
            newly_visited: set[str] = set()

            for current, path in frontier:
                for edge in self._outgoing.get(current, []):
                    self._visit_edge(edge, "outgoing", current, target_key, path,
                                      visited, newly_visited, next_frontier, found)
                for edge in self._incoming.get(current, []):
                    self._visit_edge(edge, "incoming", current, target_key, path,
                                      visited, newly_visited, next_frontier, found)

            if found:
                return found

            visited |= newly_visited
            frontier = next_frontier

        return []

    @staticmethod
    def _visit_edge(edge, direction, current, target_key, path,
                     visited, newly_visited, next_frontier, found):
        next_table = edge.target_table if direction == "outgoing" else edge.source_table
        next_key = next_table.lower()

        if next_key in visited or next_key in newly_visited:
            return

        hop = {"direction": direction, "from": current, "to": next_key, **edge.to_dict()}
        next_path = path + [hop]

        if next_key == target_key:
            found.append(next_path)
            return

        newly_visited.add(next_key)
        next_frontier.append((next_key, next_path))

    def to_dict(self) -> dict[str, Any]:
        adjacency = {
            ref: {
                "outgoing": [e.to_dict() for e in sorted(self._outgoing.get(ref, []), key=FKEdge.sort_key)],
                "incoming": [e.to_dict() for e in sorted(self._incoming.get(ref, []), key=FKEdge.sort_key)],
            }
            for ref in sorted(self.nodes)
        }

        return {
            "version": self.version,
            "nodes": {ref: node.to_dict() for ref, node in sorted(self.nodes.items())},
            "edges": [e.to_dict() for e in sorted(self.edges, key=FKEdge.sort_key)],
            "adjacency": adjacency,
            "unresolved_fks": self.unresolved_fks,
            "ambiguous_fks": self.ambiguous_fks,
            "schema_tiebreak_fks": self.schema_tiebreak_fks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FKGraph":
        nodes = {ref: GraphNode.from_dict(n) for ref, n in data["nodes"].items()}
        edges = [FKEdge.from_dict(e) for e in data["edges"]]
        return cls(
            version=data.get("version", 1),
            nodes=nodes,
            edges=edges,
            unresolved_fks=data.get("unresolved_fks", []),
            ambiguous_fks=data.get("ambiguous_fks", []),
            schema_tiebreak_fks=data.get("schema_tiebreak_fks", []),
        )


def bridge_tables(graph: FKGraph, anchors: list[str], max_hops: int) -> set[str]:
    """Tables strictly between pairs of anchors on their shortest FK
    path - the fk_graph.json-backed replacement for retrieval/graph.py's
    function of the same name and shape, so retrieval/pipeline.py can
    swap in this deterministic graph without changing its own call
    sites' expectations."""

    bridges: set[str] = set()
    anchor_keys = {a.lower() for a in anchors}

    for i, a in enumerate(anchors):
        for b in anchors[i + 1:]:
            paths = graph.find_paths(a, b, max_depth=max_hops)
            if not paths:
                continue

            for hop in paths[0]:
                for table in (hop["source_table"], hop["target_table"]):
                    if table.lower() not in anchor_keys:
                        bridges.add(table)

    return bridges


def attach_lookups(
    graph: FKGraph,
    kb_lookup: dict[str, dict[str, Any]],
    tables: list[str],
    is_lookup_fn,
) -> set[str]:
    """One-hop outgoing FK targets of `tables` that are lookup tables.

    Needs `kb_lookup` (the raw semantic_kb.json records, keyed by
    lowercase table name) alongside the graph because "is this a
    lookup table" reads business fields (table_role, business
    descriptions) that GraphNode deliberately doesn't duplicate.
    """

    lookups: set[str] = set()

    for name in tables:
        for edge in graph.get_neighbors(name)["outgoing"]:
            entry = kb_lookup.get(edge.target_table.lower())
            if entry and is_lookup_fn(entry):
                lookups.add(edge.target_table)

    return lookups


def save_graph(graph: FKGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, indent=2, sort_keys=False)
        f.write("\n")


def load_graph(path: Path) -> FKGraph:
    with open(path, "r", encoding="utf-8") as f:
        return FKGraph.from_dict(json.load(f))
