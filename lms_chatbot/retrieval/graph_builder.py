"""
Offline build step: construct a deterministic, column-level FK graph
from semantic_kb.json.

    semantic_kb.json  --(this module)-->  fk_graph.json

No LLM, no embeddings, no SQL execution - this only reads table/FK
metadata already present in the KB and turns it into a graph that can
be traversed with plain BFS (fk_graph.FKGraph.get_neighbors /
find_paths). A later step will use question-embedding similarity to
pick a starting table and then walk this graph to find the real join
path to other relevant tables - that wiring does not exist yet.

Run whenever semantic_kb.json changes:

    uv run python -m lms_chatbot.retrieval.graph_builder

Same input always produces the same fk_graph.json: table and edge
order in the output are sorted, not "whatever order the KB happened to
list things in".

FK RESOLUTION
-------------
Every table's `foreign_keys[].references_table` is matched against the
KB's own table names via classify.NameIndex, which already encodes the
project's two-step lookup:

    1. exact qualified-name match (e.g. "masters.status")
    2. unique match by bare table name (e.g. "public.status" ->
       the one table literally named "status", if there is only one)

If that yields exactly one candidate, the edge is created. Zero
candidates -> UNRESOLVED. More than one -> AMBIGUOUS. Neither case
guesses; both are logged and recorded in the output file instead of
silently dropped or silently resolved to an arbitrary candidate - the
KB is known to reuse a "public.<table>" placeholder schema for FK
targets that don't correspond to any real "public" schema, and several
bare table names (status, status_category, ...) exist in more than one
real schema.

NAMING-CONVENTION INFERENCE
----------------------------
semantic_kb.json's declared foreign_keys[] is itself curated (an
earlier LLM extraction pass) and demonstrably incomplete - real
relationships exist whose column was never recorded there at all (see
infer_naming_convention_fks). After declared FKs are resolved above,
a second deterministic pass scans every column on every table that
was NOT already the source of a declared edge: any "<word>_id" column
whose "<word>" is the exact bare name of exactly one other table in
the KB is inferred as a foreign key to that table's primary key. Same
zero/one/many resolution and same-schema tie-break as declared FKs;
same refusal to guess a wrong target. These edges are tagged
edge_type="inferred_foreign_key" (vs "foreign_key" for declared ones)
and merged into the same edge list, so JOINS rendering treats them
identically - only the validation report's inferred_* fields keep the
provenance distinct for audit.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from lms_chatbot.config import settings
from lms_chatbot.logging_config import setup_logging
from lms_chatbot.retrieval.classify import NameIndex, read_kb_tables, table_name
from lms_chatbot.retrieval.fk_graph import FKEdge, FKGraph, GraphNode, save_graph

logger = logging.getLogger(__name__)


def _column_names(table: dict[str, Any]) -> list[str]:
    return [
        c.get("column_name")
        for c in (table.get("columns") or [])
        if c.get("column_name")
    ]


def _primary_key_names(table: dict[str, Any]) -> list[str]:
    return [
        pk.get("column")
        for pk in (table.get("primary_keys") or [])
        if pk.get("column")
    ]


def _canonical_name(table: dict[str, Any]) -> str | None:
    name = table_name(table)
    if name:
        return name

    schema, bare = table.get("schema_name"), table.get("table_name")
    if schema and bare:
        return f"{schema}.{bare}"

    return None


def load_kb(path: Path) -> list[dict[str, Any]]:
    logger.info("[FK_GRAPH] Loading semantic KB from %s", path)

    kb = read_kb_tables(path)

    logger.info("[FK_GRAPH] Tables discovered: %d", len(kb))
    return kb


def build_nodes(kb: list[dict[str, Any]]) -> dict[str, GraphNode]:
    """Every table in the KB becomes a node, before any FK is looked
    at - so FK processing order can never affect which nodes exist."""

    nodes: dict[str, GraphNode] = {}

    for table in kb:
        name = _canonical_name(table)

        if not name:
            logger.warning(
                "[FK_GRAPH] Skipping table record with no resolvable name "
                "(schema_name=%r, table_name=%r)",
                table.get("schema_name"), table.get("table_name"),
            )
            continue

        if name in nodes:
            logger.warning(
                "[FK_GRAPH] Duplicate node name %r - keeping the first "
                "occurrence, ignoring the later one",
                name,
            )
            continue

        schema, _, bare = name.partition(".")
        nodes[name] = GraphNode(
            reference=name,
            schema=table.get("schema_name") or schema,
            table=table.get("table_name") or bare,
            columns=_column_names(table),
            primary_keys=_primary_key_names(table),
        )

    logger.info("[FK_GRAPH] Nodes created: %d", len(nodes))
    return nodes


def _schema_of(qualified_name: str) -> str:
    return qualified_name.split(".", 1)[0]


def resolve_fk_edges(
    kb: list[dict[str, Any]],
    nodes: dict[str, GraphNode],
    name_index: NameIndex,
) -> tuple[list[FKEdge], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    """Returns (edges, unresolved_fks, ambiguous_fks, schema_tiebreak_fks,
    total_fk_records).

    Only FKs belonging to a table that made it into `nodes` are
    considered - a table with no resolvable name has no FKs, by
    construction, not by an extra check here.

    Resolution priority, applied in order, per FK:
      1. exact qualified-name match, or a unique bare-name match
         (both via NameIndex.resolve) -> resolved outright.
      2. bare name matches more than one table (this KB has 4:
         status, status_category, sys_persistent_audit_event,
         sys_persistent_audit_evt_data, each duplicated across the
         masters/lms schemas) -> if exactly ONE candidate shares its
         schema with the FK's own source table, resolve to that one.
         This is a deterministic, logged convention (every table's own
         status/audit lookup lives in its own schema; nothing here
         references another module's private lookup table) - not a
         guess pulled from nowhere, and every use of it is recorded in
         schema_tiebreak_fks for audit, distinct from a plain 1-match
         resolution.
      3. otherwise (0 or >1 same-schema candidates) -> genuinely
         ambiguous, no edge created, logged and recorded.
    """

    node_keys = set(nodes)
    edges: list[FKEdge] = []
    unresolved: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    schema_tiebreak: list[dict[str, Any]] = []
    total_fk_records = 0

    for table in kb:
        source = _canonical_name(table)
        if not source or source not in nodes:
            continue

        for fk in table.get("foreign_keys") or []:
            total_fk_records += 1

            source_column = fk.get("column")
            raw_target = fk.get("references_table")
            target_column = fk.get("references_column")

            if not source_column or not raw_target or not target_column:
                logger.warning(
                    "[FK_GRAPH] Skipping malformed FK record on %s: %r",
                    source, fk,
                )
                continue

            candidates = [c for c in name_index.resolve(raw_target) if c in node_keys]

            if len(candidates) == 1:
                edges.append(FKEdge(source, source_column, candidates[0], target_column))
                continue

            if len(candidates) == 0:
                logger.warning(
                    "[FK_GRAPH] UNRESOLVED:\n"
                    "    source=%s\n"
                    "    column=%s\n"
                    "    target=%s",
                    source, source_column, raw_target,
                )
                unresolved.append({
                    "source_table": source,
                    "source_column": source_column,
                    "raw_target": raw_target,
                    "target_column": target_column,
                })
                continue

            same_schema = [c for c in candidates if _schema_of(c) == _schema_of(source)]

            if len(same_schema) == 1:
                resolved_target = same_schema[0]
                edges.append(FKEdge(source, source_column, resolved_target, target_column))
                logger.info(
                    "[FK_GRAPH] AMBIGUOUS_RESOLVED_BY_SCHEMA:\n"
                    "    source=%s.%s\n"
                    "    target=%s\n"
                    "    candidates=%s\n"
                    "    resolved_to=%s (same schema as source)",
                    source, source_column, raw_target, candidates, resolved_target,
                )
                schema_tiebreak.append({
                    "source_table": source,
                    "source_column": source_column,
                    "raw_target": raw_target,
                    "target_column": target_column,
                    "candidates": candidates,
                    "resolved_to": resolved_target,
                })
                continue

            logger.warning(
                "[FK_GRAPH] AMBIGUOUS:\n"
                "    source=%s.%s\n"
                "    target=%s\n"
                "    candidates=%s",
                source, source_column, raw_target, candidates,
            )
            ambiguous.append({
                "source_table": source,
                "source_column": source_column,
                "raw_target": raw_target,
                "target_column": target_column,
                "candidates": candidates,
            })

    logger.info("[FK_GRAPH] FK records discovered: %d", total_fk_records)
    logger.info("[FK_GRAPH] FK resolved: %d", len(edges))
    logger.info("[FK_GRAPH] FK resolved via schema tie-break: %d", len(schema_tiebreak))
    logger.info("[FK_GRAPH] FK unresolved: %d", len(unresolved))
    logger.info("[FK_GRAPH] FK ambiguous: %d", len(ambiguous))
    logger.info("[FK_GRAPH] Graph edges created: %d", len(edges))

    return edges, unresolved, ambiguous, schema_tiebreak, total_fk_records


EDGE_TYPE_INFERRED = "inferred_foreign_key"


def infer_naming_convention_fks(
    nodes: dict[str, GraphNode],
    name_index: NameIndex,
    declared_edges: list[FKEdge],
) -> tuple[list[FKEdge], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fill gaps the KB's own foreign_keys[] left out.

    semantic_kb.json's declared FKs are curated (an earlier LLM pass)
    and demonstrably incomplete - e.g. lms.user_training.party_id, the
    actual link from a training record to its trainee, was never
    declared, even though the column and its target table both exist.
    Column-naming convention is enough to find these deterministically:
    a "<word>_id" column whose "<word>" is the EXACT bare name of
    exactly one other table in this KB is a foreign key to that
    table's primary key, full stop - no fuzzy matching, no guessing at
    plurals or synonyms. That precision is deliberate: it only ever
    fires on an unambiguous exact match, so it can't invent a wrong
    relationship, only miss one whose column name doesn't happen to
    match a table name (those stay unresolved, same as an
    UNRESOLVED/AMBIGUOUS declared FK - logged, never silently dropped).

    Skips:
    - any column already the source of a declared edge (declared wins)
    - primary key columns (a table's own id is never a reference)
    - the bare literal column "id"

    Ambiguous bare-name matches (e.g. "status") get the same
    same-schema tie-break as declared FKs; still-ambiguous or
    zero-candidate columns are recorded, not guessed.
    """

    declared_sources = {
        (edge.source_table.lower(), edge.source_column.lower()) for edge in declared_edges
    }

    inferred: list[FKEdge] = []
    unresolved: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for source, node in nodes.items():
        pk_set = {pk.lower() for pk in node.primary_keys}

        for column in node.columns:
            lname = column.lower()

            if not lname.endswith("_id") or lname == "id":
                continue
            if (source, lname) in declared_sources:
                continue
            if lname in pk_set:
                continue

            stripped = lname[:-3]
            if not stripped:
                continue

            candidates = [c for c in name_index.resolve(stripped) if c in nodes]

            if len(candidates) == 1:
                target = candidates[0]
                target_column = nodes[target].primary_keys[0] if nodes[target].primary_keys else "id"
                if target_column not in nodes[target].columns:
                    continue
                inferred.append(FKEdge(node.reference, column, target, target_column, EDGE_TYPE_INFERRED))
                continue

            if len(candidates) == 0:
                unresolved.append({
                    "source_table": node.reference,
                    "source_column": column,
                    "attempted_bare_name": stripped,
                })
                continue

            same_schema = [c for c in candidates if _schema_of(c) == _schema_of(source)]

            if len(same_schema) == 1:
                target = same_schema[0]
                target_column = nodes[target].primary_keys[0] if nodes[target].primary_keys else "id"
                if target_column not in nodes[target].columns:
                    continue
                inferred.append(FKEdge(node.reference, column, target, target_column, EDGE_TYPE_INFERRED))
                continue

            ambiguous.append({
                "source_table": node.reference,
                "source_column": column,
                "attempted_bare_name": stripped,
                "candidates": candidates,
            })

    logger.info(
        "[FK_GRAPH] Naming-convention inference: %d edges inferred, %d unresolved, %d ambiguous",
        len(inferred), len(unresolved), len(ambiguous),
    )
    for edge in inferred:
        logger.info(
            "[FK_GRAPH] INFERRED_FK: %s.%s -> %s.%s",
            edge.source_table, edge.source_column, edge.target_table, edge.target_column,
        )

    return inferred, unresolved, ambiguous


def validate_graph(
    nodes: dict[str, GraphNode],
    edges: list[FKEdge],
    total_fk_records: int,
    unresolved: list[dict[str, Any]],
    ambiguous: list[dict[str, Any]],
    schema_tiebreak: list[dict[str, Any]],
    inferred: list[FKEdge] | None = None,
    inferred_unresolved: list[dict[str, Any]] | None = None,
    inferred_ambiguous: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Structural validation + the relationship-shape report requested
    for this step. Raises ValueError on the first structural violation
    (an edge pointing at a table, or a column, that doesn't exist) -
    those would mean build_nodes/resolve_fk_edges have a bug, not that
    the KB has a data-quality issue."""

    for edge in edges:
        if edge.source_table not in nodes:
            raise ValueError(
                f"Validation failed: edge source_table {edge.source_table!r} "
                f"not in nodes (edge={edge.to_dict()})"
            )
        if edge.target_table not in nodes:
            raise ValueError(
                f"Validation failed: edge target_table {edge.target_table!r} "
                f"not in nodes (edge={edge.to_dict()})"
            )

        source_columns = nodes[edge.source_table].columns
        if edge.source_column not in source_columns:
            raise ValueError(
                f"Validation failed: source_column {edge.source_column!r} not "
                f"found on {edge.source_table} (columns={source_columns}, "
                f"edge={edge.to_dict()})"
            )

        target_columns = nodes[edge.target_table].columns
        if edge.target_column not in target_columns:
            raise ValueError(
                f"Validation failed: target_column {edge.target_column!r} not "
                f"found on {edge.target_table} (columns={target_columns}, "
                f"edge={edge.to_dict()})"
            )

    outgoing_count: Counter[str] = Counter(e.source_table for e in edges)
    incoming_count: Counter[str] = Counter(e.target_table for e in edges)

    zero_relationship = sorted(
        n for n in nodes if outgoing_count[n] == 0 and incoming_count[n] == 0
    )
    incoming_only = sorted(
        n for n in nodes if outgoing_count[n] == 0 and incoming_count[n] > 0
    )
    outgoing_only = sorted(
        n for n in nodes if outgoing_count[n] > 0 and incoming_count[n] == 0
    )
    both = sorted(
        n for n in nodes if outgoing_count[n] > 0 and incoming_count[n] > 0
    )

    edge_counts = Counter(e.sort_key() for e in edges)
    duplicate_edges = [
        {"source_table": k[0], "source_column": k[1], "target_table": k[2],
         "target_column": k[3], "count": v}
        for k, v in edge_counts.items() if v > 1
    ]

    self_referencing = [e.to_dict() for e in edges if e.source_table == e.target_table]

    report = {
        "total_table_nodes": len(nodes),
        "total_fk_records": total_fk_records,
        "total_resolved_edges": len(edges),
        "total_resolved_via_schema_tiebreak": len(schema_tiebreak),
        "total_unresolved_edges": len(unresolved),
        "total_ambiguous_edges": len(ambiguous),
        "total_inferred_edges": len(inferred or []),
        "total_inferred_unresolved": len(inferred_unresolved or []),
        "total_inferred_ambiguous": len(inferred_ambiguous or []),
        "inferred_edges": [e.to_dict() for e in (inferred or [])],
        "inferred_unresolved": inferred_unresolved or [],
        "inferred_ambiguous": inferred_ambiguous or [],
        "tables_zero_relationships": zero_relationship,
        "tables_incoming_only": incoming_only,
        "tables_outgoing_only": outgoing_only,
        "tables_both_directions": both,
        "duplicate_fk_edges": duplicate_edges,
        "self_referencing_fk_edges": self_referencing,
    }

    logger.info(
        "[FK_GRAPH] Validation passed - %d nodes, %d edges, %d zero-relationship "
        "tables, %d duplicate edges, %d self-referencing edges",
        len(nodes), len(edges), len(zero_relationship),
        len(duplicate_edges), len(self_referencing),
    )

    return report


def build_graph(kb_path: Path | None = None) -> tuple[FKGraph, dict[str, Any]]:
    kb_path = kb_path or settings.semantic_kb_path
    kb = load_kb(kb_path)

    nodes = build_nodes(kb)
    name_index = NameIndex(kb)
    edges, unresolved, ambiguous, schema_tiebreak, total_fk_records = resolve_fk_edges(
        kb, nodes, name_index
    )

    inferred, inferred_unresolved, inferred_ambiguous = infer_naming_convention_fks(
        nodes, name_index, edges
    )
    edges = edges + inferred

    validation_report = validate_graph(
        nodes, edges, total_fk_records, unresolved, ambiguous, schema_tiebreak,
        inferred, inferred_unresolved, inferred_ambiguous,
    )

    graph = FKGraph(
        version=1,
        nodes=nodes,
        edges=edges,
        unresolved_fks=unresolved,
        ambiguous_fks=ambiguous,
        schema_tiebreak_fks=schema_tiebreak,
    )

    return graph, validation_report


def main() -> None:
    setup_logging()

    graph, validation_report = build_graph()

    output_path = settings.fk_graph_path
    save_graph(graph, output_path)
    logger.info("[FK_GRAPH] Saved graph: %s", output_path)

    report_path = output_path.with_name(output_path.stem + ".validation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
        f.write("\n")
    logger.info("[FK_GRAPH] Saved validation report: %s", report_path)

    logger.info("[FK_GRAPH] Validation report: %s", json.dumps(validation_report, indent=2))


if __name__ == "__main__":
    main()
