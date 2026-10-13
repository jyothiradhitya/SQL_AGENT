"""
Prototype: YML retrieval -> query-driven schema graph search -> SQL context.

Stage 1 (seed retrieval)
    Hybrid semantic + lexical scoring over table-level descriptions.
    Semantic always runs (handles paraphrases with no literal table/column
    names); lexical is an additive boost, never a gate (we can't expect a
    user to know exact entity names). Validated in embading.ipynb.

Stage 2 (query-driven graph search)
    Starting from the Stage-1 seeds, enumerate FK paths (bounded by
    MAX_HOPS) between every pair of seeds, score each *path* - not just
    each node - by node relevance + edge relevance + column relevance
    minus a hop penalty, and keep the best path per seed pair. This is
    exhaustive-then-score rather than beam search: at ~50 tables the
    full set of candidate paths per pair is small enough to enumerate
    directly, which is deterministic and debuggable (every candidate
    path and its score can be printed), unlike a beam that can prune the
    correct path on a bad early per-hop score with no visibility into it.

Output shape (see get_relevant_subgraph): tables / columns / joins /
paths, not just a table list - column- and join-level detail is what
lets the SQL agent resolve an id column to a human-readable value
instead of filtering on a raw id (e.g. status_id = 3) it can't explain.

Reuses the finalized graph/classify/context/embeddings modules from
lms_chatbot.retrieval; this file only prototypes the new pieces: column
and edge relevance scoring, and path-level (not just node-level) ranking.
Nothing here is wired into lms_chatbot yet - this is the pre-finalization
test bed.

Run directly:  uv run python SQL_AGENT/Vector_Search/embading.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lms_chatbot.config import settings  # noqa: E402
from lms_chatbot.retrieval import classify, context as context_mod, graph as graph_mod  # noqa: E402
from lms_chatbot.retrieval.build_index import _embedding_text  # noqa: E402
from lms_chatbot.retrieval.embeddings import cosine_similarity, get_embedder  # noqa: E402
from lms_chatbot.retrieval.graph import Edge, Graph  # noqa: E402

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

SEED_TOP_N = 6
SEED_MIN_SCORE = 0.55  # floor instead of always padding to top_n with weak matches
LEXICAL_BOOST = 0.5

MAX_HOPS = 3
HOP_PENALTY = 0.15
MAX_PATHS_PER_PAIR = 40  # safety cap on path enumeration per seed pair
TOP_COLUMNS_PER_TABLE = 8

# The `len(word) > 2` cutoff tried in the notebook silently dropped real
# 2-letter domain terms this schema uses as table prefixes (tm, ut, pc).
# The stopword list already does the noise-filtering job; length only
# needs to exclude single characters.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "in",
    "on", "to", "for", "by", "with", "all", "list", "show", "get", "find",
    "what", "which", "who", "that", "this", "these", "those", "me",
    "please", "give", "display", "tell", "want", "need", "can", "you",
    "does", "do", "did", "has", "have", "had", "not", "any",
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) >= 2 and w not in _STOPWORDS}


def lexical_score(question_tokens: set[str], bag: set[str]) -> float:
    if not question_tokens or not bag:
        return 0.0
    return len(question_tokens & bag) / len(question_tokens)


# ---------------------------------------------------------------------------
# KB loading + shared indices
# ---------------------------------------------------------------------------

def load_kb() -> list[dict[str, Any]]:
    with open(settings.semantic_kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_lexical_bag(table: dict[str, Any]) -> set[str]:
    """Literal terms a user might type that should point at this table -
    table/column names, business terms, aliases, column synonyms - as
    opposed to embedding text, which is full descriptive prose."""

    parts = []

    for key in ("business_name", "table_name", "qualified_name", "physical_qualified_name"):
        value = table.get(key)
        if value:
            parts.append(value)

    parts.extend(table.get("business_terms") or [])
    parts.extend(table.get("aliases") or [])
    parts.extend(table.get("related_business_concepts") or [])
    parts.extend(table.get("likely_user_phrases") or [])

    for column in table.get("columns") or []:
        if column.get("column_name"):
            parts.append(column["column_name"])
        parts.extend(column.get("synonyms") or [])
        parts.extend(column.get("likely_user_terms") or [])

    bag: set[str] = set()
    for part in parts:
        bag |= tokenize(part)
    return bag


def build_column_bag(column: dict[str, Any]) -> set[str]:
    parts = [column.get("column_name", "")]
    parts.extend(column.get("synonyms") or [])
    parts.extend(column.get("likely_user_terms") or [])

    desc = column.get("business_description") or column.get("likely_business_meaning") or ""
    if desc:
        parts.append(desc)

    bag: set[str] = set()
    for part in parts:
        bag |= tokenize(part)
    return bag


def score_columns(table: dict[str, Any], question_tokens: set[str]) -> list[tuple[str, float]]:
    scored = []
    for column in table.get("columns") or []:
        name = column.get("column_name")
        if not name:
            continue
        score = lexical_score(question_tokens, build_column_bag(column))
        if score > 0:
            scored.append((name, score))

    scored.sort(key=lambda row: row[1], reverse=True)
    return scored


def build_edge_metadata(kb: list[dict[str, Any]], name_index: classify.NameIndex) -> dict[tuple[str, str, str], set[str]]:
    """(from_table.lower(), fk_column, to_table.lower()) -> lexical bag for that
    FK relationship - column names + relationship_meaning + any of the
    table's useful_join_paths sentences that mention the target table.
    Registered both directions since bridging walks edges either way."""

    meta: dict[tuple[str, str, str], set[str]] = {}

    for table in kb:
        name = classify.table_name(table)
        if not name:
            continue

        join_path_sentences = table.get("useful_join_paths") or []

        for fk in table.get("foreign_keys") or []:
            fk_column = fk.get("column")
            raw_ref_table = fk.get("references_table")
            ref_column = fk.get("references_column")
            meaning = fk.get("relationship_meaning", "") or ""

            if not fk_column or not raw_ref_table or not ref_column:
                continue

            for ref_table in name_index.resolve(raw_ref_table):
                bare_ref = ref_table.split(".")[-1].lower()
                hint_text = " ".join(jp for jp in join_path_sentences if bare_ref in jp.lower())

                text = " ".join(part for part in (fk_column, ref_column, meaning, hint_text) if part)
                bag = tokenize(text)

                meta[(name.lower(), fk_column, ref_table.lower())] = bag
                meta[(ref_table.lower(), ref_column, name.lower())] = bag

    return meta


# ---------------------------------------------------------------------------
# Stage 1 - seed retrieval
# ---------------------------------------------------------------------------

@dataclass
class SchemaIndex:
    kb: list[dict[str, Any]]
    kb_lookup: dict[str, dict[str, Any]]
    anchors: list[dict[str, Any]]
    name_index: classify.NameIndex
    fk_graph: Graph
    edge_meta: dict[tuple[str, str, str], set[str]]
    embedder: Any
    anchor_vectors: dict[str, list[float]]
    lexical_bags: dict[str, set[str]] = field(default_factory=dict)

    def lexical_bag(self, table_name: str) -> set[str]:
        cached = self.lexical_bags.get(table_name)
        if cached is not None:
            return cached

        entry = self.kb_lookup.get(table_name.lower())
        bag = build_lexical_bag(entry) if entry else set()
        self.lexical_bags[table_name] = bag
        return bag


def build_index() -> SchemaIndex:
    kb = load_kb()
    anchors, _lookups = classify.split_tables(kb)

    kb_lookup = {}
    for table in kb:
        name = classify.table_name(table)
        if name:
            kb_lookup[name.lower()] = table

    name_index = classify.NameIndex(kb)
    fk_graph = graph_mod.build_fk_graph(kb, name_index)
    edge_meta = build_edge_metadata(kb, name_index)

    embedder = get_embedder()

    anchor_names, anchor_texts = [], []
    for table in anchors:
        text = _embedding_text(table)
        if text:
            anchor_names.append(classify.table_name(table))
            anchor_texts.append(text)

    anchor_vectors = dict(zip(anchor_names, embedder.embed_documents(anchor_texts)))

    return SchemaIndex(
        kb=kb,
        kb_lookup=kb_lookup,
        anchors=anchors,
        name_index=name_index,
        fk_graph=fk_graph,
        edge_meta=edge_meta,
        embedder=embedder,
        anchor_vectors=anchor_vectors,
    )


def score_seeds(index: SchemaIndex, question: str, top_n: int = SEED_TOP_N, min_score: float = SEED_MIN_SCORE) -> list[dict]:
    question_tokens = tokenize(question)
    query_vector = index.embedder.embed_query(question)

    scored = []
    for name, vector in index.anchor_vectors.items():
        semantic = cosine_similarity(query_vector, vector)
        lexical = lexical_score(question_tokens, index.lexical_bag(name))
        scored.append({
            "table": name,
            "final": semantic + LEXICAL_BOOST * lexical,
            "semantic": semantic,
            "lexical": lexical,
        })

    scored.sort(key=lambda row: row["final"], reverse=True)

    above_floor = [row for row in scored if row["final"] >= min_score][:top_n]
    return above_floor or scored[:1]  # always return at least one starting point


# ---------------------------------------------------------------------------
# Stage 2 - query-driven graph search
# ---------------------------------------------------------------------------

def enumerate_paths(fk_graph: Graph, start: str, goal: str, max_hops: int, max_paths: int = MAX_PATHS_PER_PAIR) -> list[list[Edge]]:
    """All simple paths start -> goal within max_hops edges. Exhaustive
    rather than shortest-path-only, so a 2-hop path with a much better
    relevance score can be preferred over the arbitrary 2-hop path plain
    BFS would have returned first."""

    start_key, goal_key = start.lower(), goal.lower()
    if start_key == goal_key:
        return []

    results: list[list[Edge]] = []

    def dfs(current_key: str, path_edges: list[Edge], visited: set[str]) -> None:
        if len(results) >= max_paths or len(path_edges) >= max_hops:
            return

        for edge in fk_graph.get(current_key, []):
            next_key = edge.to_table.lower()
            if next_key in visited:
                continue

            next_path = path_edges + [edge]
            if next_key == goal_key:
                results.append(next_path)
                if len(results) >= max_paths:
                    return
                continue

            dfs(next_key, next_path, visited | {next_key})

    dfs(start_key, [], {start_key})
    return results


def score_path(index: SchemaIndex, path_edges: list[Edge], question_tokens: set[str]) -> float:
    table_names = [path_edges[0].from_table] + [edge.to_table for edge in path_edges]

    node_total = 0.0
    column_bonus = 0.0
    for name in table_names:
        entry = index.kb_lookup.get(name.lower())
        if not entry:
            continue

        node_total += lexical_score(question_tokens, index.lexical_bag(name))

        scored_cols = score_columns(entry, question_tokens)
        if scored_cols:
            column_bonus += scored_cols[0][1]

    edge_total = 0.0
    for edge in path_edges:
        bag = index.edge_meta.get((edge.from_table.lower(), edge.from_column, edge.to_table.lower()), set())
        edge_total += lexical_score(question_tokens, bag)

    hop_penalty = HOP_PENALTY * max(len(path_edges) - 1, 0)

    return node_total + edge_total + column_bonus - hop_penalty


def best_paths_between_seeds(index: SchemaIndex, seed_names: list[str], question_tokens: set[str], max_hops: int = MAX_HOPS) -> list[dict]:
    """Shorter always wins over longer - relevance score only breaks ties
    among paths of the SAME hop count. Scoring path length as just another
    penalty term to be outweighed by enough accumulated node relevance
    backfired: a direct 1-hop FK would lose to a 3-hop detour through
    other relevant tables, since node relevance is summed per hop and
    keeps growing with every extra node visited."""

    chosen = []

    for a, b in combinations(seed_names, 2):
        candidates = enumerate_paths(index.fk_graph, a, b, max_hops)
        if not candidates:
            continue

        shortest_len = min(len(path) for path in candidates)
        shortest_candidates = [path for path in candidates if len(path) == shortest_len]

        scored = [(score_path(index, path, question_tokens), path) for path in shortest_candidates]
        scored.sort(key=lambda row: row[0], reverse=True)
        best_score, best_path = scored[0]

        chosen.append({
            "from": a,
            "to": b,
            "score": best_score,
            "candidates_considered": len(candidates),
            "path": best_path,
        })

    return chosen


# ---------------------------------------------------------------------------
# Assembly - Stage 1 + Stage 2 -> {tables, columns, joins, paths, context}
# ---------------------------------------------------------------------------

def get_relevant_subgraph(index: SchemaIndex, question: str, top_n: int = SEED_TOP_N,
                           min_seed_score: float = SEED_MIN_SCORE, max_hops: int = MAX_HOPS) -> dict:
    question_tokens = tokenize(question)

    seeds = score_seeds(index, question, top_n=top_n, min_score=min_seed_score)
    seed_names = [row["table"] for row in seeds]

    chosen_paths = best_paths_between_seeds(index, seed_names, question_tokens, max_hops=max_hops)

    bridge_names: set[str] = set()
    joins: list[dict] = []
    paths_out: list[list[str]] = []
    seed_keys = {name.lower() for name in seed_names}
    seen_edges: set[frozenset[str]] = set()

    for cp in chosen_paths:
        path_edges = cp["path"]
        table_seq = [path_edges[0].from_table] + [edge.to_table for edge in path_edges]
        paths_out.append(table_seq)

        for name in table_seq:
            if name.lower() not in seed_keys:
                bridge_names.add(name)

        for edge in path_edges:
            left = f"{edge.from_table}.{edge.from_column}"
            right = f"{edge.to_table}.{edge.to_column}"
            edge_key = frozenset((left, right))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            joins.append({"left": left, "right": right})

    core = seed_names + sorted(bridge_names)

    lookup_names = sorted(
        graph_mod.attach_lookups(index.kb_lookup, index.name_index, core, classify.is_lookup_table)
        - {name.lower() for name in core}
    )

    roles = (
        [(name, "seed") for name in seed_names]
        + [(name, "bridge") for name in sorted(bridge_names)]
        + [(name, "lookup") for name in lookup_names]
    )

    tables_with_roles: list[tuple[dict, str]] = []
    seen = set()
    for name, role in roles:
        key = name.lower()
        entry = index.kb_lookup.get(key)
        if entry and key not in seen:
            seen.add(key)
            tables_with_roles.append((entry, role))

    columns_out: set[str] = set()
    for entry, _role in tables_with_roles:
        name = classify.table_name(entry)

        for col_name, _score in score_columns(entry, question_tokens)[:TOP_COLUMNS_PER_TABLE]:
            columns_out.add(f"{name}.{col_name}")

        # PK/FK columns are structurally required for joins even when the
        # user's words don't lexically mention them.
        for pk in entry.get("primary_keys") or []:
            if pk.get("column"):
                columns_out.add(f"{name}.{pk['column']}")
        for fk in entry.get("foreign_keys") or []:
            if fk.get("column"):
                columns_out.add(f"{name}.{fk['column']}")

    return {
        "question": question,
        "seed_scores": seeds,
        "tables": [classify.table_name(entry) for entry, _ in tables_with_roles],
        "roles": {classify.table_name(entry): role for entry, role in tables_with_roles},
        "columns": sorted(columns_out),
        "joins": joins,
        "paths": paths_out,
        "path_scores": chosen_paths,
        "context": context_mod.render_context(tables_with_roles, index.name_index),
    }


# ---------------------------------------------------------------------------
# Manual test run
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    "who has completed their training",
    "show me tm_attachment files",
    "list all trainings and their categories",
    "which security group does a user belong to",
]


def _print_result(result: dict) -> None:
    print("=" * 100)
    print("Q:", result["question"])
    print(f"-> {len(result['tables'])} tables "
          f"({sum(1 for r in result['roles'].values() if r == 'seed')} seed, "
          f"{sum(1 for r in result['roles'].values() if r == 'bridge')} bridge, "
          f"{sum(1 for r in result['roles'].values() if r == 'lookup')} lookup)")

    print("\nSeed scores (table | final | semantic | lexical):")
    for row in result["seed_scores"]:
        print(f"  {row['table']:<30} {row['final']:.3f}   {row['semantic']:.3f}   {row['lexical']:.3f}")

    print("\nChosen bridge paths (best-scoring path per seed pair):")
    for cp in result["path_scores"]:
        path_str = " -> ".join([cp["path"][0].from_table] + [e.to_table for e in cp["path"]])
        print(f"  {cp['from']} <-> {cp['to']}: score={cp['score']:.3f} "
              f"(1 of {cp['candidates_considered']} candidates) :: {path_str}")

    print("\nRelevant columns:")
    for col in result["columns"]:
        print(" ", col)

    print("\nJoins:")
    for join in result["joins"]:
        print(f"  {join['left']} = {join['right']}")

    print("\nContext:\n")
    print(result["context"])
    print()


if __name__ == "__main__":
    schema_index = build_index()
    print(f"Indexed {len(schema_index.kb)} tables, {len(schema_index.anchor_vectors)} embedded anchors\n")

    for question in TEST_QUESTIONS:
        _print_result(get_relevant_subgraph(schema_index, question))
