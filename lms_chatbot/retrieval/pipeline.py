"""
Anchor -> Bridge -> Lookup schema retrieval.

get_relevant_schema(question) turns a natural-language question into a
bounded, joinable slice of the schema:

1. Anchor selection - embed the question, cosine-match against
   precomputed anchor-table embeddings (retrieval.build_index), take
   the top N.
2. Bridge resolution - if more than one anchor came back, find the
   tables on the *shortest* FK path connecting each pair of anchors
   (fk_graph.bridge_tables), not everything N hops out.
3. Lookup attachment - for every anchor + bridge table, pull in its
   direct FK targets that are lookup/reference tables only
   (fk_graph.attach_lookups) - the tables needed to turn an *_id
   column into something a human, or the SQL agent, can filter on by
   name.

Bridge/lookup discovery both walk the deterministic FK graph built
offline by retrieval/graph_builder.py (fk_graph.json), not the raw
semantic_kb.json foreign_keys[] field directly - that field is
frequently wrong (a "public.<table>" placeholder instead of the
target's real schema) and sometimes genuinely ambiguous by bare name.
graph_builder.py already resolved what's resolvable (including a
same-schema tie-break for the few genuinely ambiguous cases in this
KB) and logged what isn't - see that module's docstring.
fk_graph.json must exist for SchemaRetriever to construct at all
(fails fast with the command to run otherwise) - retrieval/graph.py's
older, more permissive graph (which does not do that resolution) is
superseded, not a silent fallback.

This is exposed to the SQL agent as an MCP tool (mcp_server/server.py)
rather than injected once as a static prompt: if the agent's first
attempt at SQL fails because a table it needed wasn't in the slice, it
can call this again with a refined question, or the caller can force
specific tables in via `extra_tables` - see agents/sql_agent.py's retry
loop.
"""

import json
import logging
from dataclasses import dataclass, field

from lms_chatbot.config import settings
from lms_chatbot.retrieval.build_index import _embedding_text
from lms_chatbot.retrieval.classify import (
    is_identity_table,
    is_lookup_table,
    read_kb_tables,
    split_tables,
    table_name,
)
from lms_chatbot.retrieval.context import render_context
from lms_chatbot.retrieval.embeddings import get_embedder, rank_by_similarity
from lms_chatbot.retrieval.fk_graph import attach_lookups, bridge_tables, load_graph
from lms_chatbot.retrieval.shortcuts import (
    build_shortcut_index,
    expanded_question,
    render_shortcuts,
    resolve_shortcuts,
)
from lms_chatbot.retrieval.sample_values import (
    attach_parent_labels_live,
    load_enum_like_values,
    load_enum_like_values_live,
    load_identifier_values_live,
    load_sample_values,
    load_sample_values_live,
)

logger = logging.getLogger(__name__)

# Diagnostic-only: how much of the rendered context a single log line
# shows. Never affects what is actually returned to the SQL agent.
_LOG_TRUNCATE_LEN = 4000

# See get_relevant_schema's direct-lookup promotion.
_MAX_DIRECT_LOOKUPS = 2


def _truncated(text: str, limit: int = _LOG_TRUNCATE_LEN) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, {len(text)} chars total]"


@dataclass
class RetrievalResult:
    tables: list[str] = field(default_factory=list)
    context: str = ""


class SchemaRetriever:
    def __init__(self):
        kb = read_kb_tables(settings.semantic_kb_path)

        self.kb = kb
        self.kb_lookup: dict[str, dict] = {}

        for table in kb:
            name = table_name(table)
            if name:
                self.kb_lookup[name.lower()] = table

        anchors, _lookups = split_tables(kb)
        self.anchor_names = [table_name(t) for t in anchors]

        if not settings.fk_graph_path.exists():
            raise RuntimeError(
                f"No FK graph found at {settings.fk_graph_path} - run "
                "`python -m lms_chatbot.retrieval.graph_builder` first. It builds a "
                "deterministic FK graph from semantic_kb.json (resolving ambiguous/"
                "wrong-schema FK targets instead of guessing) and is required for "
                "accurate JOIN hints."
            )

        self.graph = load_graph(settings.fk_graph_path)
        logger.info(
            "Loaded FK graph from %s: %d nodes, %d edges",
            settings.fk_graph_path, len(self.graph.nodes), len(self.graph.edges),
        )

        # Live DB first: the offline snapshot is capped at 20 rows/table
        # and goes stale (see sample_values.py's live-loader comment) -
        # it's only a fallback for when the database can't be reached.
        try:
            from sqlalchemy import create_engine

            engine = create_engine(settings.analytical_db_url, pool_pre_ping=True)
            self.sample_values = load_sample_values_live(engine, self.kb_lookup)
            self.enum_values = load_enum_like_values_live(engine, self.kb_lookup)
            attach_parent_labels_live(engine, self.sample_values, self.graph, self.kb_lookup)
            self.identifier_values = load_identifier_values_live(engine, self.kb_lookup, is_identity_table)
            engine.dispose()
            source = "live database"
        except Exception as e:  # noqa: BLE001 - degrade to snapshot, never fail startup
            logger.warning("Live value load failed (%s) - falling back to snapshot", e)
            self.sample_values = load_sample_values(settings.database_snapshot_path, self.kb_lookup)
            self.enum_values = load_enum_like_values(settings.database_snapshot_path, self.kb_lookup)
            self.identifier_values = {}
            source = str(settings.database_snapshot_path)
        logger.info(
            "Loaded real sample values for %d lookup tables and enum-like values for "
            "%d non-lookup tables from %s",
            len(self.sample_values), len(self.enum_values), source,
        )

        self.shortcuts = build_shortcut_index(kb, self.sample_values)
        logger.info("Derived %d shortcuts from table names and values", len(self.shortcuts))

        self.embedder = get_embedder()

        self.anchor_vectors: dict[str, list[float]] = {}
        if settings.table_embeddings_path.exists():
            with open(settings.table_embeddings_path, "r", encoding="utf-8") as f:
                self.anchor_vectors = json.load(f)
        else:
            logger.warning(
                "No table embeddings found at %s - run "
                "`python -m lms_chatbot.retrieval.build_index` first. "
                "Anchor selection will return no results until then.",
                settings.table_embeddings_path,
            )

        # Lookup tables are never anchors, but they still need a relevance
        # score: when a retrieval overflows max_tables, lookups are what
        # get trimmed, and without a score that trim was alphabetical -
        # observed dropping masters.request_type from "how many request
        # types are there?" (the one table the question was about) and
        # masters.product_store from every "job codes per department"
        # question. Embedded here rather than in build_index's offline
        # file: there are only a handful, and this way they can never go
        # stale against the KB.
        self.lookup_vectors: dict[str, list[float]] = {}
        lookup_names = [table_name(t) for t in _lookups]
        lookup_texts = [_embedding_text(t) for t in _lookups]
        pairs = [(n, t) for n, t in zip(lookup_names, lookup_texts) if n and t]
        if pairs:
            try:
                vectors = self.embedder.embed_documents([t for _, t in pairs])
                self.lookup_vectors = {n: v for (n, _), v in zip(pairs, vectors)}
            except Exception as e:  # noqa: BLE001 - degrade to unranked lookups
                logger.warning("Could not embed lookup tables (%s) - lookups stay unranked", e)

        missing = set(self.anchor_names) - set(self.anchor_vectors)
        if missing:
            logger.warning(
                "%d anchor tables have no embedding yet (rebuild the index): %s",
                len(missing),
                ", ".join(sorted(missing)),
            )

    @property
    def all_table_names(self) -> list[str]:
        """Every table known to the KB - the SECURITY allow-list for run_sql.

        Deliberately NOT narrowed to whatever a given question's
        retrieval selected: that selection is a prompt-relevance
        concern (which tables the agent gets told about), not a
        security boundary (which tables a query is allowed to touch).
        """
        return list(self.kb_lookup.keys())

    @property
    def table_columns(self) -> dict[str, set[str]]:
        """Every KB table's real column names, keyed like all_table_names -
        lets sql/validator.py tell a real column apart from a table alias
        used as a whole-row reference when both share one name."""
        return {
            name: {c["column_name"].lower() for c in table.get("columns", []) if c.get("column_name")}
            for name, table in self.kb_lookup.items()
        }

    def get_relevant_schema(
        self,
        question: str,
        top_n: int | None = None,
        max_tables: int | None = None,
        extra_tables: list[str] | None = None,
    ) -> RetrievalResult:
        top_n = top_n or settings.retrieval_top_n_anchors
        max_tables = max_tables or settings.retrieval_max_tables

        logger.info("[RETRIEVAL] RETRIEVAL_START question=%r top_n=%d", question, top_n)

        shortcuts = resolve_shortcuts(question, self.shortcuts)
        search_text = expanded_question(question, shortcuts)
        if shortcuts:
            logger.info("[RETRIEVAL] SHORTCUTS search_text=%r", search_text)

        query_vector = self.embedder.embed_query(search_text)
        ranked = rank_by_similarity(query_vector, self.anchor_vectors, top_n)
        if ranked:
            floor = ranked[0][1] - settings.retrieval_anchor_score_margin
            ranked = [ranked[0]] + [(n, s) for n, s in ranked[1:] if s >= floor]
        anchors = [name for name, _score in ranked]

        logger.info(
            "[RETRIEVAL] ANCHORS_SELECTED anchors=%s",
            [(name, round(score, 4)) for name, score in ranked],
        )
        for extra in extra_tables or []:
            if extra not in anchors:
                anchors.append(extra)

        if not anchors:
            logger.warning(
                "[RETRIEVAL] NO_ANCHORS question=%r - no table embeddings matched "
                "at all (empty/missing table_embeddings.json?)",
                question,
            )
            return RetrievalResult()

        bridges = sorted(
            bridge_tables(self.graph, anchors, settings.retrieval_bridge_max_hops)
            - {a.lower() for a in anchors}
        )
        logger.info("[RETRIEVAL] BRIDGES_SELECTED bridges=%s", bridges)

        # A lookup the question is directly ABOUT ("how many departments",
        # "list the request types") outscores even the best anchor - it's
        # promoted into the core set, so it's included even when no
        # anchor's FK reaches it and is never the thing trimmed below.
        # Capped at the top _MAX_DIRECT_LOOKUPS: lookup descriptions are
        # short and score in a systematically higher range than anchors,
        # so "beats the best anchor" alone let 5+ unrelated lookups in on
        # a plain "how many departments are there?". A lookup needed only
        # for a JOIN still arrives through attach_lookups below.
        ranked_lookups = rank_by_similarity(query_vector, self.lookup_vectors, len(self.lookup_vectors))
        lookup_scores = {name.lower(): score for name, score in ranked_lookups}
        best_anchor = max((score for _, score in ranked), default=1.0)
        direct_lookups = [
            name for name, score in ranked_lookups[:_MAX_DIRECT_LOOKUPS]
            if score >= best_anchor and name.lower() not in {a.lower() for a in anchors}
        ]
        logger.info("[RETRIEVAL] DIRECT_LOOKUPS_SELECTED lookups=%s", direct_lookups)

        core = anchors + bridges + direct_lookups
        # A promoted lookup's own FK targets are what make its rows
        # interpretable (masters.job_code.dept_id -> masters.product_store
        # is the only way to answer "job codes in Quality Control"), so
        # they go first; after them, most question-relevant first - either
        # way any trim for max_tables below drops the least relevant
        # lookups rather than whatever sorts last alphabetically.
        direct_parents = {
            n.lower() for n in attach_lookups(self.graph, self.kb_lookup, direct_lookups, is_lookup_table)
        } if direct_lookups else set()

        # Rank of the best core table that actually joins to a lookup.
        # Similarity alone can't tell same-shaped lookups apart (each
        # schema has its own `status` table): the one the top anchors
        # reference must survive a trim, not whichever reads closest.
        referrer_rank: dict[str, int] = {}
        for rank, table in enumerate(core):
            for edge in self.graph.get_neighbors(table)["outgoing"]:
                referrer_rank.setdefault(edge.target_table.lower(), rank)

        def lookup_priority(name: str):
            key = name.lower()
            return (
                key not in direct_parents,
                referrer_rank.get(key, len(core)),
                -lookup_scores.get(key, 0.0),
                name,
            )

        lookups = sorted(
            attach_lookups(self.graph, self.kb_lookup, core, is_lookup_table)
            - {c.lower() for c in core},
            key=lookup_priority,
        )
        logger.info("[RETRIEVAL] LOOKUPS_SELECTED lookups=%s", lookups)

        # Identity tables can chain (user_training -> party -> sys_user is
        # two hops), and attach_lookups only walks one hop from `tables` -
        # so keep expanding from whatever was just found until a pass
        # turns up nothing new. Bounded by the (tiny, fixed) identity
        # table allowlist, so this always terminates quickly.
        identities: set[str] = set()
        frontier = list(core)
        while frontier:
            found = attach_lookups(self.graph, self.kb_lookup, frontier, is_identity_table)
            new = {f for f in found if f.lower() not in {i.lower() for i in identities}}
            if not new:
                break
            identities |= new
            frontier = list(new)

        identities = sorted(
            identities
            - {c.lower() for c in core}
            - {name.lower() for name in lookups}
        )
        logger.info("[RETRIEVAL] IDENTITY_TABLES_SELECTED identities=%s", identities)

        # An identity table's own lookups (a person's department, status,
        # type) are how people get filtered - "trainings completed by
        # users in Warehouse" needs masters.party.department_id ->
        # masters.product_store, and party arrives here as an identity
        # table, whose lookups were never attached (only anchors' and
        # bridges' were), so the department names were missing entirely.
        if identities:
            identity_lookups = attach_lookups(self.graph, self.kb_lookup, identities, is_lookup_table)
            known = {n.lower() for n in lookups} | {c.lower() for c in core}
            lookups = sorted(
                set(lookups) | {n for n in identity_lookups if n.lower() not in known},
                key=lookup_priority,
            )

        selected: list[tuple[str, str]] = (
            [(name, "anchor") for name in anchors]
            + [(name, "bridge") for name in bridges]
            + [(name, "lookup") for name in direct_lookups]
            # Trim order below keeps a prefix, so this is priority order:
            # a promoted lookup's parents, then the identity tables (the
            # only way any record links to a person or login - trimming
            # them made every "... did <login> pass?" unanswerable), then
            # the remaining lookups by relevance.
            + [(name, "lookup") for name in lookups if name.lower() in direct_parents]
            + [(name, "identity") for name in identities]
            + [(name, "lookup") for name in lookups if name.lower() not in direct_parents]
        )

        if len(selected) > max_tables:
            core_count = len(anchors) + len(bridges) + len(direct_lookups)
            kept_lookups = max(max_tables - core_count, 0)
            logger.warning(
                "Retrieved %d tables, capping at %d (keeping all %d "
                "anchors/bridges, trimming lookups/identities to %d)",
                len(selected),
                max_tables,
                core_count,
                kept_lookups,
            )
            selected = selected[:core_count] + selected[core_count:core_count + kept_lookups]

        tables_with_roles = []
        seen = set()

        for name, role in selected:
            key = name.lower()
            entry = self.kb_lookup.get(key)

            if entry is None or key in seen:
                continue

            seen.add(key)
            tables_with_roles.append((entry, role))

        context = render_context(tables_with_roles, self.graph, self.sample_values, self.enum_values)
        shortcut_block = render_shortcuts(shortcuts)
        if shortcut_block:
            context = f"{shortcut_block}\n\n{context}"

        logger.info(
            "[RETRIEVAL] RETRIEVAL_RESULT tables=%s context_chars=%d",
            [table_name(t) for t, _ in tables_with_roles],
            len(context),
        )
        logger.debug("[RETRIEVAL] CONTEXT_SENT_TO_AGENT=\n%s", _truncated(context))

        return RetrievalResult(
            tables=[table_name(t) for t, _ in tables_with_roles],
            context=context,
        )
