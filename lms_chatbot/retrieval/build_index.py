"""
Offline build step: embed every anchor table's business description once
and persist the vectors to disk.

Run this whenever semantic_kb.json changes (new tables, edited business
descriptions) or EMBED_MODEL changes:

    uv run python -m lms_chatbot.retrieval.build_index

Query time (retrieval.pipeline.SchemaRetriever) only ever embeds the
user's question and loads this file - it never re-embeds the table
descriptions, unlike the original notebook which recomputed them on
every run.
"""

import json
import logging

from lms_chatbot.config import settings
from lms_chatbot.logging_config import setup_logging
from lms_chatbot.retrieval.classify import read_kb_tables, split_tables, table_name
from lms_chatbot.retrieval.embeddings import get_embedder

logger = logging.getLogger(__name__)


def _embedding_text(table: dict) -> str:
    parts = [
        table.get("business_name", ""),
        table.get("business_meaning", ""),
        table.get("business_purpose", ""),
        " ".join(table.get("business_terms") or []),
        " ".join(table.get("likely_user_phrases") or []),
    ]
    return " ".join(part for part in parts if part).strip()


def build_index() -> dict[str, list[float]]:
    kb = read_kb_tables(settings.semantic_kb_path)

    anchors, lookups = split_tables(kb)

    logger.info(
        "Loaded %d tables from KB: %d anchors, %d lookups",
        len(kb),
        len(anchors),
        len(lookups),
    )

    names = []
    texts = []

    for table in anchors:
        text = _embedding_text(table)
        if not text:
            logger.warning(
                "Skipping anchor %s: no business description to embed",
                table_name(table),
            )
            continue

        names.append(table_name(table))
        texts.append(text)

    logger.info("Embedding %d anchor table descriptions via %s", len(texts), settings.embed_model)

    embedder = get_embedder()
    vectors = embedder.embed_documents(texts)

    index = dict(zip(names, vectors))

    settings.table_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.table_embeddings_path, "w", encoding="utf-8") as f:
        json.dump(index, f)

    logger.info("Wrote %d embeddings to %s", len(index), settings.table_embeddings_path)

    return index


if __name__ == "__main__":
    setup_logging()
    build_index()
