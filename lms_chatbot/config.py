"""
Single source of truth for configuration.

Every other module reads settings from the `settings` singleton below
instead of calling os.getenv() directly, so there is exactly one place
that knows about environment variable names and defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is not defined in .env")
    return value


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _source_db_urls() -> dict[str, str]:
    """Schema name -> its source (application) database URL. NAMES lists
    the schemas in the same order as DATABASE_URL0, DATABASE_URL1, ... -
    the convention SQL_AGENT/Retrivel/extract_database_snapshot.py uses.
    Optional: without it, large-object text just can't be resolved."""
    names = [n.strip() for n in (os.getenv("NAMES") or "").split(",") if n.strip()]
    urls = {}
    for index, name in enumerate(names):
        url = os.getenv(f"DATABASE_URL{index}")
        if url:
            urls[name] = url
    return urls


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------
    ollama_url: str = field(default_factory=lambda: _require("OLLAMA_URL"))

    orchestrator_model: str = field(
        default_factory=lambda: os.getenv("ORCHESTRATOR_MODEL", "qwen2.5:3b")
    )
    sql_model: str = field(
        default_factory=lambda: os.getenv("SQL_MODEL", "qwen2.5:3b")
    )
    nlp_model: str = field(
        default_factory=lambda: os.getenv("NLP_MODEL", "qwen2.5:3b")
    )
    embed_model: str = field(
        default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text")
    )

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    analytical_db_url: str = field(
        default_factory=lambda: _require("ANALYTICAL_DB")
    )

    default_row_limit: int = field(
        default_factory=lambda: _int("DEFAULT_ROW_LIMIT", 100)
    )
    max_row_limit: int = field(default_factory=lambda: _int("MAX_ROW_LIMIT", 500))
    statement_timeout_ms: int = field(
        default_factory=lambda: _int("STATEMENT_TIMEOUT_MS", 10_000)
    )

    # The analytical DB is a CDC copy: @Lob text arrives there as a
    # PostgreSQL large-object OID, while the content itself only exists
    # in the source application databases. Read-only lookups resolve it
    # (sql/large_objects.py).
    source_db_urls: dict[str, str] = field(default_factory=_source_db_urls)
    # Longest resolved text passed to the model per value
    large_object_max_chars: int = field(
        default_factory=lambda: _int("LARGE_OBJECT_MAX_CHARS", 2000)
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    # Reused from the existing SQL_AGENT extraction pipeline rather than
    # copied, so there is one source of truth for the curated schema KB.
    semantic_kb_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "SEMANTIC_KB_PATH",
                str(REPO_ROOT / "SQL_AGENT" / "Data" / "semantic_kb.json"),
            )
        )
    )
    # Real sample rows captured from the DB by the old SQL_AGENT pipeline -
    # see retrieval/sample_values.py for why lookup tables render these.
    database_snapshot_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "DATABASE_SNAPSHOT_PATH",
                str(REPO_ROOT / "SQL_AGENT" / "Data" / "database_snapshot.json"),
            )
        )
    )
    table_embeddings_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "TABLE_EMBEDDINGS_PATH",
                str(PACKAGE_ROOT / "data" / "table_embeddings.json"),
            )
        )
    )

    # Deterministic FK graph built offline from semantic_kb.json by
    # retrieval/graph_builder.py - see that module for how it's used.
    fk_graph_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "FK_GRAPH_PATH",
                str(PACKAGE_ROOT / "data" / "fk_graph.json"),
            )
        )
    )

    retrieval_top_n_anchors: int = field(
        default_factory=lambda: _int("RETRIEVAL_TOP_N_ANCHORS", 6)
    )
    # Anchors scoring more than this far below the best anchor's cosine
    # similarity are dropped, even inside the top-N - a fixed top-N always
    # filled up with barely-related tables on narrow questions, and the
    # FK bridges between those junk anchors then consumed the whole
    # max_tables budget (observed: a job-codes question kept 6 anchors +
    # 6 bridges and trimmed every lookup, including the department table
    # it needed). The best anchor is always kept.
    retrieval_anchor_score_margin: float = field(
        default_factory=lambda: float(os.getenv("RETRIEVAL_ANCHOR_SCORE_MARGIN", "0.1"))
    )
    retrieval_max_tables: int = field(
        default_factory=lambda: _int("RETRIEVAL_MAX_TABLES", 14)
    )
    retrieval_bridge_max_hops: int = field(
        default_factory=lambda: _int("RETRIEVAL_BRIDGE_MAX_HOPS", 4)
    )

    # ------------------------------------------------------------------
    # Agent behaviour
    # ------------------------------------------------------------------
    # Safety-valve cap on the SQL agent's tool-calling loop (agents/sql_agent.py) -
    # not a retry policy: the loop has no Python-authored logic left deciding
    # what the model should do next turn, this just bounds how many turns it
    # can take before the agent gives up.
    sql_max_attempts: int = field(
        default_factory=lambda: _int("SQL_MAX_ATTEMPTS", 6)
    )

    # How many LLM turns the orchestrator gets before giving up - each
    # sql_agent_tool call consumes one turn, plus one turn for the final
    # reply, so this bounds how many separate database lookups a single
    # question can chain through, not just how many times it can retry
    # one lookup.
    orchestrator_max_turns: int = field(
        default_factory=lambda: _int("ORCHESTRATOR_MAX_TURNS", 6)
    )

    # How many prior (question, answer) turns the orchestrator is shown
    # as conversation history - see orchestrator.py's answer_question.
    conversation_history_turns: int = field(
        default_factory=lambda: _int("CONVERSATION_HISTORY_TURNS", 5)
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
