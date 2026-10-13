import hashlib
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List

import os
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("ANALYTICAL_DB")

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "Data" / "database_snapshot.json"
OUTPUT_FILE = BASE_DIR / "Data" / "semantic_kb.json"

MODEL_NAME = os.getenv("SQL_MODEL")

MAX_SAMPLE_ROWS = 15

# Parallel LLM calls (tables are independent). Ollama Cloud
# rejects ~6 concurrent requests with 429; 3 leaves headroom.
MAX_WORKERS = int(os.getenv("KB_WORKERS", "3"))

# gpt-oss reasoning effort: low is enough for table naming
REASONING = os.getenv("KB_REASONING", "low")

# Seconds before a single LLM call is abandoned and retried
REQUEST_TIMEOUT = float(os.getenv("KB_TIMEOUT", "120"))


# ============================================================
# PYDANTIC STRUCTURED OUTPUT
# ============================================================

class SemanticTable(BaseModel):

    business_name: str = Field(
        description=(
            "Specific human-readable business entity represented "
            "by the table."
        )
    )

    business_meaning: str = Field(
        description=(
            "Explain what the table represents and what one row "
            "represents."
        )
    )

    business_terms: List[str] = Field(
        default_factory=list,
        description=(
            "Useful business terminology users may use for this "
            "entity. Do not use raw column names."
        )
    )

    aliases: List[str] = Field(
        default_factory=list,
        description=(
            "Alternative names for the same business entity."
        )
    )

    likely_user_phrases: List[str] = Field(
        default_factory=list,
        description=(
            "Realistic natural-language questions users might ask "
            "about this entity."
        )
    )

    related_business_concepts: List[str] = Field(
        default_factory=list,
        description=(
            "Closely related business concepts supported by the "
            "available evidence."
        )
    )

    evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Short evidence statements supporting the classification."
        )
    )

    confidence: float = Field(
        ge=0,
        le=100,
        description=(
            "Confidence from 0 to 100 based on evidence strength."
        )
    )


# ============================================================
# OLLAMA
# ============================================================

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    reasoning=REASONING,
    # Cloud calls can hang with no reply; fail and retry instead
    client_kwargs={"timeout": REQUEST_TIMEOUT}
)


# ============================================================
# LOAD SNAPSHOT
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    database_snapshot = json.load(f)


schemas = database_snapshot.get(
    "schemas",
    []
)


# ============================================================
# SAMPLE ROWS
# ============================================================

def sample_rows(
    rows,
    limit=MAX_SAMPLE_ROWS
):

    if not rows:
        return []

    if len(rows) <= limit:
        return rows

    # Beginning
    first = rows[:5]

    # Middle
    middle_start = max(
        0,
        len(rows) // 2 - 2
    )

    middle = rows[
        middle_start:
        middle_start + 5
    ]

    # End
    last = rows[-5:]

    return first + middle + last


# ============================================================
# FOREIGN KEY EXTRACTION
# ============================================================

def extract_foreign_keys(schema_info):
    """
    Extract foreign-key relationships directly from CREATE TABLE
    schema text. This is deterministic database metadata and does
    not use the LLM.
    """
    if not schema_info:
        return []

    foreign_keys = []

    pattern = re.compile(
        r"FOREIGN\s+KEY\s*\(\s*([^)]+?)\s*\)"
        r"\s+REFERENCES\s+"
        r"([A-Za-z0-9_.\"`]+)"
        r"\s*\(\s*([^)]+?)\s*\)",
        re.IGNORECASE,
    )

    for match in pattern.finditer(str(schema_info)):
        foreign_keys.append(
            {
                "column": match.group(1).strip().strip('"`'),
                "references_table": match.group(2).strip().strip('"`').split(".")[-1],
                "references_column": match.group(3).strip().strip('"`'),
            }
        )

    return foreign_keys


# ============================================================
# COLUMN EXTRACTION
# ============================================================

COLUMN_LINE = re.compile(
    r'^\s*"?([A-Za-z_][A-Za-z0-9_]*)"?\s+([A-Z][A-Z0-9_ ]*(?:\([^)]*\))?(?:\[\])?)',
)


def parse_columns(schema_info):
    """(column_name, declared_type) pairs from the CREATE TABLE text,
    in declaration order."""

    body = str(schema_info).split("/*")[0]
    columns = []

    for line in body.splitlines():
        if re.match(r"^\s*(CONSTRAINT|PRIMARY|FOREIGN|UNIQUE|CHECK)\b", line, re.IGNORECASE):
            continue
        match = COLUMN_LINE.match(line)
        if match and not line.lstrip().upper().startswith("CREATE"):
            declared = re.sub(r"\s+NOT\s+NULL$", "", match.group(2).strip())
            columns.append((match.group(1), declared.strip()))

    return columns


def load_live_column_types():
    """{(schema, table): {column: data_type}} from the analytical DB the
    chatbot actually queries. Source-DB snapshot types can differ from
    it (e.g. a TIMESTAMP stored as epoch BIGINT), so live types win."""

    if not DATABASE_URL:
        print("WARNING: ANALYTICAL_DB not set; using snapshot column types")
        return {}

    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT table_schema, table_name, column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
            )).all()
    except Exception as e:
        print(f"WARNING: could not read live column types ({e}); "
              "using snapshot column types")
        return {}

    live = {}
    for schema, table, column, data_type in rows:
        live.setdefault((schema, table), {})[column] = data_type
    return live


LIVE_TYPES = {}

# {(schema, table): {column: "text" | "file"}} - see load_large_object_columns
LARGE_OBJECTS = {}

LO_SAMPLE_SIZE = 20


def load_large_object_columns():
    """Columns whose analytical-DB values are PostgreSQL large-object
    OIDs, not content. JPA @Lob fields are stored by Hibernate as a
    large object; the CDC copy in the analytical DB only carries the OID
    number. Detected, not listed:

    - source declares the column OID              -> "file" (binary)
    - source declares TEXT, every non-null value in the analytical DB
      is a whole number, and a sample of those numbers all exist in the
      source's pg_largeobject_metadata           -> "text"
    Read-only throughout."""

    names = [n.strip() for n in (os.getenv("NAMES") or "").split(",") if n.strip()]
    sources = {
        name: os.getenv(f"DATABASE_URL{i}")
        for i, name in enumerate(names) if os.getenv(f"DATABASE_URL{i}")
    }
    if not DATABASE_URL or not sources:
        print("WARNING: ANALYTICAL_DB/NAMES/DATABASE_URLn not set; "
              "large-object columns not detected")
        return {}

    from sqlalchemy import create_engine, text

    found = {}
    analytical = create_engine(DATABASE_URL)

    for schema in schemas:
        schema_name = schema.get("schema_name")
        candidates = []
        for table in schema.get("tables", []):
            for column, declared in parse_columns(table.get("schema_info") or ""):
                kind = declared.split("(")[0].strip().upper()
                if kind == "OID":
                    found.setdefault((schema_name, table["table_name"]), {})[column] = "file"
                elif kind == "TEXT":
                    candidates.append((table["table_name"], column))

        if not candidates or schema_name not in sources:
            continue

        try:
            with analytical.connect() as a_conn, \
                    create_engine(sources[schema_name]).connect() as s_conn:
                a_conn.execute(text("SET TRANSACTION READ ONLY"))
                s_conn.execute(text("SET TRANSACTION READ ONLY"))

                for table_name, column in candidates:
                    ref = f'"{schema_name}"."{table_name}"'
                    col = f'"{column}"'
                    all_numeric, sample = a_conn.execute(text(
                        f"SELECT bool_and({col} ~ '^[0-9]+$'), "
                        f"(array_agg({col}) FILTER (WHERE {col} ~ '^[0-9]+$'))[1:{LO_SAMPLE_SIZE}] "
                        f"FROM {ref} WHERE {col} IS NOT NULL AND {col} <> ''"
                    )).one()
                    if not all_numeric or not sample:
                        continue
                    oids = [int(v) for v in sample]
                    existing = s_conn.execute(text(
                        "SELECT count(*) FROM pg_largeobject_metadata WHERE oid = ANY(:oids)"
                    ), {"oids": oids}).scalar()
                    if existing == len(set(oids)):
                        found.setdefault((schema_name, table_name), {})[column] = "text"
        except Exception as e:
            print(f"WARNING: large-object scan failed for schema {schema_name}: {e}")

    print(f"Large-object columns: {sum(len(c) for c in found.values())} "
          f"across {len(found)} tables")
    return found

DATE_DECLARED = re.compile(r"^(TIMESTAMP|DATE)", re.IGNORECASE)


def build_columns(schema_name, table_name, schema_info):
    """Deterministic column list: names from the schema, types from the
    live analytical DB. Columns absent from the live table are dropped
    (they can't be queried). Declared dates stored as numbers are
    labelled with their encoding."""

    live = LIVE_TYPES.get((schema_name, table_name))
    large_objects = LARGE_OBJECTS.get((schema_name, table_name), {})
    columns = []

    for name, declared in parse_columns(schema_info):

        if live is None:
            column = {"column_name": name, "data_type": declared}
        else:
            live_type = live.get(name)
            if live_type is None:
                continue

            if DATE_DECLARED.match(declared) and live_type == "bigint":
                live_type = "bigint (epoch microseconds)"
            elif DATE_DECLARED.match(declared) and live_type == "integer":
                live_type = "integer (days since 1970-01-01)"

            column = {"column_name": name, "data_type": live_type}

        if name in large_objects:
            column["large_object"] = large_objects[name]

        columns.append(column)

    return columns


def extract_primary_keys(schema_info):
    """[{"column": ...}] from the CREATE TABLE's PRIMARY KEY constraint
    (lms_chatbot reads this to label lookup rows and target FK joins)."""
    match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", str(schema_info), re.IGNORECASE)
    if not match:
        return []
    return [
        {"column": col.strip().strip('"`')}
        for col in match.group(1).split(",") if col.strip()
    ]


def deterministic_fields(schema_name, table_name, schema_info):
    """Database identity, columns and FKs - never from the LLM."""

    foreign_keys = extract_foreign_keys(schema_info)
    for fk in foreign_keys:
        fk["references_table"] = f"{schema_name}.{fk['references_table']}"

    return {
        "schema_name": schema_name,
        "table_name": table_name,
        "qualified_name": f"{schema_name}.{table_name}",
        "columns": build_columns(schema_name, table_name, schema_info),
        "primary_keys": extract_primary_keys(schema_info),
        "foreign_keys": foreign_keys,
    }


# ============================================================
# SAVE KB
# ============================================================

def save_kb(semantic_tables):

    add_cross_table_fields(semantic_tables)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "tables": semantic_tables
            },
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    schema_name,
    table_name,
    schema_info,
    rows
):

    return f"""Describe what this database table stores, in business terms.

Table: {schema_name}.{table_name}

Schema:
{schema_info}

Sample rows:
{rows}

Guidelines:
- Decide what ONE row represents, mainly from the sample data and columns, not the table name alone.
- Expand abbreviations only when the data supports it.
- Be specific; do not invent anything not supported by the schema or data.

Reply with ONLY a JSON object, no markdown, in exactly this shape:
{{
  "business_name": "short specific name of what one row is",
  "business_meaning": "1-2 sentences: what the table holds and what one row is",
  "business_terms": ["3-6 business words users would use (not column names)"],
  "aliases": ["2-4 other names for the same thing"],
  "likely_user_phrases": ["2-4 realistic user questions"],
  "related_business_concepts": ["2-5 related concepts"],
  "evidence": ["short facts from the schema/data supporting this"],
  "confidence": 0-100 as a single number
}}"""


# ============================================================
# LLM CALL (manual JSON parse; gpt-oss ignores the JSON format
# constraint and may wrap/format its reply)
# ============================================================

def extract_json(text):

    match = re.search(r"\{.*\}", text or "", re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object in reply: {text[:200]!r}")

    return json.loads(match.group(0))


def invoke_with_backoff(prompt, attempts=6):
    """Ollama Cloud returns 429 when too many requests run at once,
    and sometimes hangs; wait and retry instead of failing the table.
    Timeouts get fewer retries so a cloud outage fails fast."""

    timeouts = 0

    for attempt in range(attempts):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            timed_out = "timed out" in str(e).lower() or "timeout" in type(e).__name__.lower()
            timeouts += timed_out
            if not (timed_out or "429" in str(e)) or timeouts > 1 or attempt == attempts - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 3) + random.uniform(0, 2))


def call_llm(prompt, retries=2):

    last_error = None

    for _ in range(retries + 1):

        reply = invoke_with_backoff(prompt)

        try:
            return SemanticTable.model_validate(extract_json(reply))

        except Exception as e:
            last_error = e
            prompt += (
                "\n\nYour previous reply was invalid "
                f"({str(e)[:150]}). Reply with ONLY the JSON object."
            )

    raise last_error


# ============================================================
# PROCESS ONE TABLE
# ============================================================

def schema_hash(schema_info, rows):
    """Fingerprint of a table's schema + row count; a table is
    regenerated only when this changes."""
    key = f"{schema_info}|{len(rows or [])}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def process_table(schema_name, table_info):
    """Return (qualified_name, data) or raise with a skip reason."""

    table_name = table_info.get("table_name")
    qualified_name = f"{schema_name}.{table_name}"

    if "error" in table_info:
        raise ValueError(f"extraction error: {table_info['error']}")

    schema_info = table_info.get("schema_info")
    rows = table_info.get("actual_data")

    if not schema_info:
        raise ValueError("missing schema_info")

    if rows is None:
        raise ValueError("missing actual_data")

    if len(rows) == 0:
        result = SemanticTable(
            business_name="Unknown",
            business_meaning=(
                "The table contains no rows in the available snapshot, "
                "so its business meaning cannot be reliably determined "
                "from actual data."
            ),
            evidence=["No actual rows were available."],
            confidence=0
        )
    else:
        prompt = build_prompt(
            schema_name,
            table_name,
            schema_info,
            sample_rows(rows)
        )
        result = call_llm(prompt)

    data = result.model_dump()
    data.update(deterministic_fields(schema_name, table_name, schema_info))
    data["schema_hash"] = schema_hash(schema_info, rows)

    return qualified_name, data


# ============================================================
# LOAD EXISTING KB (incremental rebuild)
# ============================================================

def load_existing_kb():
    if not OUTPUT_FILE.exists():
        return {}
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            tables = json.load(f).get("tables", [])
    except (OSError, json.JSONDecodeError):
        return {}
    return {t["qualified_name"]: t for t in tables if t.get("qualified_name")}


# ============================================================
# CROSS-TABLE FIELDS (incoming FKs, lookup vs business role)
# ============================================================

AUDIT_COLUMNS = {"created_by", "created_date", "last_modified_by", "last_modified_date"}
LABEL_COLUMNS = {"name", "label", "value", "title", "code"}
LOOKUP_MAX_COLUMNS = 9
# A table made only of these is a code list even if nothing references it
CODE_LIST_COLUMNS = {"id", "name", "description", "code", "label", "value", "sequence_no", "display_order"}

LOOKUP_ROLE = "Reference table (lookup: id -> label)"
BUSINESS_ROLE = "Business entity table"


def add_cross_table_fields(tables):
    """Deterministic fields that depend on the whole KB:

    - incoming_foreign_keys: declared FKs from other tables into this one.
    - table_role: lookup vs business table, used by
      lms_chatbot/retrieval/classify.py. A lookup is referenced by
      another table (declared FK, or a same-schema `<table>_id` column -
      some schemas declare no FKs), has a label column, is small once
      audit columns are ignored, holds no period dates or files, and
      only points at other lookups (or tables outside the KB)."""

    by_name = {t["qualified_name"]: t for t in tables}

    incoming = {}
    for t in tables:
        for fk in t.get("foreign_keys") or []:
            incoming.setdefault(fk["references_table"], []).append({
                "table": t["qualified_name"],
                "column": fk["column"],
                "references_column": fk["references_column"],
            })

    id_columns = {}
    for t in tables:
        for c in t.get("columns") or []:
            id_columns.setdefault(t["schema_name"], set()).add(c["column_name"])

    def referenced(t):
        return (
            t["qualified_name"] in incoming
            or f"{t['table_name']}_id" in id_columns.get(t["schema_name"], set())
        )

    def lookup_shaped(t):
        cols = [
            c for c in t.get("columns") or []
            if c["column_name"] not in AUDIT_COLUMNS
        ]
        names = {c["column_name"] for c in cols}
        # Lookups hold labels, not business dates (period, event, ...)
        has_date = any(
            re.search(r"date|timestamp", c["data_type"], re.IGNORECASE)
            or c["column_name"].endswith(("_date", "_time"))
            for c in cols
        )
        has_file = any("attachment" in n or "file" in n for n in names)
        return (
            bool(LABEL_COLUMNS & names)
            and len(cols) <= LOOKUP_MAX_COLUMNS
            and not has_date
            and not has_file
        )

    lookups = set()
    changed = True
    while changed:
        changed = False
        for name, t in by_name.items():
            if name in lookups or not lookup_shaped(t):
                continue
            targets = [fk["references_table"] for fk in t.get("foreign_keys") or []]
            if not all(target in lookups or target not in by_name for target in targets):
                continue
            # Referenced by another table, or (e.g. a status table whose
            # referrers live in another database) categorised only by
            # known lookups
            categorised = bool(targets) and all(target in lookups for target in targets)
            code_list = not targets and {
                c["column_name"] for c in t.get("columns") or []
            } - AUDIT_COLUMNS <= CODE_LIST_COLUMNS
            if referenced(t) or categorised or code_list:
                lookups.add(name)
                changed = True

    for name, t in by_name.items():
        t["incoming_foreign_keys"] = incoming.get(name, [])
        t["table_role"] = LOOKUP_ROLE if name in lookups else BUSINESS_ROLE

    return tables


# ============================================================
# MAIN
# ============================================================

def main():

    LIVE_TYPES.update(load_live_column_types())
    LARGE_OBJECTS.update(load_large_object_columns())

    existing = load_existing_kb()

    all_tables = [
        (schema.get("schema_name"), table)
        for schema in schemas
        for table in schema.get("tables", [])
    ]
    order = [f"{s}.{t.get('table_name')}" for s, t in all_tables]

    kb = {}
    todo = []

    for schema_name, table_info in all_tables:
        qualified_name = f"{schema_name}.{table_info.get('table_name')}"
        old = existing.get(qualified_name)
        if old and old.get("schema_hash") == schema_hash(
            table_info.get("schema_info"), table_info.get("actual_data")
        ):
            # LLM text is reused; columns/FKs are cheap, always refresh
            old.update(deterministic_fields(
                schema_name,
                table_info.get("table_name"),
                table_info.get("schema_info"),
            ))
            kb[qualified_name] = old
        else:
            todo.append((schema_name, table_info))

    print("=" * 60)
    print("SEMANTIC KB GENERATION")
    print("=" * 60)
    print(f"Model: {MODEL_NAME} (reasoning={REASONING})")
    print(f"Total tables: {len(all_tables)}")
    print(f"Unchanged (reused): {len(kb)}")
    print(f"To generate: {len(todo)} | workers: {MAX_WORKERS}")
    print("=" * 60)

    lock = threading.Lock()

    def save():
        save_kb([kb[q] for q in order if q in kb])

    start_time = datetime.now()
    done = 0
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_table, schema_name, table_info):
                f"{schema_name}.{table_info.get('table_name')}"
            for schema_name, table_info in todo
        }

        for future in as_completed(futures):
            qualified_name = futures[future]
            done += 1
            prefix = f"[{done}/{len(todo)}] {qualified_name}"

            try:
                _, data = future.result()
            except Exception as e:
                failed.append(qualified_name)
                print(f"{prefix} FAILED: {str(e)[:200]}")
                continue

            with lock:
                kb[qualified_name] = data
                # Save after every table so a crash/quota stop keeps progress
                save()

            print(
                f"{prefix} -> {data['business_name']} "
                f"({data['confidence']:.0f})"
            )

    save()

    total_seconds = (datetime.now() - start_time).total_seconds()
    low = [q for q in order if q in kb and 0 < kb[q]["confidence"] < 70]

    print("\n" + "=" * 60)
    print("SEMANTIC KB COMPLETE")
    print("=" * 60)
    print(f"In KB: {len(kb)}/{len(all_tables)}")
    print(f"Generated: {len(todo) - len(failed)} | Failed: {len(failed)}")
    for q in failed:
        print(f"  failed: {q}  (rerun to retry)")
    print(f"Low confidence (<70), review these: {len(low)}")
    for q in low:
        print(f"  {q}: {kb[q]['business_name']} ({kb[q]['confidence']:.0f})")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total time: {total_seconds:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
