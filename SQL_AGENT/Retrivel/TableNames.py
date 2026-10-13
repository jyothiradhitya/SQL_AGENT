import os
import json

from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("ANALYTICAL_DB")

BASE_DIR = Path(__file__).resolve().parent.parent

SCHEMA_FILE = BASE_DIR / "Data" / "schema_tables.json"


if not DATABASE_URL:
    raise ValueError(
        "ANALYTICAL_DB is not defined in .env"
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

db = SQLDatabase.from_uri(
    DATABASE_URL
)

engine = create_engine(
    DATABASE_URL
)

inspector = inspect(engine)


# ============================================================
# DISCOVER ALL SCHEMAS
# ============================================================

schema_tables = {}

schemas = inspector.get_schema_names()


# ============================================================
# PROCESS SCHEMAS
# ============================================================

for schema in schemas:

    # Skip PostgreSQL/system schemas
    if schema in [
        "information_schema",
        "pg_catalog"
    ]:
        continue

    tables = inspector.get_table_names(
        schema=schema
    )

    if not tables:
        continue

    schema_tables[schema] = tables

    print(
        f"{schema}: {len(tables)} tables"
    )


# ============================================================
# SAVE
# ============================================================

with open(
    SCHEMA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        schema_tables,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SUMMARY
# ============================================================

total_tables = sum(
    len(tables)
    for tables in schema_tables.values()
)

print()
print("=" * 60)
print("TABLE DISCOVERY COMPLETE")
print("=" * 60)
print(
    f"Schemas: {len(schema_tables)}"
)
print(
    f"Tables: {total_tables}"
)
print(
    f"Output: {SCHEMA_FILE}"
)
print("=" * 60)