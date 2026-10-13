import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SCHEMA_FILE = BASE_DIR / "Data" / "schema_tables.json"
OUTPUT_FILE = BASE_DIR / "Data" / "database_snapshot.json"

# Physical database schema
DB_SCHEMA = "public"


# ============================================================
# LOAD TABLE LIST
# ============================================================

with open(
    SCHEMA_FILE,
    "r",
    encoding="utf-8"
) as f:

    schema_tables = json.load(f)


# ============================================================
# READ LOGICAL SCHEMA NAMES
# ============================================================

names = os.getenv("NAMES")

if not names:
    raise ValueError(
        "NAMES is not defined in .env"
    )


schema_names = [
    name.strip()
    for name in names.split(",")
    if name.strip()
]


print("=" * 60)
print(f"Schemas found: {len(schema_names)}")
print(f"Logical schemas: {schema_names}")
print(f"Physical DB schema: {DB_SCHEMA}")
print("=" * 60)


# ============================================================
# FINAL SNAPSHOT
# ============================================================

database_snapshot = {
    "schemas": []
}


# ============================================================
# PROCESS ALL LOGICAL SCHEMAS
# ============================================================

for schema_index, logical_schema in enumerate(
    schema_names
):

    print()
    print("=" * 60)

    print(
        f"Schema [{schema_index + 1}/{len(schema_names)}]: "
        f"{logical_schema}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # DATABASE URL
    # --------------------------------------------------------

    db_uri = os.getenv(
        f"DATABASE_URL{schema_index}"
    )

    if not db_uri:

        print(
            f"ERROR: DATABASE_URL{schema_index} "
            f"not found for {logical_schema}"
        )

        continue


    # --------------------------------------------------------
    # CREATE LANGCHAIN DATABASE
    # --------------------------------------------------------

    try:

        db = SQLDatabase.from_uri(
            db_uri,
            schema=DB_SCHEMA
        )

        print(
            f"Database connection: OK "
            f"(physical schema={DB_SCHEMA})"
        )

    except Exception as e:

        print(
            f"ERROR connecting to "
            f"{logical_schema}"
        )

        print(e)

        continue


    # --------------------------------------------------------
    # GET TABLE LIST
    # --------------------------------------------------------

    tables = schema_tables.get(
        logical_schema,
        []
    )

    print(
        f"Tables found: {len(tables)}"
    )


    if not tables:

        print(
            f"No tables found for: "
            f"{logical_schema}"
        )

        continue


    schema_data = {
        "schema_name": logical_schema,
        "tables": []
    }


    # ========================================================
    # PROCESS TABLES
    # ========================================================

    for index, table_name in enumerate(
        tables,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(tables)}] "
            f"Processing: "
            f"{logical_schema}.{table_name}"
        )


        try:

            # ------------------------------------------------
            # GET LANGCHAIN TABLE INFORMATION
            #
            # It includes the table structure and
            # PK/FK information exposed by the database.
            # ------------------------------------------------

            schema_info = db.get_table_info(
                [table_name]
            )


            print(
                "  Schema information: OK"
            )


            # ------------------------------------------------
            # GET 20 SAMPLE ROWS
            #
            # Physical schema = public
            # ------------------------------------------------

            rows = db.run(
                f'''
                SELECT *
                FROM "{DB_SCHEMA}"."{table_name}"
                LIMIT 20
                '''
            )


            print(
                "  Sample rows: OK"
            )


            # ------------------------------------------------
            # PRESERVE SAME ROW REPRESENTATION
            #
            # Your old Ollama code passed db.run()
            # directly into the prompt.
            # Therefore store the same representation.
            # ------------------------------------------------

            actual_data = str(rows)


            # ------------------------------------------------
            # TABLE SNAPSHOT
            # ------------------------------------------------

            table_data = {

                "table_name":
                    table_name,

                "qualified_name":
                    f"{logical_schema}.{table_name}",

                "schema_info":
                    schema_info,

                "actual_data":
                    actual_data,

                "sample_row_count":
                    len(rows.splitlines())
                    if rows
                    else 0

            }


            schema_data["tables"].append(
                table_data
            )


            print(
                "  Status: OK"
            )


        except Exception as e:

            print()
            print(
                f"  ERROR: "
                f"{logical_schema}.{table_name}"
            )

            print(
                f"  {e}"
            )


            # ------------------------------------------------
            # STORE ERROR
            # ------------------------------------------------

            schema_data["tables"].append({

                "table_name":
                    table_name,

                "qualified_name":
                    f"{logical_schema}.{table_name}",

                "error":
                    str(e)

            })


    # --------------------------------------------------------
    # ADD SCHEMA
    # --------------------------------------------------------

    database_snapshot["schemas"].append(
        schema_data
    )


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        database_snapshot,
        f,
        indent=2,
        ensure_ascii=False,
        default=str
    )


# ============================================================
# SUMMARY
# ============================================================

total_tables = 0
successful_tables = 0
failed_tables = 0


for schema in database_snapshot["schemas"]:

    for table in schema["tables"]:

        total_tables += 1

        if "error" in table:

            failed_tables += 1

        else:

            successful_tables += 1


print()
print("=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Schemas: "
    f"{len(database_snapshot['schemas'])}"
)

print(
    f"Tables: "
    f"{total_tables}"
)

print(
    f"Successful: "
    f"{successful_tables}"
)

print(
    f"Failed: "
    f"{failed_tables}"
)

print(
    f"Saved: "
    f"{OUTPUT_FILE}"
)

print("=" * 60)