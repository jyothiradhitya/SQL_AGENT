"""
langchain_community.utilities.sql_database.SQLDatabase only reflects
ONE schema: its `schema=` constructor argument is used verbatim in
`inspector.get_table_names(schema=schema)`, and every lookup afterwards
(`get_table_info`, `get_usable_table_names`, ...) matches tables by
their bare `table.name`. That is fine for a single-schema database, but
this database spreads its ~49 tables across several schemas (masters,
lms, pc, ...), so a single `SQLDatabase` would only ever see one of
them.

MultiSchemaSQLDatabase reflects every requested schema into one shared
MetaData object and represents each table to the agent by its
schema-qualified name ("masters.party", "lms.training", ...) so the
raw agent - not this wrapper - has to decide which schema/table a
question refers to. That ambiguity is exactly what this experiment is
measuring, so it must not be hidden or pre-resolved here.

Query execution (`run` / `_execute`) is untouched: it just executes
whatever fully-qualified SQL the agent writes, so no schema-search-path
tricks are needed there.
"""

from __future__ import annotations

import logging

from sqlalchemy import MetaData, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql.sqltypes import NullType

from langchain_community.utilities.sql_database import SQLDatabase

logger = logging.getLogger(__name__)

# Schemas that are never part of the application's own data model.
_SYSTEM_SCHEMAS = {"information_schema", "pg_catalog", "pg_toast"}


class MultiSchemaSQLDatabase(SQLDatabase):
    """SQLDatabase variant that reflects tables from several schemas."""

    def __init__(
        self,
        engine: Engine,
        schemas: list[str],
        sample_rows_in_table_info: int = 3,
        max_string_length: int = 300,
    ):
        if not schemas:
            raise ValueError("MultiSchemaSQLDatabase requires at least one schema")

        self._engine = engine
        # Never used to set a session search_path (see module docstring) -
        # every table is queried through its own schema-qualified name.
        self._schema = None
        self._schemas = list(schemas)
        self._inspector = inspect(engine)

        all_tables: set[str] = set()
        for schema in self._schemas:
            for name in self._inspector.get_table_names(schema=schema):
                all_tables.add(f"{schema}.{name}")

        if not all_tables:
            raise ValueError(f"No tables found in schemas: {self._schemas}")

        self._all_tables = all_tables
        self._include_tables: set[str] = set()
        self._ignore_tables: set[str] = set()
        self._usable_tables = self._all_tables

        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._indexes_in_table_info = False
        self._custom_table_info = None
        self._max_string_length = max_string_length
        self._view_support = False

        self._metadata = MetaData()
        for schema in self._schemas:
            bare_names = [
                qualified.split(".", 1)[1]
                for qualified in self._usable_tables
                if qualified.startswith(f"{schema}.")
            ]
            if bare_names:
                self._metadata.reflect(bind=engine, schema=schema, only=bare_names)

        logger.info(
            "MultiSchemaSQLDatabase reflected %d tables across schemas=%s",
            len(self._all_tables), self._schemas,
        )

    @staticmethod
    def _qualified_name(table) -> str:
        return f"{table.schema}.{table.name}" if table.schema else table.name

    def get_table_info(self, table_names: list[str] | None = None) -> str:
        all_table_names = list(self.get_usable_table_names())
        if table_names is not None:
            missing = set(table_names) - set(all_table_names)
            if missing:
                raise ValueError(f"table_names {missing} not found in database")
            all_table_names = table_names

        wanted = set(all_table_names)
        meta_tables = [
            table
            for table in self._metadata.sorted_tables
            if self._qualified_name(table) in wanted
        ]

        tables = []
        for table in meta_tables:
            for col_name, column in list(table.columns.items()):
                if type(column.type) is NullType:
                    table._columns.remove(column)

            table_info = str(CreateTable(table).compile(self._engine)).rstrip()

            if self._sample_rows_in_table_info:
                table_info += f"\n\n/*\n{self._get_sample_rows(table)}\n*/"

            tables.append(table_info)

        tables.sort()
        return "\n\n".join(tables)

    @classmethod
    def from_engine(
        cls,
        engine: Engine,
        schemas: list[str] | None = None,
        **kwargs,
    ) -> "MultiSchemaSQLDatabase":
        if schemas is None:
            inspector = inspect(engine)
            schemas = [
                s for s in inspector.get_schema_names() if s not in _SYSTEM_SCHEMAS
            ]
            logger.info("Auto-discovered schemas: %s", schemas)
        return cls(engine, schemas=schemas, **kwargs)
