"""
Resolve PostgreSQL large-object references into their real content.

JPA @Lob fields are stored by Hibernate as PostgreSQL large objects: the
row holds only an OID number, the content lives in pg_largeobject. The
analytical DB is a CDC copy that carries the OID but none of the large
objects, so a plain SELECT returns "91334" where the user wrote
"<p>proposed 1</p>". Which columns are affected is detected at KB build
time (SQL_AGENT/Retrivel/kb.py marks them `large_object: text|file`).

After a query runs, every result column that is a plain reference to
such a column is resolved here: OIDs are fetched with lo_get() from the
schema's SOURCE database in a read-only transaction, decoded as UTF-8
once the bytes are verified to be text, and HTML is reduced to plain
text. A number is never passed on as if it were the content - if the
content can't be fetched, the value says so instead.

The agent itself can't call lo_get (validator.py blocks every lo_*
function, since lo_import/lo_export touch the server filesystem), and
it can't filter on these columns in SQL either - the analytical DB has
no content to compare against. check_query() turns such an attempt into
a clear error instead of a silently-wrong empty result.
"""

import html
import json
import logging
import re
from typing import Any

from sqlalchemy import create_engine, text
from sqlglot import exp

logger = logging.getLogger(__name__)

_FILE_SIGNATURES = (
    (b"\x89PNG", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"GIF8", "GIF image"),
    (b"%PDF", "PDF document"),
    (b"PK\x03\x04", "Office/ZIP document"),
    (b"\xd0\xcf\x11\xe0", "legacy Office document"),
)


class LargeObjectResolver:
    def __init__(self, kb_tables: list[dict[str, Any]], source_urls: dict[str, str], max_chars: int = 2000):
        # (schema, table) -> {column: "text" | "file"}
        self.columns: dict[tuple[str, str], dict[str, str]] = {}
        for table in kb_tables:
            lob = {
                c["column_name"].lower(): c["large_object"]
                for c in table.get("columns") or []
                if c.get("large_object") and c.get("column_name")
            }
            if lob:
                key = ((table.get("schema_name") or "").lower(), (table.get("table_name") or "").lower())
                self.columns[key] = lob

        self.source_urls = {k.lower(): v for k, v in source_urls.items()}
        self.max_chars = max_chars
        self._engines: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Mapping query columns back to real table columns
    # ------------------------------------------------------------------

    def _aliases(self, select: exp.Select) -> dict[str, tuple[str, str]]:
        """alias/bare name -> (schema, table) for the real tables in
        this SELECT's own FROM/JOINs."""
        aliases = {}
        # sqlglot names the FROM clause "from_" (newer) or "from" (older)
        from_clause = select.args.get("from_") or select.args.get("from")
        sources = [from_clause] + list(select.args.get("joins") or [])
        for source in sources:
            if source is None:
                continue
            table = source.this if hasattr(source, "this") else None
            if not isinstance(table, exp.Table) or not isinstance(table.this, exp.Identifier):
                continue
            ref = (table.db.lower(), table.name.lower())
            aliases[table.alias_or_name.lower()] = ref
            aliases[table.name.lower()] = ref
        return aliases

    def _lob_kind(self, column: exp.Column, aliases: dict[str, tuple[str, str]]):
        name = column.name.lower()
        if column.table:
            ref = aliases.get(column.table.lower())
            candidates = [ref] if ref else []
        else:
            candidates = list(set(aliases.values()))
        for schema, table in candidates:
            if not schema:
                # Unqualified table name: match any schema holding it
                for (s, t), cols in self.columns.items():
                    if t == table and name in cols:
                        return s, cols[name]
                continue
            kind = self.columns.get((schema, table), {}).get(name)
            if kind:
                return schema, kind
        return None

    def output_columns(self, tree: exp.Expression) -> list[tuple[str, str] | None]:
        """Per output column of the top-level SELECT: (schema, kind) when
        it is a plain reference to a large-object column, else None."""
        select = tree if isinstance(tree, exp.Select) else None
        if select is None:
            return []
        aliases = self._aliases(select)
        result = []
        for projection in select.expressions:
            target = projection.this if isinstance(projection, exp.Alias) else projection
            result.append(self._lob_kind(target, aliases) if isinstance(target, exp.Column) else None)
        return result

    def check_query(self, tree: exp.Expression) -> None:
        """Refuse filtering/sorting/grouping/transforming a large-object
        column: the analytical DB only holds the OID number, so any such
        comparison silently runs against the wrong value. IS [NOT] NULL
        and COUNT() are fine - the OID's presence is real."""
        if not self.columns:
            return
        for select in tree.find_all(exp.Select):
            aliases = self._aliases(select)
            for column in select.find_all(exp.Column):
                found = self._lob_kind(column, aliases)
                if not found:
                    continue
                parent = column.parent
                while isinstance(parent, exp.Paren):
                    parent = parent.parent
                if isinstance(parent, (exp.Is, exp.Count)):
                    continue
                if isinstance(parent, exp.Alias) and isinstance(parent.parent, exp.Select):
                    continue
                if isinstance(parent, exp.Select) and column in parent.expressions:
                    continue
                if isinstance(parent, exp.Not) and isinstance(parent.this, exp.Is):
                    continue
                raise ValueError(
                    f"Column {column.sql()} holds stored document text that lives in the "
                    "source system - the database only has a reference number for it, so "
                    "it cannot be searched, compared, sorted, grouped or transformed in "
                    "SQL. Select the column plainly (no function around it) and its real "
                    "text is filled in automatically in the result; then read that text "
                    "to answer. To find rows by their content, select the candidate rows "
                    "with this column and inspect the returned text."
                )

    # ------------------------------------------------------------------
    # Fetching and decoding
    # ------------------------------------------------------------------

    def _engine(self, schema: str):
        if schema not in self._engines:
            url = self.source_urls.get(schema)
            self._engines[schema] = create_engine(url, pool_pre_ping=True) if url else None
        return self._engines[schema]

    def _fetch(self, schema: str, oids: set[int]) -> dict[int, bytes] | None:
        engine = self._engine(schema)
        if engine is None:
            return None
        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            conn.execute(text("SET LOCAL statement_timeout = 10000"))
            rows = conn.execute(
                text(
                    "SELECT m.oid::bigint, lo_get(m.oid) FROM pg_largeobject_metadata m "
                    "WHERE m.oid = ANY(CAST(:oids AS oid[]))"
                ),
                {"oids": sorted(oids)},
            ).fetchall()
        return {int(oid): bytes(data) for oid, data in rows}

    def _describe_file(self, data: bytes) -> str:
        kind = next((name for sig, name in _FILE_SIGNATURES if data.startswith(sig)), "binary file")
        return f"[{kind}, {len(data):,} bytes - not text, content not shown]"

    def _decode(self, data: bytes, kind: str) -> str:
        if kind == "file" or any(data.startswith(sig) for sig, _ in _FILE_SIGNATURES):
            return self._describe_file(data)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            return self._describe_file(data)
        if "\x00" in content:
            return self._describe_file(data)

        content = _to_plain_text(content)
        if len(content) > self.max_chars:
            content = content[: self.max_chars] + f" ... [truncated, {len(content):,} characters in total]"
        return content

    def resolve(self, tree: exp.Expression, payload: dict) -> dict:
        """Replace large-object OIDs in `payload` rows with their content."""
        targets = [
            (index, found) for index, found in enumerate(self.output_columns(tree)) if found
        ]
        if not targets or not payload.get("rows"):
            return payload

        wanted: dict[str, set[int]] = {}
        for index, (schema, _kind) in targets:
            for row in payload["rows"]:
                oid = _as_oid(row[index]) if index < len(row) else None
                if oid is not None:
                    wanted.setdefault(schema, set()).add(oid)

        fetched: dict[str, dict[int, bytes] | None] = {}
        for schema, oids in wanted.items():
            try:
                fetched[schema] = self._fetch(schema, oids)
            except Exception as e:  # noqa: BLE001 - reported per value below
                logger.warning("[LARGE_OBJECT] fetch failed for schema %s: %s", schema, e)
                fetched[schema] = None

        resolved_columns, file_columns = [], []
        for index, (schema, kind) in targets:
            (file_columns if kind == "file" else resolved_columns).append(payload["columns"][index])
            contents = fetched.get(schema)
            for row in payload["rows"]:
                oid = _as_oid(row[index]) if index < len(row) else None
                if oid is None:
                    continue
                if contents is None:
                    row[index] = "[stored text could not be retrieved - the source system is unavailable]"
                elif oid not in contents:
                    row[index] = "[stored text not found in the source system]"
                else:
                    row[index] = self._decode(contents[oid], kind)

        logger.info(
            "[LARGE_OBJECT] resolved text=%s files=%s oids=%d",
            resolved_columns, file_columns, sum(len(o) for o in wanted.values()),
        )
        payload["large_object_columns"] = resolved_columns
        payload["file_columns"] = file_columns
        return payload


def _as_oid(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


_BLOCK_TAGS = re.compile(r"</?(p|div|br|li|tr|h[1-6]|ul|ol|table)\b[^>]*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>]+>")


def _to_plain_text(content: str) -> str:
    """HTML (rich-text editor output) -> readable plain text; JSON is
    compacted; anything else is returned as stored."""
    stripped = content.strip()
    if stripped.startswith(("{", "[")):
        try:
            return json.dumps(json.loads(stripped), ensure_ascii=False, separators=(", ", ": "))
        except ValueError:
            return stripped
    if "<" in stripped and _ANY_TAG.search(stripped):
        text_only = _BLOCK_TAGS.sub("\n", stripped)
        text_only = html.unescape(_ANY_TAG.sub("", text_only)).replace("\xa0", " ")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text_only.splitlines()]
        return "\n".join(line for line in lines if line)
    return stripped
