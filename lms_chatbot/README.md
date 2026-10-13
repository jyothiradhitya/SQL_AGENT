# lms_chatbot

Rebuild of the LMS text-to-SQL chatbot with: a real MCP server/client
(stdio transport) instead of in-process function calls, an
anchor -> bridge -> lookup schema retrieval pipeline instead of
dumping the whole schema (or an unbounded FK-graph BFS) into the
prompt, structured logging instead of `print()`, and centralized
config instead of `os.getenv()` scattered across files.

The old `SQL_AGENT/`, `NLP_AGENT/`, `main.py` and `_legacy_inprocess_mcp.py`
at the repo root are untouched and still work standalone; this is a
parallel rebuild, not a patch.

## Why each piece exists

- **`retrieval/`** - a question rarely names a table directly, and
  most tables here store nothing but foreign-key ids (`status_id`,
  `category_id`, ...), so resolving them requires joins. The old
  approach either dumped the entire schema (too big) or did an
  unbounded FK-graph BFS from a seed table (also too big - depth 3
  from one table pulled in ~35 of the 49 known tables). This package
  instead: embeds a question against *anchor* (business-entity) tables
  only, finds the *shortest* FK path connecting whichever anchors
  matched (not everything N hops out), and attaches one hop of
  *lookup* (id -> label) tables so the agent can resolve ids by
  joining and filtering on a name column instead of guessing an
  integer.

- **`mcp_server/` + `mcp_client/`** - real MCP protocol (the `mcp`
  PyPI package, stdio transport, `FastMCP`), not the old `mcp.py`'s
  plain Python function calls. `get_relevant_schema` is exposed as a
  tool the SQL agent LLM calls itself (and can re-call with a refined
  question if its first attempt at SQL fails for a missing table),
  rather than schema being injected once as a static prompt block.

- **`sql/`** - the validator (`validator.py`) is a close port of the
  existing `SQL_AGENT/validate_sql.py`, which was already solid:
  AST-based (not regex) blocking of DML/DDL and dangerous functions
  (`pg_sleep`, `dblink`, file-read functions, ...), a table allow-list
  enforced against the *entire* known schema (not narrowed to
  whatever one question's retrieval selected - that distinction
  matters, see `SchemaRetriever.all_table_names`), a hard row-limit
  ceiling, and execution inside a read-only, timeout-bounded
  transaction.

- **`agents/`** - `sql_agent.py` (drives the MCP tool-calling loop),
  `nlp_agent.py` (turns the raw query result into a natural-language
  answer). The top-level `orchestrator.py` is itself a tool-calling
  agent that decides whether a question needs `sql_agent_tool` (and
  then `nlp_agent_tool`) at all, rather than a separate classifier
  branching in code - see its docstring for why the old
  DIRECT/DATABASE classifier (`agents/router.py`) was removed. All
  per-request state is local variables/closures inside these
  functions - there is no module-level mutable state to leak across
  concurrent requests, which is what `_legacy_inprocess_mcp.py`'s
  session objects existed to patch over.

## Setup

1. Add to the repo-root `.env` (existing `OLLAMA_URL` / `ANALYTICAL_DB`
   are reused as-is):
   ```
   ORCHESTRATOR_MODEL=qwen2.5:3b
   SQL_MODEL=qwen3:8b
   NLP_MODEL=qwen2.5:3b
   EMBED_MODEL=nomic-embed-text
   LOG_LEVEL=INFO
   ```
   All of the above have defaults matching the old code, so this step
   is optional unless you want different models.

2. Build the anchor-table embedding index (needed once, and again
   whenever `SQL_AGENT/Data/semantic_kb.json` changes):
   ```
   uv run python -m lms_chatbot.retrieval.build_index
   ```

3. Run the chatbot:
   ```
   uv run python -m lms_chatbot.main
   ```

## Known gaps / next steps

- `lms.training_attachment` is tagged `"Reference table"` in
  `semantic_kb.json` even though it's really a training<->attachment
  junction table, not an id->label lookup. It got classified as a
  lookup table here because `retrieval/classify.py` trusts the KB's
  own `table_role` field first. Worth fixing at the KB-generation
  step rather than special-casing it here.
- The MCP server subprocess is shared across the whole CLI session.
  That's fine for this single-user CLI; if this becomes a concurrent
  multi-user service, check whether one `ClientSession` handles
  overlapping in-flight tool calls correctly, or give each request its
  own `MCPClient`/subprocess.
- No automated tests yet (validator.py in particular deserves unit
  tests given what it's protecting), no CI, no Dockerfile - out of
  scope for this pass, called out here so it isn't mistaken for
  "done."
