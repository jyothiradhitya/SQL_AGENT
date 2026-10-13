"""
SQL agent driven by real MCP tool calls.

Unlike the old SQL_AGENT/agent_test.py, the schema is not pre-injected
into the system prompt - the model is handed two tools and told to
call get_relevant_schema itself before writing SQL. If its first
attempt at SQL fails because a table it needed wasn't in the schema it
was shown, it is instructed to call get_relevant_schema again with a
more specific question, rather than the caller guessing what to
force-include.

Python's role is limited to the two tools' actual mechanics - schema
retrieval (retrieval/pipeline.py) and SQL execution/validation
(sql/executor.py, sql/validator.py) - both running out-of-process
behind the real MCP server (mcp_server/server.py). Everything about
*when* to call which tool, how many times, and when to give up is the
model's own judgment call, driven by real tool results in the loop
below. An earlier version also injected a synthetic "reminder" user
message after each turn (e.g. "call a tool now", "fix the SQL error")
to steer a small local model that didn't reliably keep itself on
track. That scaffolding is gone: with a capable model
(settings.sql_model), those synthetic nudges were pure Python-authored
"logic" standing in for reasoning the model can do on its own from the
plain tool results already in its context.

All state (the message history, whether SQL has succeeded yet) lives
in local variables inside run_sql_agent()'s call stack. Nothing is
stored on a module-level object, so concurrent calls for different
users' questions never share state - the cross-request data leak that
_legacy_inprocess_mcp.py's session objects existed to patch over
structurally cannot happen here.
"""

import logging
import time

from langchain_core.messages import ToolMessage

from lms_chatbot.agents.llms import get_sql_llm
from lms_chatbot.config import settings
from lms_chatbot.mcp_client.client import MCPClient, to_openai_tool_schema, tool_result_text

logger = logging.getLogger(__name__)

SQL_AGENT_SYSTEM_PROMPT = """
You are a PostgreSQL SQL agent with two tools:

1. get_relevant_schema(question) - returns the tables, columns and
   join paths relevant to a database question.
2. run_sql(sql) - validates and executes ONE PostgreSQL SELECT/WITH
   query.

RULES:

- ALWAYS call get_relevant_schema first, before writing any SQL.
- QUESTION may be a rewrite of what the user typed, shown as USER'S
  ORIGINAL WORDING when the two differ. The rewrite is there to fill in
  context the user left out (a pronoun, a subject from earlier in the
  conversation, one part of a split question); for any term the user
  actually wrote themselves, their own wording decides what it means.
  If QUESTION names a different kind of thing than the user's own words
  do, fetch what the user's words refer to, and base your
  get_relevant_schema call on those words.
- Each COLUMNS entry is written as table_name.column_name so that when
  several tables are shown together you always know which table a
  column actually belongs to - a column belongs ONLY to the table it
  is written under, never to a different table shown elsewhere just
  because that other table seems like a plausible owner too (e.g. a
  "login"-like column shown under one table does NOT mean a
  similarly-themed table elsewhere also has it - check its own
  COLUMNS entry). This notation describes ownership only; it is not
  SQL syntax to copy - write your own table aliases in the query as
  normal.
- Each call to get_relevant_schema is expensive (a slow model call on
  its own). NEVER call it twice in a row with the same or a barely
  reworded question just to double-check something you already have -
  if the tables/columns/JOINS you need are already visible from an
  earlier call this conversation, write the SQL now. Only call it
  again when you have identified, specifically, a table or column you
  need that is genuinely absent from every schema result you've seen
  so far, and phrase that call around that missing thing.
- Use ONLY tables and columns shown in the schema you were given.
- NEVER invent a column, table, alias, or relationship.
- NEVER use SELECT *.
- NEVER select a credential/security column (password_hash, auth_string,
  token, session_id, activation_key, reset_key, reset_remarks, ip) or a
  direct personal contact/identity column (email, mobile_number,
  birthdate, profile_picture, a user's photo/image_url) even if a
  question asks for it directly or a schema you retrieved shows one -
  these are hard-blocked and run_sql will reject the query. If a
  question can only be answered by returning one of these, don't retry
  with a different phrasing - stop and say plainly that this
  information can't be provided.
- A column named like a date/time (from_date, thru_date, start_time,
  end_time, created_date, effective_date, ...) is NOT necessarily a
  real TIMESTAMP/DATE column - check its type in the COLUMNS line
  first. In this database:
    * BIGINT means the column holds a Unix epoch timestamp in
      MICROSECONDS, not seconds - convert it with
      to_timestamp(column / 1000000.0) before comparing it to a date,
      extracting a field from it, or displaying it.
    * INTEGER means the column holds a day count since 1970-01-01 (not
      seconds, not a real DATE) - convert it with
      (DATE '1970-01-01' + column) before comparing it to a date or
      displaying it.
    * A column already typed TIMESTAMP WITHOUT TIME ZONE or DATE needs
      no conversion - use it directly.
  Never write a raw comparison (column > '2024-01-01', EXTRACT(...
  FROM column), etc.) against a BIGINT or INTEGER date-shaped column
  without converting it first - it will either error (type mismatch)
  or silently compare against the wrong unit.
- SELECT ONLY the columns that directly answer the question - not
  every column a table has, and not every column the schema shows you
  just because it's available. Add a column beyond what was asked
  only if it's a label needed to make the requested column meaningful
  (e.g. joining an id to its name so the id itself doesn't need to be
  shown).
- When the question asks WHICH/WHAT thing (which department, which
  user, what category...), return that thing's name/label - JOIN the
  id to the table it references to get it. A bare id (department_id =
  5) is never an answer a person can use.
- To count, rank, or compare per entity (per user, per training, per
  department...), GROUP BY that entity's own id and show a label for it
  alongside - never group by a label column alone (name, display_name,
  title, ...): labels aren't unique, and grouping by one silently
  merges different entities that happen to share it. For a person, the
  label MUST be their login/username (JOIN through to the login table
  when it isn't on the row) - a person's name fields are often a
  shared title or blank, and don't tell people apart.
- Use JOIN only when the requested answer actually needs another
  table - the same rule applies to JOINs as to columns: a table being
  reachable in the schema you were given is not a reason to include it.
- NEVER add a WHERE condition, status filter, or any other restriction
  that the question did not ask for. A schema showing a status/lookup
  column does not mean the question wants to filter on it. If the
  question names no condition, return unfiltered rows.
- Many columns only store an id (status_id, category_id, type_id...).
  Never filter on one of these by guessing an integer value, and never
  invent or guess the text of a label/name value either. If the
  user's term ("active", "SOP training", "QA", ...) corresponds to a
  lookup table shown under JOINS, JOIN to that table and filter on its
  label column instead. Some lookup tables include a VALUES list of
  their real id/name/description/acronym rows (acronym is only shown
  when a row's name has one) - use it to help you FIND the right row
  by matching the user's term against any field shown for that row,
  including acronym, not by guessing what the value "should" look like
  (it will not always be the literal word: "active" can be a row whose
  name is something else entirely, with "active" only appearing in its
  description; "QA" can be a row whose acronym field reads exactly
  "QA" even though its name reads "Quality Assurance").
  IMPORTANT: acronym is NOT a real column on the table - it exists
  only in this VALUES list to help you identify the row, and writing
  `<table>.acronym` in SQL will fail with an unknown-column error. Once
  you've found the matching row this way, the value you actually put
  in the SQL filter MUST be that row's value under a column that IS
  listed in the table's own COLUMNS line (almost always `name`), copied
  verbatim - never `acronym` itself, and never anything you made up. If
  no VALUES row matches the user's term under any of this, say so
  instead of filtering on a guess.
- created_by, last_modified_by, and similar audit columns record who
  created/edited the ROW in the system (often a system/service
  account) - they are NEVER the person a "for <name/username>" or
  "<name>'s ..." question is asking about, even when they are the only
  text-like column visible. The same holds for any column recording
  someone's ROLE on a row (supervisor, approver, trainer, reviewer,
  initiator, ...): it may hold usernames, but it names who supervised
  or approved the record, not whose record it is - use it only when the
  question asks about that role. The real link to a specific person is a
  dedicated identity column (e.g. party_id, emp_id, user_id,
  trainee_id) that joins toward a party/user/employee table. If the
  schema you were given has no such column on a relevant table, call
  get_relevant_schema again asking specifically for that person/user/
  party/login table instead of guessing an audit column.
- A question naming a specific person is asking about a FACT/record
  table (one row per event/assignment - it has the identity column
  described above), never a catalog/definition table (one row per
  type/program/category - it only describes what a thing IS, not who
  it happened to). Pick the table by whether it HAS a person-identity
  column, not by which table's name most resembles a word in the
  question - a table can be named almost identically to a related one
  (e.g. "training" vs "user_training") while only one of them actually
  links to a person. A username is a login, not a person's name - it
  belongs in a sys_user/login-style column, never in a first_name/
  last_name/display_name column, even if you have no better place to
  put it.
- If run_sql fails because a table or column you need was not in the
  schema you were given, call get_relevant_schema again with a more
  specific or differently-worded question. Do not guess.
- Call run_sql with exactly ONE query at a time.
- If run_sql succeeds but the result is empty or zero, it comes with
  EMPTY_RESULT_CHECK: how many rows the table has, and how many match
  with the JOINs alone or with each condition removed. Review it once.
  If the emptiness comes from a condition or JOIN the question did not
  actually ask for (the rows exist, and your extra condition/join is
  what removes them), run the query again without it. If it comes from
  exactly what the question asked, the empty result is the real answer
  - stop without calling a tool, and it stands.
- Do not explain your reasoning while you work - only call tools. Stop
  calling tools once run_sql has succeeded. If you genuinely cannot
  get the requested data (no matching table/column exists anywhere in
  the schema you retrieved, even after refining your schema lookup),
  stop and say plainly, in one line, what could not be found and why -
  do not fabricate a result or keep repeating the same failing call.
"""

# Diagnostic-only: how much of any single logged text blob (LLM content,
# tool args, tool result) to keep before truncating. Purely for log
# readability - never affects what is sent to the LLM or the database.
_LOG_TRUNCATE_LEN = 4000


def _truncated(value, limit: int = _LOG_TRUNCATE_LEN) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, {len(text)} chars total]"


def _tool_result_status(result_text: str) -> str:
    if result_text.startswith("SQL_SUCCESS"):
        return "SUCCESS"
    if result_text.startswith("SQL_ERROR"):
        return "ERROR"
    if result_text.startswith("NO_TABLES_FOUND"):
        return "EMPTY"
    return "SUCCESS"


async def run_sql_agent(
    question: str, mcp_client: MCPClient, user_question: str | None = None
) -> str:
    logger.info("SQL agent start: %r user_question=%r", question, user_question)

    mcp_tools = await mcp_client.list_tools()
    tool_schemas = [to_openai_tool_schema(t) for t in mcp_tools]
    tool_names = [t.name for t in mcp_tools]

    logger.debug("SQL agent tools: %s", tool_names)

    llm_with_tools = get_sql_llm().bind_tools(tool_schemas)

    user_content = f"QUESTION: {question}"
    if user_question and user_question.strip() != question.strip():
        user_content += f"\nUSER'S ORIGINAL WORDING: {user_question}"

    messages: list = [
        {"role": "system", "content": SQL_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    logger.debug("SQL agent initial messages: %s", messages)

    # Bounded to settings.sql_max_attempts turns as a safety valve, not a
    # retry policy - there is no Python-authored logic left in this loop
    # deciding what the model should do next turn (see module docstring).
    reviewed_empty = False
    pending_success: str | None = None

    for turn in range(1, settings.sql_max_attempts + 1):
        turn_start = time.monotonic()
        logger.info(
            "[SQL_AGENT] Turn %d/%d START question=%r",
            turn, settings.sql_max_attempts, question,
        )
        logger.info(
            "[LLM] Request START model=%r tools=%d tool_names=%s message_count=%d",
            settings.sql_model, len(tool_names), tool_names, len(messages),
        )

        try:
            llm_start = time.monotonic()
            response = await llm_with_tools.ainvoke(messages)
            llm_duration = time.monotonic() - llm_start
        except Exception:
            logger.exception(
                "[LLM] Request FAILED turn=%d/%d", turn, settings.sql_max_attempts
            )
            raise

        tool_calls = getattr(response, "tool_calls", [])
        content = getattr(response, "content", None) or ""
        logger.info("[LLM] Response received duration=%.2fs", llm_duration)

        if not tool_calls:
            # The model judged itself done without (or without further)
            # calling run_sql. Its own text is trusted as the outcome -
            # either it's explaining why the data can't be found, or it's
            # mistaken - rather than Python forcing another turn with a
            # synthetic "call a tool now" instruction. After an empty-result
            # review, stopping means it accepted that empty result, which
            # is then the outcome.
            logger.info("[SQL_AGENT] NO_TOOL_CALL - ending turn=%d content=%r", turn, _truncated(content))
            if pending_success is not None:
                return pending_success
            return content.strip() or "I could not retrieve the requested database information."

        logger.info(
            "[LLM] TOOL_CALLS count=%d names=%s",
            len(tool_calls), [tc.get("name") for tc in tool_calls],
        )
        messages.append(response)

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            logger.info("[TOOL] %s START args=%s", tool_name, _truncated(tool_args, 1000))
            if tool_name == "run_sql":
                logger.info("[SQL_AGENT] GENERATED_SQL:\n%s", tool_args.get("sql"))

            tool_start = time.monotonic()
            try:
                result = await mcp_client.call_tool(tool_name, tool_args)
                result_text = tool_result_text(result)
                tool_status = _tool_result_status(result_text)
            except Exception as e:
                result_text = f"SQL_ERROR\nERROR: tool call failed: {e}"
                tool_status = "ERROR"
                logger.exception("[TOOL] %s raised an exception", tool_name)

            tool_duration = time.monotonic() - tool_start
            logger.info(
                "[TOOL] %s COMPLETED status=%s duration=%.2fs",
                tool_name, tool_status, tool_duration,
            )
            if tool_name == "run_sql":
                logger.info("[SQL_AGENT] SQL_RESULT status=%s:\n%s", tool_status, _truncated(result_text))
            else:
                logger.debug("[TOOL] %s result=%s", tool_name, _truncated(result_text))

            messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_call_id, name=tool_name)
            )

            # A successful run_sql ends the agent immediately - it is the
            # one deterministic contract this function honors, not a
            # judgment call left open to the model: the orchestrator is
            # waiting for exactly this "SQL_SUCCESS"-prefixed payload.
            if tool_name == "run_sql" and result_text.startswith("SQL_SUCCESS"):
                # The query travels with its data so the orchestrator can
                # tell what the rows actually mean - it was observed stating
                # "users in the PC department completed 57" when the SQL
                # behind that 57 had no department filter at all.
                success = f"{result_text}\nQUERY_RUN: {' '.join(str(tool_args.get('sql', '')).split())}"
                # An empty/zero result gets ONE look from the model before
                # it's final: EMPTY_RESULT_CHECK (sql/empty_result.py)
                # shows whether the rows exist and which condition removed
                # them, and only this agent has the schema to judge whether
                # that condition was asked for. The orchestrator, which
                # can't see the schema, was observed reading "19 rows
                # exist, the type JOIN leaves 1" and still answering "there
                # are no departments".
                if "EMPTY_RESULT_CHECK:" in result_text and not reviewed_empty:
                    reviewed_empty = True
                    pending_success = success
                    logger.info("[SQL_AGENT] EMPTY_RESULT_REVIEW turn=%d", turn)
                    continue
                logger.info("[SQL_AGENT] SUCCESS turn=%d/%d", turn, settings.sql_max_attempts)
                return success

        logger.info(
            "[SQL_AGENT] Turn %d/%d completed, continuing duration=%.2fs",
            turn, settings.sql_max_attempts, time.monotonic() - turn_start,
        )

    if pending_success is not None:
        return pending_success
    logger.warning("[SQL_AGENT] SQL_AGENT_GAVE_UP after %d turns", settings.sql_max_attempts)
    return "I could not retrieve the requested database information."
