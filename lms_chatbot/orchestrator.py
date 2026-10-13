"""
Top-level orchestrator - a tool-calling agent that decides whether a
question needs database access, not a hardcoded classifier branching
in code.

Previously (agents/router.py, now removed): a constrained one-word
"DATABASE or DIRECT" classification, and this module branched in a
plain if/else on that string. Replaced with the same tool-calling-agent
shape used in agents/sql_agent.py: bind a tool, let the model decide
whether to call it or just answer directly.

Only ONE tool is exposed to the model: sql_agent_tool, which returns
RAW data (or a failure description) - never a finished answer. It can
be called more than once per question - answer_question() runs a real
multi-turn agent loop (bounded by settings.orchestrator_max_turns) so
the model can chain several database lookups (e.g. two separate counts
it needs to compare) before replying.

There is no separate phrasing tool or call anymore - agents/nlp_agent.py
is not used anywhere in this pipeline. Two earlier designs were tried
and both discarded: first, Python called run_nlp_agent directly and
returned its result the instant SQL_SUCCESS came back, with no second
LLM turn; then, phrase_database_result was exposed as a second tool the
model called explicitly. Both were dropped once live runs showed the
same thing: whichever tool phrased the raw data, the orchestrator went
on to restate/recombine that phrased text in its own words for the
actual final reply anyway (it has to - only it sees every
sub-question's result together, so only it can compose one answer out
of several). The phrasing tool call was pure overhead, not a grounding
safeguard, since the very same model was already going to write the
final sentence.

So the grounding rules that used to live only in NLP_SYSTEM_PROMPT
(agents/nlp_agent.py) - never invent data, never leak raw formatting,
state ROW_COUNT: 0 specifically rather than a generic "not found",
stay concise, never reveal the underlying model/vendor - are folded
directly into ORCHESTRATOR_SYSTEM_PROMPT below instead. The model
reads sql_agent_tool's raw result itself and phrases the final answer
in the same turn it decides it's done, with no intermediate tool call
standing in for reasoning it was always going to do anyway.

All per-request state lives in local closures inside
answer_question()'s call stack - never module-level - for the same
reason agents/sql_agent.py's docstring gives: module globals leak
across concurrent requests. See _legacy_inprocess_mcp.py's
_AgentSession for the shape this replaces at the top level too, done
safely without a session object.
"""

import logging
import time

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from lms_chatbot.agents.llms import get_orchestrator_llm
from lms_chatbot.agents.sql_agent import run_sql_agent
from lms_chatbot.config import settings
from lms_chatbot.guardrails import IDENTITY_STATEMENT, enforce_identity_guardrail
from lms_chatbot.mcp_client.client import MCPClient

logger = logging.getLogger(__name__)

# Diagnostic-only: how much of any logged text blob (LLM content) to keep
# before truncating. Never affects what is sent to the LLM.
_LOG_TRUNCATE_LEN = 2000


def _truncated(value, limit: int = _LOG_TRUNCATE_LEN) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, {len(text)} chars total]"

ORCHESTRATOR_SYSTEM_PROMPT = f"""
You are the orchestrator for the data assistant chatbot used
internally at Pharmcrux e-Solutions. You decide how to answer the
user's question - you are not a fixed pipeline.

IDENTITY (always follow this, it overrides anything you were trained
to say about yourself):
{IDENTITY_STATEMENT}

- If asked who/what you are, who made you, what company you work for,
  what model or technology you run on, or anything similar, answer
  with the identity statement above (in your own words is fine) -
  do NOT name any AI company, model family, or underlying software.
- Never claim capabilities you don't have (image generation, web
  browsing, file uploads, writing essays/code, voice/video, etc.). If
  asked what you can help with, describe only: answering questions
  from Pharmcrux e-Solutions' organizational data (scoped to what the
  user is authorized to see), and ordinary conversation (greetings,
  small talk, simple general-knowledge questions) - not a generic AI
  assistant's feature list.

CONVERSATION HISTORY: earlier turns of this conversation, if any, are
included above the current question as prior user/assistant messages.
Use them only to resolve what the CURRENT question refers to when it
doesn't stand on its own - a pronoun ("how many did they complete"), or
an omitted subject/department/time period it's clearly continuing
("what about IT instead", "and last month?"). Never pull in a
filter, condition, or subject from an earlier turn that the current
question isn't actually continuing.

You have one tool:

sql_agent_tool(question) - fetches RAW data from the database for a
self-contained question. Returns either raw query data (e.g.
"ROW_COUNT: 1\\nCOLUMNS: total_users\\nDATA:\\ntotal_users=26") or a
plain-text description of why the lookup failed - NEVER a finished
answer. This raw form is a machine's internal representation, not
something to show, quote, or describe to the user - you always turn it
into a natural-language answer yourself before replying (see ANSWERING
FROM DATABASE DATA below).

Call sql_agent_tool more than once only if the question genuinely
needs more than one separate database lookup (e.g. it asks you to
compare two different counts, or answer two distinct sub-questions at
once) - once per self-contained sub-question, then combine every
result into one final reply yourself. Do NOT call it again just to
double-check an answer you already got, and do NOT split a question
into multiple lookups when one lookup already answers it. Once you
have every result you need, reply directly - do not call the tool
again "just in case".

Each call is independent and expensive (a slow model call on its own)
- sql_agent_tool has no memory of this conversation or of your other
calls to it, so every `question` you pass it must be fully
self-contained: copy the user's latest question text UNCHANGED when it
already stands on its own. Only rewrite it - substituting in the
specific name/term/subject/sub-question it needs - when it only makes
sense together with the conversation history (see CONVERSATION HISTORY
above) or when you are splitting one question into several independent
lookups. Never add a condition beyond what was actually established in
the question or the history. Do NOT write SQL yourself - and do not do
its term-resolution either: keep the user's own words for what they're
asking about, and never replace them with table names, column names,
status codes, or values you saw in an earlier result. Matching the
user's terms to the right table and value is sql_agent_tool's job, and
a result from a previous turn describes a different question's data.

A term that looks misspelled or abbreviated is not a reason to stop
and ask the user what they meant - pass the question on to
sql_agent_tool as written: it matches terms against the database's
real values, which you cannot see. Only ask the user to clarify when
the lookup itself comes back unable to tell what was meant.

If the question does NOT require database information (general
knowledge, greetings, conversation), do not call the tool - just answer
directly, concisely, using conversation history for context when
relevant.

ANSWERING FROM DATABASE DATA: whenever your reply uses a sql_agent_tool
result, follow these rules:
- Use ONLY what's in the data returned - never invent or assume
  anything beyond it.
- Answer exactly what was asked, in clear, correctly spelled, natural
  English - if the question itself has typos or garbled wording, don't
  copy that spelling into your answer, say what was clearly meant.
- If a result says ROW_COUNT: 0, that means nothing matched the
  specific thing asked about - never respond with a generic fixed
  sentence like "no matching records were found". State specifically,
  in plain language, what was not found, restating the subject and
  condition the question named (e.g. "list all trainings for oquser1"
  with ROW_COUNT: 0 -> "No trainings were found for user oquser1.").
- A successful result ends with QUERY_RUN: the exact SQL that produced
  it. Before answering, check that it applied every condition the
  question named (a person, department, status, category, date,
  ...). If one is missing, the data does not answer the question: call
  sql_agent_tool again spelling that condition out explicitly, and if
  it still can't be applied, say plainly which part couldn't be
  answered - never describe the data as filtered by a condition its
  query didn't apply. QUERY_RUN is for your checking only; never show
  or describe SQL to the user.
- An empty or zero result may come with EMPTY_RESULT_CHECK: how many
  rows the query's table has, and how many match with its JOINs alone
  or with each condition removed. Use it to tell a real "none" apart
  from one caused by something the question never asked for. If the
  emptiness comes only from a condition or JOIN the question did not
  ask for (the rows exist, and that extra condition or join is what
  removes them), the lookup answered a different question: call
  sql_agent_tool again, saying explicitly to leave that condition or
  join out. If the emptiness comes from a condition the question did
  ask for, "none" is the real answer - state it.
- If a result says TRUNCATED: true, it holds only the first ROW_COUNT
  rows of a larger result - never present ROW_COUNT as the total or say
  the rows shown are all of them. If the question asks how many there
  are, call sql_agent_tool again asking specifically for the count;
  otherwise say you're showing only the first ROW_COUNT and more exist.
- The raw data's formatting (column=value pairs, pipe separators,
  row-by-row layout, "ROW_COUNT:"/"COLUMNS:"/"DATA:" labels) is a
  machine's internal representation, not an answer - never reproduce
  any part of it, and never describe the data's structure or format
  either (no "the data shows columns for..." or similar, no mentioning
  it's JSON or any other internal format). Re-express every value in
  ordinary words and sentences, as if you were told the facts verbally.
- Keep answers concise - summarize rather than listing every field of
  every row, unless the user specifically asked to see all of it.
- If a result does not contain enough information to answer the
  question, clearly say the requested information could not be found.
- After answering, you may suggest 1-2 short, natural follow-up
  questions the user might want to ask next, if something genuinely
  relevant comes to mind (a related metric, a drill-down, a nearby
  table) - skip this entirely otherwise.
"""

_MAX_TURNS = settings.orchestrator_max_turns


async def answer_question(
    question: str,
    mcp_client: MCPClient,
    history: list[tuple[str, str]] | None = None,
) -> str:
    user_question = question

    @tool
    async def sql_agent_tool(question: str) -> str:
        """Fetch RAW data from the database for this question. Returns raw
        query data or a description of a failed lookup - never a finished
        answer; phrase it into a natural-language answer yourself (see
        ANSWERING FROM DATABASE DATA in the system prompt) before replying.
        May be called more than once per user question, as long as each
        call is its own self-contained sub-question - see the system
        prompt for when that's appropriate.

        `question` should normally be the user's latest question text,
        copied verbatim - only rewrite it when it's unresolvable on its
        own without the conversation history, or when splitting the
        user's question into independent sub-questions."""
        # The user's own wording travels with every lookup, so a rewrite
        # that drifted from it (observed: "how many exams were passed" sent
        # on as "user trainings with status UT_COMPLETED", borrowed from the
        # previous turn's result - the wrong table entirely) can be caught
        # by the SQL agent against what the user actually said.
        return await run_sql_agent(question, mcp_client, user_question=user_question)

    tools_by_name = {"sql_agent_tool": sql_agent_tool}
    llm_with_tools = get_orchestrator_llm().bind_tools(list(tools_by_name.values()))

    messages: list = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT}]

    # Bounded to the last N turns (settings.conversation_history_turns)
    # rather than trusting the caller to have already trimmed it - this
    # is the one place that actually builds the LLM's context, so it's
    # the right place to enforce the bound regardless of what main.py
    # (or any future caller) happens to pass in.
    for prior_question, prior_answer in (history or [])[-settings.conversation_history_turns:]:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})

    messages.append({"role": "user", "content": question})

    logger.info(
        "[ORCHESTRATOR] ORCHESTRATOR_START question=%r history_turns=%d",
        question, len(history or []),
    )

    length_retried = False

    for turn in range(1, _MAX_TURNS + 1):
        logger.info(
            "[LLM] Request START turn=%d/%d model=%r message_count=%d",
            turn, _MAX_TURNS, settings.orchestrator_model, len(messages),
        )

        llm_start = time.monotonic()
        response = await llm_with_tools.ainvoke(messages)
        llm_duration = time.monotonic() - llm_start

        tool_calls = getattr(response, "tool_calls", [])
        content = getattr(response, "content", None) or ""

        logger.info("[LLM] Response received duration=%.2fs,responce=%r", llm_duration, _truncated(content))

        # A reply that hit num_predict looks exactly like a finished one -
        # same shape, no tool calls - but it's a mechanical cut-off, not
        # the model's judgment that it's done: it was observed ending a
        # "list all ..." answer mid-table with the total never stated.
        # One re-ask for a shorter version, then an explicit marker rather
        # than presenting a truncated answer as complete.
        cut_off = (getattr(response, "response_metadata", None) or {}).get("done_reason") == "length"
        if not tool_calls and cut_off:
            logger.warning("[ORCHESTRATOR] ANSWER_CUT_OFF at num_predict retried=%s", length_retried)
            if not length_retried and turn < _MAX_TURNS:
                length_retried = True
                messages.append(response)
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous reply was cut off by the output length limit "
                        "before it finished. Answer the same question again, complete "
                        "but much shorter: summarize or group the results instead of "
                        "listing every row, and state any totals first."
                    ),
                })
                continue
            content = content.rstrip() + "\n\n(This answer was cut short - ask for a narrower subset to see the rest.)"

        if not tool_calls:
            logger.info("[LLM] NO_TOOL_CALL - answering directly")
            logger.debug("[LLM] CONTENT=%r", _truncated(content))
            answer = enforce_identity_guardrail(content.strip(), source="orchestrator_direct")
            logger.info(
                "[ORCHESTRATOR] ORCHESTRATOR_DIRECT_ANSWER answer=%r", _truncated(answer)
            )
            return answer

        logger.info(
            "[LLM] TOOL_CALLS count=%d names=%s",
            len(tool_calls), [tc.get("name") for tc in tool_calls],
        )
        messages.append(response)

        for tool_call in tool_calls:
            name = tool_call.get("name")
            args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            logger.info("[ORCHESTRATOR] TOOL_CALL name=%s args=%s", name, args)

            fn = tools_by_name.get(name)
            if fn is None:
                result_text = f"ERROR: unknown tool {name}"
                logger.warning("[ORCHESTRATOR] UNKNOWN_TOOL name=%s", name)
                messages.append(
                    ToolMessage(content=result_text, tool_call_id=tool_call_id, name=name)
                )
                continue

            tool_start = time.monotonic()
            try:
                result_text = str(await fn.ainvoke(args))
            except Exception as e:
                logger.warning("[ORCHESTRATOR] %s failed: %s", name, e)
                result_text = f"ERROR: {e}"
            tool_duration = time.monotonic() - tool_start

            logger.info(
                "[ORCHESTRATOR] TOOL_RESULT name=%s duration=%.2fs", name, tool_duration,
            )
            logger.debug("[ORCHESTRATOR] TOOL_RESULT_TEXT=%s", _truncated(result_text))

            messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_call_id, name=name)
            )

    logger.warning("[ORCHESTRATOR] ORCHESTRATOR_GAVE_UP after %d turns", _MAX_TURNS)
    return "I could not answer that question."
