import logging
import time

from lms_chatbot.agents.llms import get_nlp_llm
from lms_chatbot.config import settings
from lms_chatbot.guardrails import enforce_identity_guardrail

logger = logging.getLogger(__name__)

# Diagnostic-only: how much of any logged text blob (raw DB data, LLM
# content) to keep before truncating. Never affects what is sent to the LLM.
_LOG_TRUNCATE_LEN = 2000


def _truncated(value, limit: int = _LOG_TRUNCATE_LEN) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, {len(text)} chars total]"

NLP_SYSTEM_PROMPT = """
You are a communicating agent, not a data dump. Your ONLY job is to
read database results and speak the answer to a person the way a
knowledgeable colleague would say it out loud - not the way the data
was stored or formatted.

You receive:
1. USER QUESTION
2. DATABASE DATA

Use ONLY the database data provided.

Rules:
1. Never invent information.
2. Never query a database.
3. Never generate SQL.
4. Never assume information that is not in DATA.
5. Answer exactly what the user asked - but always in clear, correctly
   spelled, natural English. If the question itself contains typos or
   garbled wording (e.g. "aretrainings", "ther"), do NOT copy that
   spelling into your answer - say what they clearly meant instead
   (e.g. "trainings").
6. If DATA says ROW_COUNT: 0, that means nothing matched the specific
   thing the question asked about - never respond with a generic,
   fixed sentence like "No matching records were found." Reason from
   the USER QUESTION itself and state, specifically, what was not
   found, restating the subject and condition the question named in
   plain language (e.g. "list all trainings for oquser1" -> "No
   trainings were found for user oquser1."; "how many inactive SOPs"
   -> "No inactive SOPs were found."). Do not invent a reason why
   nothing was found - just state clearly what came up empty.
7. DATA's formatting (column=value pairs, pipe separators, row-by-row
   layout) is a machine's internal representation, not an answer.
   NEVER reproduce that formatting, or any part of it, in your answer.
   Re-express every value you use in ordinary words and sentences, as
   if you were told the facts verbally and were now repeating them to
   someone, not reading a printout to them.
8. Do not describe the data's structure or format at all - not as
   prose either (no "the data shows columns for..." or similar).
9. Do not say that the data is JSON.
10. Do not explain internal processing.
11. Keep the answer concise - summarize rather than listing every field
    of every row, unless the user specifically asked to see all of it.
12. If the database result does not contain enough information to
    answer the question, clearly say that the requested information
    could not be found.
13. Never state or imply what underlying AI model, vendor, or software
    powers you (no "as a language model", "I was trained by...",
    naming any AI company or model, etc.). You have no information
    about that and never discuss it - you are simply the Pharmcrux
    e-Solutions data assistant, answering from the organization's data
    scoped to what the user is authorized to see.
14. After the answer, suggest 1-2 short, natural follow-up questions
    the user might want to ask next, based on the question and the
    data (a related metric, a drill-down into one of the rows, a
    nearby table, etc.). When ROW_COUNT is 0, a suggestion can still
    help (e.g. checking a nearby spelling, a different user/filter) -
    but keep every suggestion genuinely relevant, not generic; skip
    this entirely if nothing relevant comes to mind.

Respond using EXACTLY this two-line structure, nothing before or after
it, no other labels or headers:

ANSWER: <the final natural-language answer, following all rules above>
SUGGESTIONS: <1-2 short follow-up questions separated by " | ", or the
single word none if rule 14 says to skip this>
"""


_ANSWER_MARKER = "ANSWER:"
_SUGGESTIONS_MARKER = "SUGGESTIONS:"
_NO_SUGGESTIONS = {"none", "n/a", "-", ""}


def _parse_structured_response(raw: str) -> tuple[str, list[str]]:
    """Split the model's `ANSWER: ... SUGGESTIONS: ...` response into
    its parts. A formatting instruction in a prompt is a request, not a
    guarantee (see guardrails.py on this same class of small-model
    unreliability) - if the model didn't follow the format, the whole
    response is treated as the answer with no suggestions, rather than
    failing the request over a cosmetic feature.
    """
    if _ANSWER_MARKER not in raw:
        return raw.strip(), []

    after_answer = raw.split(_ANSWER_MARKER, 1)[1]

    if _SUGGESTIONS_MARKER in after_answer:
        answer_part, suggestions_part = after_answer.split(_SUGGESTIONS_MARKER, 1)
    else:
        answer_part, suggestions_part = after_answer, ""

    answer = answer_part.strip()
    suggestions_raw = suggestions_part.strip()

    if suggestions_raw.lower() in _NO_SUGGESTIONS:
        return answer, []

    suggestions = [s.strip(" -•") for s in suggestions_raw.split("|")]
    return answer, [s for s in suggestions if s]


def _format_with_suggestions(answer: str, suggestions: list[str]) -> str:
    if not suggestions:
        return answer
    bullets = "\n".join(f"- {s}" for s in suggestions[:2])
    return f"{answer}\n\nYou might also want to ask:\n{bullets}"


def _format_raw_fallback(question: str, database_data: str) -> str:
    """Deterministic, LLM-free formatting for when the phrasing model
    produced no usable content even after a retry. This must NEVER just
    hand back `database_data` as-is (or mention that phrasing
    "failed") - that machine-formatted `col=val | col=val` block, and
    admitting internal processing details, are exactly what
    NLP_SYSTEM_PROMPT rules 7-10 forbid the LLM from doing. Skipping
    the LLM step entirely is not a reason to skip those same rules.

    There's no LLM available on this path to reason rule 6's way to a
    fully-phrased "no trainings were found for user X" - but echoing
    the question is still strictly more useful than a flat generic
    sentence, and needs no reasoning to do.
    """
    if "ROW_COUNT: 0" in database_data or "RESULT: NO_RESULTS" in database_data:
        return f"No matching records were found for: {question}"

    columns: list[str] = []
    data_rows: list[str] = []
    in_data = False
    for line in database_data.splitlines():
        if line.startswith("COLUMNS:"):
            columns = [c.strip() for c in line[len("COLUMNS:"):].split(",")]
        elif line == "DATA:":
            in_data = True
        elif in_data and line.strip():
            data_rows.append(line)

    if not data_rows:
        return "The query ran successfully, but the result could not be formatted."

    formatted_rows = []
    for row in data_rows:
        parts = []
        for pair in row.split(" | "):
            column, sep, value = pair.partition("=")
            parts.append(f"{column.strip()}: {value.strip()}" if sep else pair.strip())
        formatted_rows.append(", ".join(parts))

    if len(formatted_rows) == 1:
        return formatted_rows[0]

    bullets = "\n".join(f"- {r}" for r in formatted_rows)
    return f"Here's what I found ({len(formatted_rows)} results):\n{bullets}"


_NLP_MAX_ATTEMPTS = 2


async def run_nlp_agent(question: str, database_data: str) -> str:
    logger.info("[NLP_AGENT] NLP_AGENT_START question=%r data_chars=%d", question, len(database_data))
    logger.debug("[NLP_AGENT] DATABASE_DATA=%s", _truncated(database_data))

    llm = get_nlp_llm()

    prompt = f"""
USER QUESTION
=============

{question}


DATABASE DATA
=============

{database_data}


Generate the final answer.
"""

    messages = [
        {"role": "system", "content": NLP_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    raw = ""
    for attempt in range(1, _NLP_MAX_ATTEMPTS + 1):
        logger.info(
            "[LLM] Request START model=%r attempt=%d/%d", settings.nlp_model, attempt, _NLP_MAX_ATTEMPTS
        )
        llm_start = time.monotonic()
        response = await llm.ainvoke(messages)
        llm_duration = time.monotonic() - llm_start
        logger.info("[LLM] Response received duration=%.2fs", llm_duration)

        raw = (response.content or "").strip()
        logger.debug("[LLM] CONTENT=%r", _truncated(raw))

        if raw:
            break

        logger.warning(
            "[NLP_AGENT] NLP_AGENT_EMPTY_RESPONSE attempt=%d/%d duration=%.2fs - %s",
            attempt, _NLP_MAX_ATTEMPTS, llm_duration,
            "retrying" if attempt < _NLP_MAX_ATTEMPTS else "giving up, using deterministic fallback",
        )

    if not raw:
        return _format_raw_fallback(question, database_data)

    answer, suggestions = _parse_structured_response(raw)
    answer = enforce_identity_guardrail(answer, source="nlp_agent")

    if not answer:
        logger.warning("[NLP_AGENT] NLP_AGENT_EMPTY_ANSWER - parsed answer was blank")
        return _format_raw_fallback(question, database_data)

    final = _format_with_suggestions(answer, suggestions)

    logger.info(
        "[NLP_AGENT] NLP_AGENT_SUCCESS answer_chars=%d suggestion_count=%d",
        len(answer), len(suggestions),
    )
    return final
