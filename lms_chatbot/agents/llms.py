"""One place to construct each ChatOllama instance, so model choice and
inference parameters per role (routing/SQL/NLP) are tuned in one spot.

IMPORTANT: the constructor kwarg to disable a reasoning model's <think>
step is `reasoning`, NOT `think` - langchain_ollama.ChatOllama has no
`think` field (confirmed via ChatOllama.model_fields) and its pydantic
config is `extra="ignore"`, so passing `think=False` is silently
dropped, no error - the model keeps its default reasoning behavior.
For qwen3:8b this meant every call spent its whole num_predict budget
inside <think>...</think> and returned empty content with no tool
call. Always use `reasoning=False` here.

IMPORTANT (same trap, bigger budget): the num_predict values below are
sized for the LARGEST expected tool call, not the average one. A
multi-table JOIN's SQL text plus its surrounding tool-call JSON can
easily run past 250 tokens on its own, and cloud-hosted reasoning
models (e.g. gpt-oss:120b-cloud) have been observed NOT to fully honor
`reasoning=False` the way local models do - they still spend some of
the budget on hidden reasoning/analysis-channel tokens. Either way the
symptom is identical to the qwen3:8b case above: the response comes
back with empty content AND no tool_calls, which
agents/sql_agent.py's retry loop can only interpret as "the model
chose not to call a tool" and nudge again - it has no way to tell that
answer apart from a genuine truncation, so a tight num_predict just
burns through all sql_max_attempts on a question that only needed more
room to finish one response. Keep this generous rather than tight.

IMPORTANT (same trap, different kwarg): there is no bare `timeout=`
field either (also confirmed via ChatOllama.model_fields - and
model_config.get("extra") == "ignore" swallows it silently, no error).
Passing `timeout=300000` here used to do NOTHING - the request had no
client-side timeout at all, so a fully hung Ollama backend (crashed,
network partition - not just "slow") would block forever with no way
to recover. The real mechanism is `client_kwargs={"timeout": <seconds>}`,
forwarded to the underlying ollama.Client -> httpx.Client. Set
generously (not tightly): observed real durations on the current
hardware/model range from ~90s to ~1500s (25 min) for a single call,
so anything shorter would start killing calls that are merely slow,
not actually stuck.
"""

from langchain_ollama import ChatOllama

from lms_chatbot.config import settings

# Seconds. A ceiling against a genuinely hung backend, not a latency
# budget - see the module docstring for why this can't be tight.
_REQUEST_TIMEOUT_SECONDS = 1800


def get_orchestrator_llm() -> ChatOllama:
    # num_predict is sized for the orchestrator's LARGEST final answer,
    # which it now phrases itself from raw sql_agent_tool results - a
    # "list all ..." question legitimately renders up to
    # settings.default_row_limit rows. 1024 was sized back when this
    # call only routed/answered small talk, and was observed cutting a
    # 100-row answer off mid-table, before the total it had fetched was
    # ever stated (plus gpt-oss spends part of the budget on hidden
    # reasoning, per the module docstring). orchestrator.py also checks
    # done_reason for a length cut-off as a backstop.
    return ChatOllama(
        model=settings.orchestrator_model,
        base_url=settings.ollama_url,
        reasoning=False,
        temperature=0,
        num_predict=8192,
        client_kwargs={"timeout": _REQUEST_TIMEOUT_SECONDS},
    )


def get_sql_llm() -> ChatOllama:
    # Multi-table JOIN SQL (see module docstring) needs more headroom
    # than a single-table SELECT - this is the agent most likely to hit
    # the empty-content/no-tool-call truncation failure.
    return ChatOllama(
        model=settings.sql_model,
        base_url=settings.ollama_url,
        temperature=0,
        num_predict=2048,
        reasoning=False,
        client_kwargs={"timeout": _REQUEST_TIMEOUT_SECONDS},
    )


def get_nlp_llm() -> ChatOllama:
    # 1024 was observed too tight for cloud reasoning models
    # (gpt-oss:20b-cloud) on larger result sets: the empty-content/
    # no-visible-output failure this module's docstring describes for
    # gpt-oss:120b-cloud reproduces here too, and got more likely once
    # nlp_agent.py started asking for a structured ANSWER:/SUGGESTIONS:
    # response, which needs more output room than a bare answer did.
    return ChatOllama(
        model=settings.nlp_model,
        base_url=settings.ollama_url,
        temperature=0,
        num_predict=2048,
        reasoning=False,
        client_kwargs={"timeout": _REQUEST_TIMEOUT_SECONDS},
    )
