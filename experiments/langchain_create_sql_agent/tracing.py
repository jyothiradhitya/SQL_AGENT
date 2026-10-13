"""
Callback handler that turns every LLM call, tool call, SQL query and
final answer inside a single AgentExecutor.invoke() into both a live
log line (so you can watch the trajectory as it happens) and a
structured record (so the run can be compared/replayed afterwards).

verbose=True on the AgentExecutor already prints a rough trace, but it
prints to stdout in whatever format that AgentExecutor version chooses
and captures nothing programmatically. This handler is the source of
truth for the "record, for every question: tables inspected, tools
called, SQL generated, SQL result/error, LLM call count, final answer"
requirement - verbose=True stays on only as a human-readable companion.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger("experiments.langchain_create_sql_agent.trace")

_TRUNCATE_LEN = 3000


def _truncated(value: Any, limit: int = _TRUNCATE_LEN) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated, {len(text)} chars total]"


_SQL_TOOL_NAMES = {"sql_db_query", "sql_db_query_checker"}
_SCHEMA_TOOL_NAMES = {"sql_db_schema", "sql_db_list_tables"}


@dataclass
class ToolCallRecord:
    name: str
    args: str
    output: str
    duration_s: float
    status: str  # "SUCCESS" | "ERROR"


@dataclass
class LLMCallRecord:
    duration_s: float
    content: str


@dataclass
class QuestionTrace:
    question: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tables_inspected: set[str] = field(default_factory=set)
    sql_queries: list[dict] = field(default_factory=list)
    final_answer: str | None = None
    error: str | None = None
    agent_duration_s: float | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "tables_inspected": sorted(self.tables_inspected),
            "tool_calls": [
                {
                    "name": tc.name,
                    "args": tc.args,
                    "output": tc.output,
                    "duration_s": round(tc.duration_s, 3),
                    "status": tc.status,
                }
                for tc in self.tool_calls
            ],
            "sql_queries": self.sql_queries,
            "llm_call_count": len(self.llm_calls),
            "llm_total_duration_s": round(sum(c.duration_s for c in self.llm_calls), 3),
            "agent_duration_s": (
                round(self.agent_duration_s, 3)
                if self.agent_duration_s is not None
                else None
            ),
            "final_answer": self.final_answer,
            "error": self.error,
        }


def _tool_input_to_str(tool_input: Any) -> str:
    if isinstance(tool_input, dict):
        # AgentAction.tool_input is usually {"query": "..."} for the
        # tool-calling agent type; flatten to the raw string when there
        # is exactly one arg, which is the case for every SQL toolkit tool.
        if len(tool_input) == 1:
            return str(next(iter(tool_input.values())))
        return str(tool_input)
    return str(tool_input)


class SQLAgentTracer(BaseCallbackHandler):
    """One instance per question - do not reuse across invoke() calls."""

    def __init__(self, question: str):
        super().__init__()
        self.trace = QuestionTrace(question=question)
        self._llm_starts: dict[UUID, float] = {}
        self._tool_starts: dict[UUID, tuple[str, str, float]] = {}

    # -- LLM -----------------------------------------------------------

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, **kwargs
    ) -> None:
        self._llm_starts[run_id] = time.monotonic()
        n_messages = sum(len(batch) for batch in messages)
        logger.info("[LLM] call START messages=%d", n_messages)
        logger.debug("[LLM] messages=%s", _truncated(messages))

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs
    ) -> None:
        if run_id in self._llm_starts:
            return
        self._llm_starts[run_id] = time.monotonic()
        logger.info("[LLM] call START prompts=%d", len(prompts))

    def on_llm_end(
        self, response: LLMResult, *, run_id, parent_run_id=None, **kwargs
    ) -> None:
        start = self._llm_starts.pop(run_id, None)
        duration = time.monotonic() - start if start is not None else 0.0

        content = ""
        tool_calls = []
        try:
            generation = response.generations[0][0]
            message = getattr(generation, "message", None)
            content = getattr(message, "content", "") or getattr(
                generation, "text", ""
            )
            tool_calls = getattr(message, "tool_calls", []) or []
        except (IndexError, AttributeError):
            pass

        self.trace.llm_calls.append(
            LLMCallRecord(duration_s=duration, content=str(content))
        )

        if tool_calls:
            logger.info(
                "[LLM] call END duration=%.2fs tool_calls=%s",
                duration, [tc.get("name") for tc in tool_calls],
            )
        else:
            logger.info("[LLM] call END duration=%.2fs content=%r", duration, _truncated(content, 300))

    def on_llm_error(self, error: BaseException, *, run_id, parent_run_id=None, **kwargs) -> None:
        self._llm_starts.pop(run_id, None)
        logger.warning("[LLM] call FAILED error=%s", error)

    # -- Tools -----------------------------------------------------------

    def on_tool_start(
        self, serialized, input_str, *, run_id, parent_run_id=None, inputs=None, **kwargs
    ) -> None:
        name = (serialized or {}).get("name", "unknown_tool")
        self._tool_starts[run_id] = (name, input_str, time.monotonic())
        logger.info("[TOOL] %s START args=%r", name, _truncated(input_str, 500))

        if name in _SCHEMA_TOOL_NAMES:
            for part in input_str.split(","):
                part = part.strip()
                if part:
                    self.trace.tables_inspected.add(part)

    def on_tool_end(self, output: Any, *, run_id, parent_run_id=None, **kwargs) -> None:
        started = self._tool_starts.pop(run_id, None)
        name, input_str, start = started if started else ("unknown_tool", "", time.monotonic())
        duration = time.monotonic() - start

        output_str = getattr(output, "content", output)
        output_str = str(output_str)
        is_error = output_str.strip().lower().startswith("error")
        status = "ERROR" if is_error else "SUCCESS"

        logger.info("[TOOL] %s END status=%s duration=%.2fs", name, status, duration)
        logger.debug("[TOOL] %s output=%s", name, _truncated(output_str))

        self.trace.tool_calls.append(
            ToolCallRecord(
                name=name,
                args=_truncated(input_str, 1000),
                output=_truncated(output_str),
                duration_s=duration,
                status=status,
            )
        )

        if name in _SQL_TOOL_NAMES:
            self.trace.sql_queries.append(
                {
                    "tool": name,
                    "sql": input_str,
                    "result_or_error": _truncated(output_str),
                    "status": status,
                }
            )

    def on_tool_error(self, error: BaseException, *, run_id, parent_run_id=None, **kwargs) -> None:
        started = self._tool_starts.pop(run_id, None)
        name, input_str, start = started if started else ("unknown_tool", "", time.monotonic())
        duration = time.monotonic() - start

        logger.warning("[TOOL] %s RAISED duration=%.2fs error=%s", name, duration, error)

        self.trace.tool_calls.append(
            ToolCallRecord(
                name=name,
                args=_truncated(input_str, 1000),
                output=f"EXCEPTION: {error}",
                duration_s=duration,
                status="ERROR",
            )
        )
        if name in _SQL_TOOL_NAMES:
            self.trace.sql_queries.append(
                {
                    "tool": name,
                    "sql": input_str,
                    "result_or_error": f"EXCEPTION: {error}",
                    "status": "ERROR",
                }
            )

    # -- Agent -----------------------------------------------------------

    def on_agent_action(self, action: AgentAction, *, run_id, parent_run_id=None, **kwargs) -> None:
        logger.info(
            "[AGENT] ACTION tool=%s input=%r",
            action.tool, _truncated(_tool_input_to_str(action.tool_input), 300),
        )

    def on_agent_finish(self, finish: AgentFinish, *, run_id, parent_run_id=None, **kwargs) -> None:
        answer = finish.return_values.get("output", "")
        self.trace.final_answer = answer
        logger.info("[AGENT] FINISH answer=%r", _truncated(answer, 500))
