"""
Central logging setup.

Call `setup_logging()` exactly once, at the entrypoint of whichever
process is starting (main.py for the CLI, mcp_server/server.py for the
tool server subprocess). Every other module just does:

    import logging
    logger = logging.getLogger(__name__)

and logs normally; the handler/format/color is configured here so it
never has to be repeated.

Every component's logger is colored by name (see _COMPONENT_COLORS) so
a fast-scrolling terminal with both the CLI and the MCP server's
subprocess logging interleaved can still be told apart at a glance -
retrieval in green, the SQL agent in magenta, the orchestrator in
cyan, the MCP server/client in yellow, SQL validation/execution in
red, everything else in the default color. Colors are skipped
automatically when the stream isn't a real terminal (e.g. piped to
`tee` or a file), so log files stay plain text.
"""

import logging
import os
import sys

from dotenv import dotenv_values

from lms_chatbot.config import settings

_CONFIGURED = False

_RESET = "\033[0m"

_LEVEL_COLORS = {
    "DEBUG": "90",  # bright black / gray - detail, not signal
    "INFO": "37",  # default
    "WARNING": "33",  # yellow
    "ERROR": "31",  # red
    "CRITICAL": "97;41",  # white on red
}

# Longest-prefix-matched against the logger name (e.g. "lms_chatbot.
# retrieval.pipeline" matches the "lms_chatbot.retrieval" entry).
_COMPONENT_COLORS = {
    "lms_chatbot.orchestrator": "36",  # cyan - top-level routing
    "lms_chatbot.agents.sql_agent": "35",  # magenta - SQL agent loop
    "lms_chatbot.agents.nlp_agent": "34",  # blue - answer phrasing
    "lms_chatbot.agents.llms": "34",
    "lms_chatbot.retrieval": "32",  # green - schema retrieval
    "lms_chatbot.mcp_server": "33",  # yellow - MCP tool server
    "lms_chatbot.mcp_client": "33",
    "lms_chatbot.sql": "31",  # red - validation/execution, where a bad query dies
    "lms_chatbot.startup_checks": "37",
    "__main__": "36",
}

# Protocol-plumbing loggers: near-zero debugging value for "what did the
# LLM get/return", extremely high line volume at DEBUG (a single LLM call
# produces a dozen+ httpcore lines, every MCP round-trip logs its own
# request/response envelope). Always quiet, regardless of LOG_LEVEL - if
# you specifically need to debug the HTTP/MCP transport itself, comment
# the relevant name out here rather than raising LOG_LEVEL, which would
# also unmute them everywhere else.
_ALWAYS_QUIET = ("httpcore", "mcp.server.lowlevel.server", "mcp.client", "asyncio")


def _component_color(name: str) -> str:
    best_prefix = ""
    best_color = "37"
    for prefix, color in _COMPONENT_COLORS.items():
        if (name == prefix or name.startswith(prefix + ".")) and len(prefix) > len(best_prefix):
            best_prefix, best_color = prefix, color
    return best_color


class _ColorFormatter(logging.Formatter):
    def __init__(self, datefmt: str, use_color: bool):
        super().__init__(datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            message = f"{message}\n{record.exc_text}"

        timestamp = self.formatTime(record, self.datefmt)
        level = record.levelname
        name = record.name

        if not self.use_color:
            return f"{timestamp} {level:<8} {name}: {message}"

        level_color = _LEVEL_COLORS.get(level, "37")
        name_color = _component_color(name)
        return (
            f"\033[90m{timestamp}{_RESET} "
            f"\033[{level_color}m{level:<8}{_RESET} "
            f"\033[{name_color}m{name}{_RESET}: {message}"
        )


def setup_logging(stream=sys.stderr) -> None:
    global _CONFIGURED

    if _CONFIGURED:
        return

    # Chat mode (main.py without --debug) sets LMS_LOG_FILE so logs from
    # this process AND the MCP server subprocess (which inherits the
    # environment) go to a file instead of burying the conversation.
    log_file = os.getenv("LMS_LOG_FILE")
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        use_color = False
    else:
        use_color = hasattr(stream, "isatty") and stream.isatty()
        handler = logging.StreamHandler(stream)
    handler.setFormatter(_ColorFormatter(datefmt="%H:%M:%S", use_color=use_color))

    resolved_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.addHandler(handler)

    for noisy in _ALWAYS_QUIET:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Quiet down noisy third-party loggers unless the user asked for DEBUG.
    if settings.log_level.upper() != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    _CONFIGURED = True

    # Self-report what actually got resolved and from where. load_dotenv()
    # (config.py) does NOT override a LOG_LEVEL already present in the
    # shell - if your .env says DEBUG but a stray `$env:LOG_LEVEL` from an
    # earlier session is still set, the .env value is silently ignored
    # with no other visible symptom besides "DEBUG lines don't show up".
    # dotenv_values() (unlike load_dotenv()) only *reads* the file without
    # touching os.environ, so comparing the two tells us which one won.
    env_value = os.environ.get("LOG_LEVEL")
    dotenv_value = dotenv_values().get("LOG_LEVEL")
    if env_value != dotenv_value:
        source = f"shell environment (LOG_LEVEL={env_value!r}) - OVERRIDING .env's LOG_LEVEL={dotenv_value!r}"
    elif env_value is not None:
        source = f".env (LOG_LEVEL={env_value!r})"
    else:
        source = "default (LOG_LEVEL not set anywhere)"

    logging.getLogger(__name__).info(
        "Logging configured: level=%s, source=%s, color=%s",
        logging.getLevelName(resolved_level), source, use_color,
    )
