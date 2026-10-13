"""
CLI entrypoint. Launches the MCP server as a subprocess, keeps one
connection open for the life of the CLI session, and hands each
question to the top-level orchestrator agent (see orchestrator.py).

Run from the repo root:
    uv run python -m lms_chatbot.main            # chat (logs -> logs/chatbot.log)
    uv run python -m lms_chatbot.main --debug    # chat with logs on the console
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Chat mode keeps the conversation readable: every log line (this process
# and the MCP server subprocess, which inherits the environment) goes to
# a file. Must be decided before setup_logging() runs below.
DEBUG = "--debug" in sys.argv[1:]
LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "chatbot.log"
if not DEBUG:
    os.environ["LMS_LOG_FILE"] = str(LOG_FILE)

from lms_chatbot.logging_config import setup_logging  # noqa: E402
from lms_chatbot.mcp_client.client import MCPClient  # noqa: E402
from lms_chatbot.orchestrator import answer_question  # noqa: E402
from lms_chatbot.startup_checks import StartupCheckError, run_startup_checks  # noqa: E402

# Windows' console defaults to the cp1252 codepage, which can't encode
# characters an LLM answer routinely contains (narrow no-break spaces,
# smart hyphens/quotes, ...) - observed crashing print(answer) outright
# with UnicodeEncodeError on a real gpt-oss:120b-cloud response using
# U+202F and U+2011. UTF-8 with errors="replace" never crashes on an
# unencodable character; at worst it substitutes one visibly instead of
# taking the whole CLI down.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

setup_logging()
logger = logging.getLogger(__name__)

WELCOME = """\
Hi! I'm your data assistant. Ask me anything about your records -
change controls, requests, inventory, equipment, users and more.

  new   start a fresh conversation
  exit  quit
"""


async def _thinking(stop: asyncio.Event) -> None:
    """A small "Thinking..." indicator while an answer is being worked out."""
    frames = ["Thinking   ", "Thinking.  ", "Thinking.. ", "Thinking..."]
    i = 0
    while not stop.is_set():
        print(f"\rAssistant: {frames[i % len(frames)]}", end="", flush=True)
        i += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.4)
        except asyncio.TimeoutError:
            pass
    print("\r" + " " * 30 + "\r", end="", flush=True)


async def _ask(question: str, mcp_client, history) -> str:
    stop = asyncio.Event()
    spinner = asyncio.create_task(_thinking(stop))
    try:
        return await answer_question(question, mcp_client, history)
    finally:
        stop.set()
        await spinner


async def main() -> None:
    try:
        run_startup_checks()
    except StartupCheckError as e:
        print(f"I couldn't start up:\n{e}", file=sys.stderr)
        sys.exit(1)

    print("Starting up...", end="", flush=True)
    async with MCPClient() as mcp_client:
        tools = await mcp_client.list_tools()
        logger.info("Connected to MCP server. Tools: %s", [t.name for t in tools])
        print("\r" + " " * 20 + "\r", end="")
        print(WELCOME)
        if not DEBUG:
            print(f"(Logs: {LOG_FILE} - run with --debug to see them here)\n")

        # Whole-session conversation memory, as (question, answer) pairs
        # in order asked - answer_question() only ever looks at the last
        # settings.conversation_history_turns of these, so this can just
        # grow for the life of the CLI session without pruning here.
        history: list[tuple[str, str]] = []

        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not question:
                continue

            command = question.lower()
            if command in {"exit", "quit", "bye"}:
                print("Assistant: Goodbye!")
                break
            if command in {"new", "clear", "reset"}:
                history.clear()
                print("Assistant: Sure - starting a fresh conversation.\n")
                continue

            try:
                answer = await _ask(question, mcp_client, history)
            except (KeyboardInterrupt, asyncio.CancelledError):
                # Both are BaseException subclasses (CancelledError since
                # Python 3.8), not Exception - `except Exception` below
                # never catches Ctrl+C during an in-flight (typically
                # multi-minute, on this hardware) LLM call, so it must be
                # handled explicitly or it crashes out through asyncio.run()
                # with a raw traceback instead of exiting cleanly.
                print("\nAssistant: Stopped. Goodbye!")
                break
            except Exception:
                logger.exception("Unhandled error answering question")
                answer = (
                    "Sorry, something went wrong while I was working on that. "
                    "Could you try again, or ask it a little differently?"
                )
            else:
                # Only a real answer joins history - a failed turn isn't
                # something later questions should be able to refer back
                # to as if it had actually said something.
                history.append((question, answer))

            print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
