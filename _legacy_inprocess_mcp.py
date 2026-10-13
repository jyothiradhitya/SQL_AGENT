from langchain_core.tools import StructuredTool

from SQL_AGENT.agent_test import run_sql_agent
from NLP_AGENT.nlp_agent import run_nlp_agent


# ============================================================
# PER-REQUEST ORCHESTRATION STATE
#
# The previous version kept _sql_called / _nlp_called / etc.
# as MODULE-LEVEL globals. That works for a single-user CLI
# loop, but this is meant to become an enterprise chatbot that
# will serve multiple concurrent users/requests. Module globals
# are shared by every request in the process, so:
#
#   Request A calls sql_agent_tool -> _sql_called = True
#   Request B (running concurrently) calls sql_agent_tool
#     -> sees _sql_called == True and gets Request A's
#        cached _sql_result instead of running its own query.
#
# That's a real cross-user data leak, not just a cosmetic bug.
#
# Fix: state now lives on a small per-request session object.
# Each request creates its own session via new_session() and
# gets back tool instances that close over THAT session only.
# ============================================================

class _AgentSession:

    def __init__(self):
        self.sql_called = False
        self.nlp_called = False
        self.sql_result = None


def new_session():
    """
    Create a fresh, isolated orchestration session for one
    user request, plus the tool instances bound to it.

    Usage:

        session, tools = new_session()
        agent_executor = build_agent(tools)   # your LLM/agent code
        ...
    """

    session = _AgentSession()

    # --------------------------------------------------------
    # SQL AGENT TOOL
    # --------------------------------------------------------

    def sql_agent_tool(question: str) -> str:
        """
        Retrieve database information.

        Use this exactly once when the user's question
        requires database information.

        The SQL agent itself:
        - generates SQL
        - validates SQL
        - executes SQL
        - returns database data

        Do not call this tool again after it succeeds.
        """

        print()
        print("=" * 60)
        print("[MCP] SQL AGENT TOOL")
        print("=" * 60)

        print("[MCP] Question:")
        print(question)

        # ----------------------------------------------------
        # HARD DUPLICATE GUARD (scoped to this session only)
        # ----------------------------------------------------

        if session.sql_called:

            print("[MCP] SQL agent was already called")
            print("[MCP] Returning cached database result")

            return (
                "SQL_ALREADY_EXECUTED\n"
                "The database query has already been executed.\n"
                "Do NOT call sql_agent_tool again.\n"
                "Use the database result already returned."
                "\n\n"
                + (session.sql_result or "")
            )

        session.sql_called = True

        print("[MCP] Calling SQL agent")

        try:

            result = run_sql_agent(
                question
            )

            session.sql_result = result

            print("[MCP] SQL agent completed")
            print("[MCP] Database result returned")

            return (
                "SQL_EXECUTION_COMPLETE\n"
                "DATABASE_RESULT:\n"
                + result
                + "\n\n"
                "IMPORTANT:\n"
                "The SQL agent has already completed.\n"
                "Do NOT call sql_agent_tool again."
            )

        except Exception as e:

            print("[MCP] SQL agent failed")
            print(f"[MCP] ERROR: {e}")

            # Allow a retry on genuine failure — only a SUCCESS
            # is guarded against repeats.
            session.sql_called = False

            return (
                "SQL_AGENT_ERROR\n"
                f"{str(e)}"
            )

    # --------------------------------------------------------
    # NLP AGENT TOOL
    # --------------------------------------------------------

    def nlp_agent_tool(
        question: str,
        database_data: str,
    ) -> str:
        """
        Convert database results into the final natural-language
        answer.

        Use this exactly once after sql_agent_tool succeeds.
        """

        print()
        print("=" * 60)
        print("[MCP] NLP AGENT TOOL")
        print("=" * 60)

        print("[MCP] Question:")
        print(question)

        # ----------------------------------------------------
        # HARD DUPLICATE GUARD (scoped to this session only)
        # ----------------------------------------------------

        if session.nlp_called:

            print("[MCP] NLP agent was already called")

            return (
                "NLP_ALREADY_EXECUTED\n"
                "Do not call nlp_agent_tool again."
            )

        session.nlp_called = True

        print("[MCP] Calling NLP agent")

        try:

            result = run_nlp_agent(
                question,
                database_data,
            )

            print("[MCP] NLP agent completed")

            return (
                "NLP_FINAL_ANSWER\n"
                + result
            )

        except Exception as e:

            print("[MCP] NLP agent failed")
            print(f"[MCP] ERROR: {e}")

            session.nlp_called = False

            return (
                "NLP_AGENT_ERROR\n"
                f"{str(e)}"
            )

    tools = [
        StructuredTool.from_function(
            sql_agent_tool,
            name="sql_agent_tool",
        ),
        StructuredTool.from_function(
            nlp_agent_tool,
            name="nlp_agent_tool",
        ),
    ]

    return session, tools
