import os
import sys

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from SQL_AGENT.agent_test import run_sql_agent


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")

if not OLLAMA_URL:
    raise ValueError(
        "OLLAMA_URL is not defined in .env"
    )

MODEL = "qwen2.5:3b"

VERBOSE = (
    "-v" in sys.argv
    or "--verbose" in sys.argv
)


# ============================================================
# ORCHESTRATOR LLM
# ============================================================

orchestrator_llm = ChatOllama(
    model=MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
    num_predict=50,
)


# ============================================================
# ORCHESTRATOR PROMPT
# ============================================================

ORCHESTRATOR_PROMPT = """
You are a routing classifier.

Your ONLY job is to decide whether the user's question
requires information from the database.

Return exactly ONE word.

DATABASE

or

DIRECT

Return DATABASE if the question requires
information stored in the database.

Return DIRECT if the question can be answered
without database information.

Do not explain.

Do not generate SQL.

Do not answer the question.

Examples:

"list all users"
DATABASE

"show inactive employees"
DATABASE

"how many students are there"
DATABASE

"what is the capital of India"
DIRECT

"hello"
DIRECT
"""


# ============================================================
# STARTUP
# ============================================================

print("=" * 70)
print("[ORCHESTRATOR] Creating classifier")
print("=" * 70)

print(
    "[ORCHESTRATOR] Model:",
    MODEL
)

print("[ORCHESTRATOR] Classifier ready")

print("=" * 70)


# ============================================================
# ROUTER
# ============================================================

def classify_question(question: str) -> str:

    print()
    print("=" * 70)
    print("[ORCHESTRATOR] ROUTING")
    print("=" * 70)

    print("[ORCHESTRATOR] QUESTION:")
    print(question)

    print()
    print("[ORCHESTRATOR] Calling Ollama classifier")

    response = orchestrator_llm.invoke(
        [
            {
                "role": "system",
                "content": ORCHESTRATOR_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]
    )

    decision = response.content.strip().upper()

    print(
        "[ORCHESTRATOR] Raw decision:",
        repr(decision)
    )

    # ========================================================
    # NORMALIZE DECISION
    # ========================================================

    if "DATABASE" in decision:

        final_route = "DATABASE"

    elif "DIRECT" in decision:

        final_route = "DIRECT"

    else:

        print(
            "[ORCHESTRATOR] Unknown classifier output"
        )

        print(
            "[ORCHESTRATOR] Safe default: DATABASE"
        )

        final_route = "DATABASE"

    print(
        "[ORCHESTRATOR] FINAL ROUTE:",
        final_route
    )

    return final_route


# ============================================================
# DIRECT ANSWER
# ============================================================

def answer_directly(
    question: str
) -> str:

    print()
    print("=" * 70)
    print("[ORCHESTRATOR] DIRECT WORKFLOW")
    print("=" * 70)

    print(
        "[ORCHESTRATOR] Calling Ollama "
        "for direct answer"
    )

    response = orchestrator_llm.invoke(
        [
            {
                "role": "system",
                "content": """
Answer the user's question directly.

Be concise.

Do not discuss:
- routing
- agents
- databases
- internal processing
""",
            },
            {
                "role": "user",
                "content": question,
            },
        ]
    )

    answer = response.content.strip()

    print()
    print("[ORCHESTRATOR] DIRECT ANSWER:")
    print(answer)

    return answer


# ============================================================
# DATABASE WORKFLOW
# ============================================================

def answer_from_database(
    question: str
) -> str:

    print()
    print("=" * 70)
    print("[ORCHESTRATOR] DATABASE WORKFLOW")
    print("=" * 70)

    # ========================================================
    # SQL / ANSWER AGENT
    # ========================================================

    print()
    print("[ORCHESTRATOR] STEP 1")
    print("[ORCHESTRATOR] Calling SQL/ANSWER agent")

    database_answer = run_sql_agent(
        question
    )

    print()
    print("[ORCHESTRATOR] SQL/ANSWER agent completed")

    if not database_answer:

        print(
            "[ORCHESTRATOR] Agent returned empty answer"
        )

        return (
            "I could not retrieve the requested "
            "database information."
        )

    print()
    print("=" * 70)
    print("[ORCHESTRATOR] DATABASE ANSWER")
    print("=" * 70)

    print(database_answer)

    print("=" * 70)

    return database_answer


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def run_agent(
    question: str
) -> str:

    print()
    print("=" * 70)
    print("[ORCHESTRATOR] START")
    print("=" * 70)

    print("[ORCHESTRATOR] USER QUESTION:")
    print(question)

    # ========================================================
    # STEP 1 — ROUTE
    # ========================================================

    route = classify_question(
        question
    )

    # ========================================================
    # DIRECT
    # ========================================================

    if route == "DIRECT":

        return answer_directly(
            question
        )

    # ========================================================
    # DATABASE
    # ========================================================

    return answer_from_database(
        question
    )


# ============================================================
# CLI
# ============================================================

while True:

    print()

    question = input(
        "Ask a question "
        "(type 'exit' to quit): "
    ).strip()

    if not question:
        continue

    if question.lower() in {
        "exit",
        "quit",
    }:

        print("Exiting.")
        break

    try:

        final_answer = run_agent(
            question
        )

        print()
        print("=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(final_answer)

        print("=" * 70)

    except KeyboardInterrupt:

        print()
        print(
            "[ORCHESTRATOR] Interrupted"
        )

        break

    except Exception as e:

        print()
        print("=" * 70)
        print("[ORCHESTRATOR] ERROR")
        print("=" * 70)

        print(
            type(e).__name__,
            ":",
            e
        )

        print("=" * 70)