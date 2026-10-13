import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL")

if not OLLAMA_URL:
    raise ValueError("OLLAMA_URL is not defined in .env")

MODEL = "qwen2.5:3b"


# ============================================================
# NLP LLM
# ============================================================

llm = ChatOllama(
    model=MODEL,
    base_url=OLLAMA_URL,
    temperature=0,
    num_predict=250,
)


# ============================================================
# NLP PROMPT
# ============================================================

NLP_SYSTEM_PROMPT = """
You are a database answer-generation agent.

Your ONLY job is to convert database results
into a natural-language answer.

You receive:

1. USER QUESTION
2. DATABASE DATA

Use ONLY the database data provided.

Rules:

1. Never invent information.

2. Never query a database.

3. Never generate SQL.

4. Never assume information that is not in DATA.

5. Answer exactly what the user asked.

6. If DATA says ROWS_FOUND: 0,
   respond:
   No matching records were found.

7. For lists, directly provide the requested values.

8. Do not describe the JSON structure.

9. Do not say that the data is JSON.

10. Do not explain internal processing.

11. Keep the answer concise.

12. If the database result does not contain
    enough information to answer the question,
    clearly say that the requested information
    could not be found.

Return ONLY the final natural-language answer.
"""


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def run_nlp_agent(
    question: str,
    database_data: str,
) -> str:

    print()
    print("=" * 70)
    print("[NLP AGENT] START")
    print("=" * 70)

    print("[NLP AGENT] QUESTION:")
    print(question)

    print()
    print("[NLP AGENT] DATABASE DATA:")
    print(database_data)

    prompt = f"""
USER QUESTION
=============

{question}


DATABASE DATA
=============

{database_data}


Generate the final answer.
"""


    print()
    print("[NLP AGENT] Calling Ollama")

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": NLP_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )

    answer = response.content.strip()

    print("[NLP AGENT] Ollama completed")

    print()
    print("[NLP AGENT] FINAL ANSWER:")
    print(answer)

    print("=" * 70)

    return answer