"""
Benchmark experiment: LangChain's `create_sql_agent()` against the real
PostgreSQL database and the local Ollama SQL model, with no semantic KB
and no curated schema retrieval - just the vanilla toolkit LangChain
ships (sql_db_list_tables / sql_db_schema / sql_db_query_checker /
sql_db_query) pointed at the raw database.

This package is intentionally isolated from lms_chatbot/ and SQL_AGENT/:
it only reads connection settings from lms_chatbot.config, it never
imports the production agent, retrieval, or MCP code. Run with:

    python -m experiments.langchain_create_sql_agent
"""
