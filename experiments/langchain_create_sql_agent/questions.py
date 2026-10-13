"""
Deliberately ambiguous natural-language questions used to probe whether
the raw create_sql_agent() picks the right table when several tables
could plausibly answer the same everyday word.

Known ambiguity in this schema (see SQL_AGENT/Data/schema_tables.json
for the full table lists), which is exactly why each concept below is
worth asking about:

- "users"       -> pc.sys_user (system/login users) vs masters.party
                   (people/organizations) vs lms.tm ("tm" - training
                   member/trainee, not obviously named "user" at all).
- "products"    -> there is no bare "product" table at all, only
                   masters.product_store / product_store_type /
                   product_store_user_group. A correct agent should
                   notice this rather than guess.
- "stores"      -> masters.product_store vs product_store_type vs
                   product_store_user_group.
- "training"    -> lms.training, training_category, training_type,
                   training_schedule, training_session, training_tm,
                   user_training, training_session_register - many
                   candidates, several sound interchangeable.
- "attachments" -> lms.training_attachment vs lms.tm_attachment vs
                   lms.lms_attachment - three different attachment
                   tables in the same schema.
"""

DEFAULT_QUESTIONS: list[str] = [
    "List all the users in the system.",
    "How many products are there?",
    "Show me all the stores and how many products each one has.",
    "What are the different types of training available?",
    "Which trainings have attachments, and what are the attachment file names?",
    "How many users have completed a training?",
    "List all active users along with the trainings they are enrolled in.",
    "Show me all attachments related to training sessions.",
    "List all trainees along with their job codes.",
    "Which store has the most users assigned to it?",
]
