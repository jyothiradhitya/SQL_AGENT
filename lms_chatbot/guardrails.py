"""
Shared identity/self-description guardrail for every LLM call in the
answer path (orchestrator direct answers, and agents/nlp_agent.py's own
answers, on the rare path where something still calls it directly).

Lives here rather than in orchestrator.py or nlp_agent.py so either can
import it without depending on the other.
"""

import logging

logger = logging.getLogger(__name__)

IDENTITY_STATEMENT = (
    '''I'm the Data Assistant provided by PharmCrux e-Solutions. I help authorized users of 
    PharmCrux client organizations retrieve and understand data from their organization's 
    connected systems.I only provide information that the current user is authorized to access.
    Data belonging to other PharmCrux clients or organizations is strictly isolated and must never
    be disclosed.I'm intended for data-related questions within the PharmCrux product, 
    not as a general-purpose AI assistant. Information about the underlying AI model, vendor, 
    infrastructure, or implementation is internal and should not be disclosed.
'''
)

# Phrases that, if they show up in an answer, mean the underlying base
# model has leaked its own training identity (vendor/model name)
# instead of staying on IDENTITY_STATEMENT. This is a code-level
# backstop, not just a prompt rule: "you must say X" is a prompt rule,
# not a control mechanism, and small local models (qwen2.5:3b here)
# don't reliably follow it - the same reasoning orchestrator.py's
# docstring gives for why sql/validator.py enforces its table
# allow-list in code rather than trusting the prompt.
#
# Deliberately NOT including generic phrases like "created by",
# "trained by", "developed by", "built on" - a plain substring match
# has no way to tell "I was created by OpenAI" apart from an ordinary
# database answer like "user oquser10 was created by the admin
# account". That's not a hypothetical: this schema has literal
# created_by/last_modified_by audit columns and "who created/trained
# user X" is a completely normal question in an LMS - every one of
# those phrases was a guaranteed, recurring false-positive trap in
# this specific app, not just a generic risk. The vendor names and
# self-description phrases below stay specific enough to not collide
# with real business data, so they carry the actual leak-detection
# weight here.
_IDENTITY_LEAK_MARKERS = (
    "openai", "chatgpt", "gpt-3", "gpt-4", "gpt3", "gpt4",
    "anthropic",
    "meta ai", "mistral ai", "qwen", "alibaba", "ollama",
    "language model", "large language model",
    "an ai model", "an ai assistant", "conversational ai",
)


def enforce_identity_guardrail(answer: str, *, source: str) -> str:
    """If `answer` leaks the underlying model/vendor identity, replace
    it outright with IDENTITY_STATEMENT - a partial leak surrounded by
    otherwise-fine text is still a leak, so this doesn't try to patch
    around it. `source` is diagnostic-only, identifying which caller
    triggered the block in logs.
    """
    lowered = answer.lower()
    if any(marker in lowered for marker in _IDENTITY_LEAK_MARKERS):
        logger.warning(
            "[GUARDRAIL] IDENTITY_LEAK_BLOCKED source=%s original=%r",
            source, answer[:2000],
        )
        return "Not allowed to answer that question. " + IDENTITY_STATEMENT
    return answer
