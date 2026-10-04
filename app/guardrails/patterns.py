import re

# Attempts to override the system prompt or escape the assistant's role,
# whether coming from the user's question or (via content_guard) retrieved text.
INJECTION_PATTERNS = [
    re.compile(r"ignore (all |any )?(previous|prior|above|earlier) instructions", re.IGNORECASE),
    re.compile(r"disregard (the |your )?(system prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"reveal (your |the )?(system prompt|instructions)", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"act as (a|an) ", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"pretend (you|to) (are|be)", re.IGNORECASE),
    re.compile(r"do anything now", re.IGNORECASE),
    re.compile(r"\bDAN\b"),
]

# Requests for credentials, internal configuration, or raw database access —
# none of which this system should ever have a legitimate reason to answer.
SENSITIVE_DATA_PATTERNS = [
    re.compile(r"\bpassword(s)?\b", re.IGNORECASE),
    re.compile(r"\bapi[ _-]?key(s)?\b", re.IGNORECASE),
    re.compile(r"\bsecret[ _-]?key(s)?\b", re.IGNORECASE),
    re.compile(r"\bcredentials?\b", re.IGNORECASE),
    re.compile(r"connection string", re.IGNORECASE),
    re.compile(r"database (schema|credentials|password)", re.IGNORECASE),
    re.compile(r"\benv(ironment)? variables?\b", re.IGNORECASE),
    re.compile(r"select\s+\*\s+from", re.IGNORECASE),
    re.compile(r"drop\s+table", re.IGNORECASE),
    re.compile(r"union\s+select", re.IGNORECASE),
]


def matches_any(text: str, patterns: list[re.Pattern]) -> str | None:
    """Returns the pattern string of the first match, or None. The specific
    pattern is returned (not just a bool) so callers/logs can say *why*
    something was flagged."""
    for pattern in patterns:
        if pattern.search(text):
            return pattern.pattern
    return None
