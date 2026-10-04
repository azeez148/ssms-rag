import re


def clean_content(text: str) -> str:
    """Generic, domain-agnostic text normalization before chunking/hashing."""
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
