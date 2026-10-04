import re

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def extract_cited_indices(answer: str, num_sources: int) -> set[int]:
    """Finds bracketed citation markers like [1], [2] in the answer text and
    returns the ones that correspond to an actual retrieved source (1-indexed).
    Out-of-range numbers (e.g. [99] when only 3 sources exist) are ignored —
    they're not a valid citation, whether from a model mistake or unrelated text.
    """
    found = {int(match) for match in _CITATION_PATTERN.findall(answer)}
    return {index for index in found if 1 <= index <= num_sources}
