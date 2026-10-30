"""The ``[shot] person`` text the pipeline's prompts need ("add {person}, standing still facing the camera").

v1 has no captioner: the text comes from what the user typed, else the preset manifest's own description. This is
the one place a local captioning model would plug in later (look at ``photo`` and describe the person and clothes).
"""
from __future__ import annotations

import re
from pathlib import Path

MAX_CHARS = 300


def describe_person(photo: Path, wearing: str, default: str) -> str:
    """Return a short noun phrase for the person in ``photo``.

    ``wearing`` is the user's "describe what they're wearing" text (may be empty), so it is read as clothing:
    "a long red raincoat" -> "a person wearing a long red raincoat". ``default`` is the preset manifest's
    ``[shot] person``.
    """
    text = re.sub(r"[{}\[\]<>\"\\]", "", " ".join((wearing or "").split()))[:MAX_CHARS].strip(" .,;")
    if not text:
        return default
    low = text.lower()
    if low.startswith(("wearing ", "in ", "dressed ")):
        return f"a person {text}"
    return f"a person wearing {text}"
