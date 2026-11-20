from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> Any:
    """Pull the first balanced JSON object/array out of an LLM reply."""
    if not text:
        raise ValueError("empty model output")
    m = _FENCE.search(text)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    for cand in candidates:
        start = next((i for i, ch in enumerate(cand) if ch in "{["), None)
        while start is not None:
            end = _match(cand, start)
            if end is not None:
                try:
                    return json.loads(cand[start : end + 1])
                except json.JSONDecodeError:
                    pass
            start = next((i for i in range(start + 1, len(cand)) if cand[i] in "{["), None)
    raise ValueError("no valid JSON found in model output")


def _match(s: str, start: int) -> int | None:
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None
