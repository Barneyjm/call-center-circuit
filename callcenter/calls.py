"""Sample calls live in calls/*.json: an id, a channel, a transcript, and for the offline
backend a table of expected answers a person wrote after reading it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_calls(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    return [json.loads(f.read_text()) | {"file": f.name} for f in files]
