"""Sample calls live in calls/*.json: an id, a channel, a transcript, and for the offline
backend a table of expected answers a person wrote after reading it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_calls(path: str | Path) -> list[dict[str, Any]]:
    """A directory of .json calls, or of .wav recordings, or one file of either. A
    recording is routed from the audio itself; its .json twin, if there is one next to
    the calls directory, is kept only so the table can show what was said."""
    p = Path(path)
    files = sorted(list(p.glob("*.json")) + list(p.glob("*.wav"))) if p.is_dir() else [p]
    calls = []
    for f in files:
        if f.suffix == ".wav":
            twin = next((t for t in (f.with_suffix(".json"), f.parent.parent / f.with_suffix(".json").name) if t.exists()), None)
            meta = json.loads(twin.read_text()) if twin else {}
            calls.append({"id": f.stem, "channel": "phone", "audio": str(f), "transcript": meta.get("transcript", ""), "file": f.name})
        else:
            calls.append(json.loads(f.read_text()) | {"file": f.name})
    return calls
