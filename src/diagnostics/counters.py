from __future__ import annotations

DIAG_ENABLED: bool = False

_COUNTERS: dict[str, int] = {}


def incr(key: str) -> None:
    if not DIAG_ENABLED:
        return
    _COUNTERS[key] = _COUNTERS.get(key, 0) + 1


def snapshot_and_reset() -> dict[str, int]:
    snap = dict(_COUNTERS)
    _COUNTERS.clear()
    return snap


def set_enabled(enabled: bool) -> None:
    global DIAG_ENABLED
    DIAG_ENABLED = enabled
