from collections import deque
import time
from typing import Deque, Optional, Tuple

SeenSignature = Tuple[str, float, str]

_seen: Deque[SeenSignature] = deque(maxlen=100)


def check_duplicate(sig: str, window: int = 600) -> Optional[str]:
    now = time.time()
    for stored_sig, ts, call_id in _seen:
        if stored_sig == sig and (now - ts) < window:
            return call_id
    return None


def register_signature(sig: str, call_id: str) -> None:
    _seen.append((sig, time.time(), call_id))
