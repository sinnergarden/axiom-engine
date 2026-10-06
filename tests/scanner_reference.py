"""Frozen scalar collector from Engine merge13d2801, for bounded equivalence checks.

Only scalar collection changed; the existing grammar remains shared.
"""
import json
from axiom_engine.core.contracts import ContractError, canonical, require, _pairs
from axiom_engine.runtime.stock_stream_inputs import _CanonicalIndex

def _scalar(self):
    stage = self._memory.stage()
    raw = bytearray()
    def take():
        stage.reserve(1)
        raw.append(self._take())
    try:
        if self._peek() == 34:
            take(); escape = False
            while True:
                byte = self._peek()
                require(byte is not None, "Truncated stock JSON string")
                take()
                if byte == 34 and not escape:
                    break
                escape = byte == 92 and not escape
        else:
            while self._peek() not in (None, 44, 93, 125):
                take()
        size = len(raw)
        # Raw, decoded scalar, canonical text and encoded comparison can
        # coexist. Reserve that growth before invoking either decoder.
        stage.reserve(3*size)
        try:
            value = json.loads(raw, object_pairs_hook=_pairs)
        except (ValueError, UnicodeError) as exc:
            raise ContractError("Malformed stock JSON scalar") from exc
        require(canonical(value).encode() == raw, "Noncanonical stock JSON scalar")
        raw = None
        stage.release(3*size)
        return value, stage
    except Exception:
        stage.close()
        raise

class ReferenceIndex(_CanonicalIndex):
    _scalar = _scalar
