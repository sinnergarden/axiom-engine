"""Small immutable JSON contracts; no artifact discovery or I/O.

JSON null is the missing numeric value (the in-memory equivalent of a NaN mask).
Numeric inputs are float64-compatible finite values; inf must be handled by the
caller explicitly. Unknown is a reserved wire object, including inside policies.
"""
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
import re

ABI = "axiom.feature/1"
SEMANTICS = "axiom.operators/1"


class ContractError(ValueError):
    """Malformed, unsupported, or inadmissible execution input."""


def require(ok, message):
    if not ok:
        raise ContractError(message)


def fields(obj, names):
    require(type(obj) is dict and set(obj) == set(names.split()),
            f"Expected exact fields: {names}")


def text(value):
    require(type(value) is str and bool(value.strip()), "Expected nonempty string")


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Expected finite number")


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "Invalid integer policy")


def session(value):
    try:
        require(type(value) is str and date.fromisoformat(value).isoformat() == value,
                "Expected ISO session")
    except (ValueError, TypeError) as exc:
        raise ContractError("Invalid ISO session") from exc


def timestamp(value):
    try:
        require(type(value) is str and len(value) == 20 and value.endswith('Z'),
                "Use UTC YYYY-MM-DDTHH:MM:SSZ")
        datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
    except (ValueError, TypeError) as exc:
        raise ContractError("Invalid UTC timestamp") from exc


def digest(value):
    require(type(value) is str and re.fullmatch(r'sha256:[0-9a-f]{64}', value) is not None,
            "Expected immutable sha256 binding")


def _pairs(pairs):
    out = {}
    for k, v in pairs:
        require(k not in out, "Duplicate JSON key: " + k)
        out[k] = v
    return out


def _walk(value):
    if type(value) is dict:
        require(all(type(k) is str for k in value), "JSON object keys must be strings")
        if 'contract_type' in value:
            require(value['contract_type'] == 'Unknown', "Unsupported reserved contract_type")
            fields(value, 'contract_type contract_version metadata unknown_id reason required_evidence')
            require(value['contract_version'] == '1', "Unknown wire version")
            for k in ('unknown_id', 'reason', 'required_evidence'):
                text(value[k])
            require(type(value['metadata']) is dict, "Unknown metadata must be an object")
            yield value
        for v in value.values():
            yield from _walk(v)
    elif type(value) in (list, tuple):
        for v in value:
            yield from _walk(v)
    else:
        require(value is None or type(value) in (str, bool, int, float), "Not a JSON value")
        if type(value) in (int, float):
            number(value)


def unresolved(value):
    if isinstance(value, Document):
        value = value.to_dict()
    found = {}
    for obj in _walk(value):
        key = obj['unknown_id']
        require(key not in found or found[key] == obj, "Conflicting Unknown: " + key)
        found[key] = obj
    return tuple(found.values())


def canonical(value):
    tuple(_walk(value))
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


@dataclass(frozen=True)
class Document:
    payload: str

    def __post_init__(self):
        require(type(self.payload) is str, "Use from_dict for object inputs")
        try:
            value = json.loads(self.payload, object_pairs_hook=_pairs)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
        require(type(value) is dict, "Contract must be an object")
        object.__setattr__(self, 'payload', canonical(value))

    @classmethod
    def from_dict(cls, value):
        return cls(canonical(value))

    def to_dict(self):
        return json.loads(self.payload)

    @property
    def identity(self):
        return 'sha256:' + hashlib.sha256(self.payload.encode()).hexdigest()


class FactBatch(Document):
    """Explicit daily rows and fixed event streams with source bindings."""


class FeaturePlan(Document):
    """Frozen sequential single-assignment plan; construction permits UNKNOWN drafts."""


class ExecutionContext(Document):
    """Calendar, per-session cutoff, history keys, output keys and reference rows."""


class FeatureFrame(Document):
    """Ordered output, per-cell availability/validity and unchanged source bindings."""


def schema(columns):
    require(type(columns) is list and bool(columns), "Empty schema")
    names = []
    for c in columns:
        fields(c, 'name dtype unit stage missing')
        text(c['name']); text(c['unit'])
        require(c['dtype'] in ('float64', 'bool', 'string', 'date'), "Unsupported dtype")
        require(c['stage'] in ('fact', 'base', 'cross_sectional'), "Unsupported stage")
        require(c['missing'] in ('preserve', 'reject'), "Unsupported column missing policy")
        names.append(c['name'])
    require(len(set(names)) == len(names), "Duplicate schema column")
    return names


def check_value(value, column):
    if value is None:
        require(column['missing'] == 'preserve', "Missing value in required column")
    elif column['dtype'] == 'float64':
        number(value)
    elif column['dtype'] == 'bool':
        require(type(value) is bool, "Expected bool")
    elif column['dtype'] == 'date':
        session(value)
    else:
        text(value)
