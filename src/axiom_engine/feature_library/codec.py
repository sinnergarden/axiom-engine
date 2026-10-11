"""Canonical immutable file primitives shared by Feature and Research storage."""
from hashlib import sha256
import json
from pathlib import Path

def require(ok, message):
    if not ok: raise ValueError(message)

def digest(value):
    return 'sha256:' + sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def seal(value, key): return {**value, key: digest(value)}

def file_digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return 'sha256:' + h.hexdigest()

def file_fingerprint(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

def is_ref(value):
    return type(value) is str and len(value)==71 and value.startswith('sha256:') and all(
        c in '0123456789abcdef' for c in value[7:])
