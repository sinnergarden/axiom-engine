"""Exact existing content-addressed buffers; no Data, Research or math imports."""
from array import array
from hashlib import sha256
import math
import os
from pathlib import Path
import sys
import tempfile
from .codec import digest, file_digest, write_json, require, seal, file_fingerprint

DTYPES = {
    'float64_le': ('d', 8), 'bool_u8': ('B', 1),
    'int64_le': ('q', 8), 'uint64_le': ('Q', 8),
    'int32_le': ('i', 4), 'uint8': ('B', 1),
}
BUFFER_FIELDS = {'path', 'file_digest', 'dtype', 'shape', 'buffer_digest'}


def shape_size(shape):
    require(type(shape) is list and bool(shape) and all(type(n) is int and n >= 0
            for n in shape), 'explicit nonnegative integer buffer shape required')
    return math.prod(shape)


def _buffer_array(values, dtype):
    require(dtype in DTYPES, 'unsupported matrix buffer dtype')
    # Native column handoff stays columnar. Snapshot before writing; no mutable
    # caller can change the hash's input while publication is in progress.
    np=sys.modules.get('numpy')
    if np is not None and isinstance(values,np.ndarray):
        expected={'float64_le':'<f8','bool_u8':'u1','int32_le':'<i4',
                  'int64_le':'<i8','uint64_le':'<u8','uint8':'u1'}[dtype]
        require(values.ndim==1 and values.dtype==np.dtype(expected),'native matrix dtype/shape mismatch')
        require(bool(np.isfinite(values).all()),'finite native matrix values required')
        if dtype=='bool_u8':require(bool(((values==0)|(values==1)).all()),'native bool values must be0/1')
        owner = values
        while isinstance(owner, np.ndarray) and owner.base is not None:
            owner = owner.base
        if isinstance(owner, bytes) and values.flags.c_contiguous:
            return values
        return np.frombuffer(values.tobytes(),dtype=values.dtype)
    code, size = DTYPES[dtype]
    out = array(code)
    require(out.itemsize == size, 'platform raw buffer size mismatch')
    for value in values:
        if dtype == 'float64_le':
            require(type(value) in (int, float) and math.isfinite(value),
                    'finite binary64 value required')
        elif dtype == 'bool_u8':
            require(type(value) is bool, 'boolean matrix value required')
        else:
            require(type(value) is int, 'integer matrix value required')
            bits = size * 8
            low = -(1 << (bits-1)) if dtype.startswith('int') else 0
            high = (1 << (bits-1)) if dtype.startswith('int') else (1 << bits)
            require(low <= value < high, 'matrix integer out of range')
        out.append(value)
    if sys.byteorder != 'little' and size > 1:
        out.byteswap()
    return out


def _verify_written(path, expected, marks=None, *, _file_hasher=None):
    """Bind the final byte check to the file epoch transferred to its owner."""
    file_digest = _file_hasher or globals()["file_digest"]
    path=Path(path)
    require(not path.is_symlink() and path.is_file(),'written source must be a regular file')
    before=file_fingerprint(path)
    require(file_digest(path)==expected and not path.is_symlink() and
            file_fingerprint(path)==before,'written source changed during final verification')
    if marks is not None:marks[str(path.resolve())]=before
    return expected


def write_buffer(root, values, *, dtype, shape, _verified_marks=None, _file_hasher=None):
    """Publish one finite, explicit raw buffer and return its exact descriptor."""
    file_digest = _file_hasher or globals()["file_digest"]
    try:
        shape = list(shape)
        count = shape_size(shape)
        packed = _buffer_array(values, dtype)
        require(len(packed) == count, 'matrix buffer shape mismatch')
        raw = memoryview(packed).cast('B')
        reference = 'sha256:' + sha256(raw).hexdigest()
        directory = Path(root)/'buffers'; directory.mkdir(parents=True, exist_ok=True)
        path = directory/(reference[7:]+'.bin')
        if path.exists():
            require(not path.is_symlink() and file_digest(path) == reference,
                    'existing matrix buffer changed')
        else:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix='.buffer-', dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(raw)
                # A concurrent identical publisher is harmless, but corrupt/orphaned
                # data is always verified before it can become a descriptor.
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    require(not path.is_symlink() and file_digest(path) == reference,
                            'concurrent matrix buffer changed')
            finally:
                if temporary is not None: temporary.unlink(missing_ok=True)
        require(path.stat().st_size == count*DTYPES[dtype][1],
                'published matrix buffer mismatch')
        _verify_written(path,reference,_verified_marks,_file_hasher=file_digest)
        return {'path': str(path.resolve()), 'file_digest': reference, 'dtype': dtype,
                'shape': shape, 'buffer_digest': reference}
    finally:
        values=packed=raw=None


def write_part(root, value, ref_key, *, _verified_marks=None, _file_hasher=None):
    """Publish a sealed canonical JSON child; byte and content refs stay separate."""
    file_digest = _file_hasher or globals()["file_digest"]
    require(type(value) is dict, 'matrix JSON object required')
    require(value.get(ref_key) == digest({k:v for k,v in value.items() if k != ref_key}),
            'matrix child content identity mismatch')
    directory = Path(root)/'metadata'; directory.mkdir(parents=True, exist_ok=True)
    path = directory/(value[ref_key][7:]+'.json')
    with tempfile.NamedTemporaryFile(prefix='.metadata-', dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write_json(temporary, value)
        expected = file_digest(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(not path.is_symlink() and file_digest(path) == expected,
                    'existing matrix metadata changed')
    finally:
        temporary.unlink(missing_ok=True)
    _verify_written(path,expected,_verified_marks,_file_hasher=file_digest)
    return {'path': str(path.resolve()), 'file_digest': expected, ref_key: value[ref_key]}
