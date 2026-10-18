"""Read published immutable manifests and own readonly mmap buffers."""
from copy import deepcopy
from contextlib import contextmanager
import math
from pathlib import Path
import sys
from .codec import (digest, file_digest, file_fingerprint, read_json, require, is_ref as _ref)
from .storage import BUFFER_FIELDS

DTYPES = {'float64_le':'<f8','bool_u8':'u1','int64_le':'<i8',
          'uint64_le':'<u8','int32_le':'<i4','uint8':'u1'}

def _fields(value, fields, message):
    require(type(value) is dict and set(value)==set(fields), message)

class VerifiedMatrixStore:
    """Bounded reads of immutable files published by the preparation owner."""
    def __init__(self, *, maximum_source_bytes=8*1024**3,
                 maximum_matrix_bytes=512*1024**2, maximum_parent_bytes=64*1024**2,
                 _path_resolver=None):
        for value in (maximum_source_bytes, maximum_matrix_bytes, maximum_parent_bytes):
            require(type(value) is int and value>0, 'positive matrix byte budget required')
        self.maximum_source_bytes=maximum_source_bytes
        self.maximum_matrix_bytes=maximum_matrix_bytes
        self.maximum_parent_bytes=maximum_parent_bytes
        self._resolve=_path_resolver or (lambda path:Path(path))
        self.descriptors={}; self.fingerprints={}; self._hashes={}; self.arrays={}; self.json={}
        self.closed=False; self.borrowers=0; self._np=None
        self._operation_depth=0; self._operation_paths=set(); self._invalid=False
        self.metrics={'source_bytes':0,'hash_bytes':0,'file_hash_calls':0,
            'unique_descriptor_admissions':0,'mmap_opens':0,'json_decode_calls':0,
            'source_stat_calls':0,'lifecycle_check_calls':0}

    @property
    def np(self):
        if self._np is None:
            import numpy as np
            self._np=np
        return self._np

    def _retain_entry(self, mapping, key, value): mapping[key]=value

    def read_json(self, descriptor, ref_key):
        self._register(descriptor)
        path=descriptor['path']
        if path not in self.json:
            require(self.fingerprints[path][2]<=self.maximum_parent_bytes, 'matrix parent byte budget exceeded')
            self.json[path]=read_json(self._resolve(path))
            self.metrics['json_decode_calls']+=1
        return self.json[path]

    def close(self):
        require(self.borrowers==0, 'matrix backing still borrowed')
        try:
            if not self._invalid: self.check()
        finally:
            for array in self.arrays.values():
                if hasattr(array, '_mmap'): array._mmap.close()
            self.arrays.clear(); self.json.clear(); self.closed=True

    def __enter__(self): self.check(); return self
    def __exit__(self,*args): self.close()

    def _register(self, descriptor):
        self._available()
        path=descriptor['path']
        if path in self.descriptors:return False
        mark=file_fingerprint(self._resolve(path))
        self.metrics['source_stat_calls']+=1
        require(self.metrics['source_bytes']+mark[2]<=self.maximum_source_bytes, 'matrix source byte budget exceeded')
        self._retain_entry(self.descriptors,path,descriptor)
        self._retain_entry(self.fingerprints,path,mark)
        # These are publisher-supplied references, not newly computed hashes.
        self._retain_entry(self._hashes,path,descriptor['file_digest'])
        self.metrics['source_bytes']+=mark[2]
        self.metrics['unique_descriptor_admissions']+=1
        return True

    def _hash_once(self,path):
        """Explicit producer inspection; ordinary reads never call this hook."""
        self._available();path=str(Path(path).absolute())
        value=file_digest(self._resolve(path))
        size=self._resolve(path).stat().st_size
        self.metrics['hash_bytes']+=size;self.metrics['file_hash_calls']+=1
        return value

    def buffer(self, descriptor):
        _fields(descriptor, BUFFER_FIELDS, 'exact matrix buffer descriptor required')
        require(descriptor['dtype'] in DTYPES and type(descriptor['shape']) is list and
                bool(descriptor['shape']) and all(type(n) is int and n >= 0 for n in descriptor['shape']),
                'matrix buffer dtype/shape required')
        require(descriptor['buffer_digest'] == descriptor['file_digest'], 'raw buffer digest mismatch')
        self._register(descriptor); path = descriptor['path']
        key = path,descriptor['dtype'],tuple(descriptor['shape'])
        if key not in self.arrays:
            dtype = self.np.dtype(DTYPES[descriptor['dtype']])
            size = math.prod(descriptor['shape']) * dtype.itemsize
            require(size == self.fingerprints[path][2], 'matrix buffer byte length mismatch')
            if size:
                array = self.np.memmap(self._resolve(path), dtype=dtype, mode='r', shape=tuple(descriptor['shape']))
                self.metrics['mmap_opens'] += 1
            else:
                array = self.np.empty(descriptor['shape'], dtype=dtype); array.flags.writeable = False
            require(not array.flags.writeable, 'readonly matrix buffer required')
            self._retain_entry(self.arrays,key,array)
            self._check_epoch(path)
        return self.arrays[key]


    def _available(self):
        require(not self.closed, 'matrix store is closed')
        require(not self._invalid, 'saved matrix source changed; initialize a fresh batch')

    def _check_epoch(self, path, mark=None):
        self._available()

    def check_path(self, path):
        self._available()

    def validate_boundary(self):
        self._available()

    @contextmanager
    def operation(self):
        self._available()
        self._operation_depth+=1
        try:yield self
        finally:self._operation_depth-=1

    def check(self):
        self._available()
