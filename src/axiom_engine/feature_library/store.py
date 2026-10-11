"""Hash once per file epoch and own readonly mmap buffers; Research adds its proofs."""
from copy import deepcopy
import math
from pathlib import Path
from .codec import (digest, file_digest, file_fingerprint, read_json, require, is_ref as _ref)
from .storage import BUFFER_FIELDS

DTYPES = {'float64_le':'<f8','bool_u8':'u1','int64_le':'<i8',
          'uint64_le':'<u8','int32_le':'<i4','uint8':'u1'}

def _fields(value, fields, message):
    require(type(value) is dict and set(value)==set(fields), message)

class VerifiedMatrixStore:
    """Bounded source admission. Never persist a trusted flag across readers."""
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
        self.metrics={'source_bytes':0,'hash_bytes':0,'file_hash_calls':0,
            'unique_descriptor_admissions':0,'mmap_opens':0,'json_decode_calls':0}

    @property
    def np(self):
        if self._np is None:
            import numpy as np
            self._np=np
        return self._np

    def _retain_entry(self, mapping, key, value): mapping[key]=value

    def read_json(self, descriptor, ref_key):
        _fields(descriptor, {'path','file_digest',ref_key}, 'exact matrix parent descriptor required')
        require(_ref(descriptor[ref_key]), 'matrix parent content ref required')
        self._register(descriptor); path=descriptor['path']
        require(self.fingerprints[path][2]<=self.maximum_parent_bytes, 'matrix parent byte budget exceeded')
        if path not in self.json:
            value=read_json(self._resolve(path))
            self.check()
            require(type(value) is dict and value.get(ref_key)==descriptor[ref_key] and
                digest({k:v for k,v in value.items() if k!=ref_key})==descriptor[ref_key],
                'matrix parent content mismatch')
            self.json[path]=value; self.metrics['json_decode_calls']+=1
        return self.json[path]

    def close(self):
        self.check(); require(self.borrowers==0, 'matrix backing still borrowed')
        for array in self.arrays.values():
            if hasattr(array, '_mmap'): array._mmap.close()
        self.arrays.clear(); self.json.clear(); self.closed=True

    def __enter__(self): self.check(); return self
    def __exit__(self,*args): self.close()

    def _register(self, descriptor):
        require(not self.closed, 'matrix store is closed')
        path = descriptor.get('path')
        require(type(path) is str and Path(path).is_absolute() and not self._resolve(path).is_symlink(),
                'fixed absolute matrix file required')
        require(_ref(descriptor.get('file_digest')), 'matrix file digest required')
        if path in self.descriptors:
            old = self.descriptors[path]
            # Raw content-addressed files can appear as a 2-D partition and
            # the corresponding flat Core buffer. Each logical descriptor is
            # still checked below; only physical byte admission is shared.
            require(old == descriptor or (set(old) == BUFFER_FIELDS and
                    set(descriptor) == BUFFER_FIELDS and old['file_digest'] == descriptor['file_digest'] and
                    old['buffer_digest'] == descriptor['buffer_digest']), 'conflicting matrix descriptor')
            require(file_fingerprint(self._resolve(path)) == self.fingerprints[path], 'saved matrix source changed')
            return False
        mark = file_fingerprint(self._resolve(path))
        require(self._hash_once(path) == descriptor['file_digest'], 'matrix file digest mismatch')
        require(file_fingerprint(self._resolve(path)) == mark, 'matrix file changed during hash')
        self._retain_entry(self.descriptors,path,deepcopy(descriptor)); self._retain_entry(self.fingerprints,path,mark)
        return True


    def _hash_once(self,path):
        """Internal hook for the existing Qlib closure validator."""
        path=str(Path(path).absolute()); physical=self._resolve(path)
        require(not physical.is_symlink(),'fixed matrix file required')
        mark=file_fingerprint(physical)
        if path not in self._hashes:
            require(self.metrics['source_bytes']+mark[2]<=self.maximum_source_bytes,'matrix source byte budget exceeded')
            value=file_digest(physical)
            require(file_fingerprint(physical)==mark,'matrix file changed during hash')
            self._retain_entry(self._hashes,path,value); self._retain_entry(self.fingerprints,path,mark)
            self.metrics['source_bytes']+=mark[2]; self.metrics['hash_bytes']+=mark[2]; self.metrics['file_hash_calls']+=1
            self.metrics['unique_descriptor_admissions']+=1
        require(file_fingerprint(physical)==self.fingerprints[path],'saved matrix source changed')
        return self._hashes[path]


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
            if descriptor['dtype'] == 'bool_u8':
                require(bool(((array == 0) | (array == 1)).all()), 'matrix bool bytes must be 0/1')
            self._retain_entry(self.arrays,key,array)
        return self.arrays[key]


    def check(self):
        require(not self.closed, 'matrix store is closed')
        require(all(file_fingerprint(self._resolve(p)) == mark for p, mark in self.fingerprints.items()),
                'saved matrix source changed; initialize a fresh batch')
