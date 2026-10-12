"""Bounded column transport for Engine-owned, fully admitted input snapshots.

This codec changes delivery layout only. It neither creates Native identities
nor admits an arbitrary input. Public factories remain the sole producers.
"""
from array import array
import json
import math
import re
import tempfile
from time import perf_counter

from ..core.contracts import ContractError, fields, integer, require

CODEC = "stock_input_columns_v1"
INPUT_SEMANTICS = "stock_input_projection_semantics_v1"
_ESCAPES = re.compile(r'["\\\x00-\x1f]')


def encoded_size(value):
    """Exact canonical size; repeated strings measured once in bounded slices."""
    strings = {}
    def size(v):
        kind = type(v)
        if kind is str:
            if v not in strings:
                n = 2
                for offset in range(0,len(v),512):
                    part = v[offset:offset+512]
                    n += len(part.encode("utf-8"))
                    n += sum(1 if m[0] in ('"','\\','\b','\t','\n','\f','\r') else 5 for m in _ESCAPES.finditer(part))
                strings[v] = n
            return strings[v]
        if kind in (list,tuple): return 2+max(0,len(v)-1)+sum(map(size,v))
        if kind is dict:
            require(all(type(k) is str for k in v), "Input object keys must be strings")
            return 2+max(0,len(v)-1)+sum(size(k)+1+size(x) for k,x in v.items())
        if v is None: return 4
        if kind is bool: return 4 if v else 5
        if kind is int: return len(str(v))
        require(kind is float and math.isfinite(v), "Input contains unsupported/nonfinite value")
        return len(repr(v))
    return size(value)


def encoding_bound(value):
    """Conservative UTF-8/escaping bound without serializing every scalar."""
    kind = type(value)
    if kind is str: return 2 + 6*len(value)
    if kind in (list, tuple): return 2 + max(0, len(value)-1) + sum(map(encoding_bound, value))
    if kind is dict:
        require(all(type(k) is str for k in value), "Input object keys must be strings")
        return 2 + max(0, len(value)-1) + sum(encoding_bound(k)+1+encoding_bound(v) for k,v in value.items())
    if value is None: return 4
    if kind is bool: return 5
    if kind is int: return len(str(value))
    require(kind is float and math.isfinite(value), "Input contains unsupported/nonfinite value")
    return len(repr(value))


def _cells(value):
    if type(value) is dict: return 1+len(value)+sum(_cells(v) for v in value.values())
    if type(value) in (list, tuple): return 1+sum(_cells(v) for v in value)
    return 1


def pack(value):
    strings, ids = [], {}
    def string(s):
        if s not in ids: ids[s] = len(strings); strings.append(s)
        return ids[s]
    def node(v):
        if type(v) is str: return ["s", string(v)]
        if type(v) is dict:
            names = sorted(v)
            return ["m", [string(n) for n in names], [node(v[n]) for n in names]]
        if type(v) in (list, tuple):
            if v and all(type(r) is dict for r in v):
                names = sorted({k for r in v for k in r})
                columns, missing = [], []
                for name in names:
                    columns.append(node([r.get(name) for r in v]))
                    missing.append([i for i,r in enumerate(v) if name not in r])
                return ["c", len(v), [string(n) for n in names], columns, missing]
            if all(x is None or type(x) is str for x in v):
                return ["q", [None if x is None else string(x) for x in v]]
            if all(x is None or type(x) in (bool, int, float) for x in v): return ["a", list(v)]
            if v and all(type(x) in (list, tuple) and all(type(s) is str for s in x) for x in v):
                unique, indexes, found = [], [], {}
                for x in v:
                    key = tuple(x)
                    if key not in found: found[key] = len(unique); unique.append(node(x))
                    indexes.append(found[key])
                return ["d", unique, indexes]
            return ["l", [node(x) for x in v]]
        require(v is None or type(v) in (bool,int,float), "Unsupported input transport value")
        if type(v) is float: require(math.isfinite(v), "Nonfinite input transport value")
        return v
    root = node(value)
    return {"codec": CODEC, "strings": strings, "root": root}


def unpack(value):
    fields(value, "codec strings root")
    require(value["codec"] == CODEC and type(value["strings"]) is list and
            all(type(s) is str for s in value["strings"]), "Unsupported input column codec")
    strings = value["strings"]
    def string(i):
        require(type(i) is int and 0 <= i < len(strings), "Input string index outside dictionary")
        return strings[i]
    def node(v):
        if type(v) is not list:
            require(v is None or type(v) in (bool,int,float), "Malformed input column scalar")
            if type(v) is float: require(math.isfinite(v), "Nonfinite input column scalar")
            return v
        require(v and type(v[0]) is str, "Malformed input column node")
        tag = v[0]
        if tag == "s":
            require(len(v) == 2, "Malformed input string node"); return string(v[1])
        if tag == "m":
            require(len(v) == 3 and type(v[1]) is list and type(v[2]) is list and len(v[1]) == len(v[2]), "Malformed input object columns")
            names = [string(i) for i in v[1]]
            require(len(set(names)) == len(names), "Duplicate input object column")
            return {k:node(x) for k,x in zip(names,v[2])}
        if tag == "c":
            require(len(v) == 5 and type(v[1]) is int and v[1] >= 0 and
                    all(type(x) is list for x in v[2:]) and len(v[2]) == len(v[3]) == len(v[4]), "Malformed input table")
            names = [string(i) for i in v[2]]
            require(len(set(names)) == len(names), "Duplicate input table column")
            columns = [node(x) for x in v[3]]
            require(all(type(c) is list and len(c) == v[1] for c in columns), "Input table column length mismatch")
            for absent in v[4]:
                require(type(absent) is list and all(type(i) is int and 0 <= i < v[1] for i in absent) and
                        absent == sorted(set(absent)), "Input table missing mask mismatch")
            rows = [dict(zip(names, cells)) for cells in zip(*columns)] if names else [{} for _ in range(v[1])]
            for name, absent in zip(names,v[4]):
                for i in absent: rows[i].pop(name)
            return rows
        if tag in ("q", "a", "l"):
            require(len(v) == 2 and type(v[1]) is list, "Malformed input array")
            if tag == "q": return [None if i is None else string(i) for i in v[1]]
            if tag == "a":
                require(all(x is None or type(x) in (bool,int,float) for x in v[1]), "Malformed numeric column")
                require(all(type(x) is not float or math.isfinite(x) for x in v[1]), "Nonfinite numeric column")
                return v[1]
            return [node(x) for x in v[1]]
        require(tag == "d" and len(v) == 3 and type(v[1]) is list and type(v[2]) is list, "Malformed dictionary column")
        values = [node(x) for x in v[1]]
        require(all(type(x) is list and all(type(s) is str for s in x) for x in values), "Malformed dictionary values")
        require(all(type(i) is int and 0 <= i < len(values) for i in v[2]), "Input dictionary index outside table")
        # Independent row containers preserve the existing mutable-copy contract.
        return [list(values[i]) for i in v[2]]
    return node(value["root"])


def measured_size(value, maximum):
    """Validate columns and measure expansion before allocating original rows."""
    fields(value,"codec strings root")
    require(value["codec"] == CODEC and type(value["strings"]) is list and
            all(type(s) is str for s in value["strings"]), "Unsupported input column codec")
    sizes = [encoded_size(s) for s in value["strings"]]
    def text(i):
        require(type(i) is int and 0 <= i < len(sizes), "Input string index outside dictionary")
        return sizes[i]
    def total(n):
        require(n <= maximum, "Saved input expansion exceeds decoded budget")
        return n
    def array_sizes(v, depth):
        require(depth <= 128 and type(v) is list and v and type(v[0]) is str, "Malformed/deep input column")
        tag = v[0]
        if tag in ("q","a","l"):
            require(len(v) == 2 and type(v[1]) is list, "Malformed input column array")
            if tag == "q": return [4 if i is None else text(i) for i in v[1]]
            if tag == "a":
                require(all(x is None or type(x) in (bool,int,float) for x in v[1]), "Malformed numeric column")
                return [encoded_size(x) for x in v[1]]
            return [size(x,depth+1) for x in v[1]]
        if tag == "d":
            require(len(v) == 3 and type(v[1]) is list and type(v[2]) is list, "Malformed dictionary column")
            require(all(type(x) is list and len(x) == 2 and x[0] == "q" and type(x[1]) is list and
                        all(type(i) is int for i in x[1]) for x in v[1]), "Malformed dictionary values")
            unique = [size(x,depth+1) for x in v[1]]
            require(all(type(i) is int and 0 <= i < len(unique) for i in v[2]), "Input dictionary index outside table")
            return [unique[i] for i in v[2]]
        require(tag == "c" and len(v) == 5 and type(v[1]) is int and v[1] >= 0 and
                all(type(x) is list for x in v[2:]) and len(v[2]) == len(v[3]) == len(v[4]), "Malformed input table")
        # A row is at least two bytes; reject before the expanded size vector.
        require(2*v[1] <= maximum, "Saved input row count exceeds decoded budget")
        require(len(set(v[2])) == len(v[2]), "Duplicate input table column")
        rows = [2]*v[1]; present = [0]*v[1]
        for key,column,missing in zip(v[2],v[3],v[4]):
            key_size = text(key); cells = array_sizes(column,depth+1)
            require(len(cells) == v[1] and type(missing) is list and
                    all(type(i) is int and 0 <= i < v[1] for i in missing) and missing == sorted(set(missing)),
                    "Input table coverage/missing mask mismatch")
            absent = set(missing)
            for i,cell in enumerate(cells):
                if i not in absent: rows[i] += key_size+1+cell; present[i] += 1
        return [total(n+max(0,p-1)) for n,p in zip(rows,present)]
    def size(v, depth):
        require(depth <= 128, "Saved input column nesting exceeded")
        if type(v) is not list: return encoded_size(v)
        require(v and type(v[0]) is str, "Malformed input column node")
        if v[0] == "s":
            require(len(v) == 2, "Malformed input string node"); return text(v[1])
        if v[0] == "m":
            require(len(v) == 3 and type(v[1]) is list and type(v[2]) is list and
                    len(v[1]) == len(v[2]) and len(set(v[1])) == len(v[1]), "Malformed input object columns")
            return total(2+max(0,len(v[1])-1)+sum(text(k)+1+size(x,depth+1) for k,x in zip(v[1],v[2])))
        values = array_sizes(v,depth+1)
        return total(2+max(0,len(values)-1)+sum(values))
    return total(size(value["root"],0))


class InputStore:
    """One bounded record partition per original owner window, with offsets."""
    def __init__(self, maximum, fixed, label):
        integer(maximum,1); require(fixed <= maximum, label+" exceeds quota before allocation")
        self.maximum, self.fixed, self.label = maximum, fixed, label
        self.offsets = array("Q"); self.file = tempfile.TemporaryFile(mode="w+b")
        self.bytes = 0; self.write_seconds = 0; self.expanded = {}; self.lengths = {}; self.records = []

    @property
    def used(self): return self.fixed+len(self.offsets)*8+self.bytes

    def charge(self, size):
        require(self.used+size <= self.maximum, self.label+" exceeds quota before index growth")
        self.fixed += size

    def write(self, value, memory):
        started = perf_counter(); require(self.used+16 <= self.maximum, self.label+" exceeds quota before index growth")
        self.charge(128)
        memory.replace_global(("private_offsets",id(self)),(len(self.offsets)+2)*8)
        memory.replace_global(("private_record_index",id(self)),(len(self.records)+1)*128)
        expanded = encoded_size(value)
        start = self.file.seek(0,2)
        with memory.stage() as scratch:
            scratch.reserve(16*_cells(value)+1024)
            encoded = pack(value)
            bound = encoding_bound(encoded)
            # C encoder is selected only after reserving its full worst-case
            # output and UTF-8 buffers; small budgets use bounded iterencode.
            encoder = json.JSONEncoder(ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(",",":"))
            if 3*bound <= memory.available:
                scratch.reserve(3*bound)
                text = encoder.encode(encoded); raw = text.encode("utf-8")
                require(self.used+16+len(raw) <= self.maximum, self.label+" exceeds quota before write")
                self.file.write(raw); text = raw = None
            else:
                scratch.reserve(4096)
                for text in encoder.iterencode(encoded):
                    # String output is bounded by the admitted column graph;
                    # reserve before UTF-8 allocation, even on the slow path.
                    charge = 4*len(text)+64; scratch.reserve(charge)
                    raw = text.encode("utf-8")
                    require(self.fixed+len(self.offsets)*8+16+self.file.tell()+len(raw) <= self.maximum,
                            self.label+" exceeds quota before write")
                    self.file.write(raw); raw = text = None; scratch.release(charge)
            encoded = None
        length = self.file.tell()-start
        self.offsets.extend((start,length)); self.expanded[start] = expanded
        self.lengths[start] = length
        self.records.append((start,length,expanded)); self.bytes = self.file.tell()
        self.write_seconds += perf_counter()-started
        return len(self.offsets)//2-1

    def write_metadata(self, reference, header, memory):
        number = self.write({"contract_version":"stock_signal_metadata_v1","signal_ref":reference,"header":header},memory)
        start,length = self.offsets[2*number:2*number+2]; del self.offsets[2*number:2*number+2]
        self.charge(16+len(reference)); memory.replace_global(("private_offsets",id(self)),len(self.offsets)*8)
        return {"signal_ref":reference,"offset":start,"length":length}

    def _read_record(self, number, memory, budget):
        start,length = self.offsets[2*number:2*number+2]
        return self._read_span(start,length,memory,budget)

    def _read_span(self, start, length, memory, budget):
        require(self.lengths.get(start) == length and start in self.expanded,
                "Input location outside original snapshot")
        stage = memory.stage()
        try:
            stage.reserve(3*self.expanded[start]+3*length)
            self.file.seek(start); raw = bytearray(); remaining = length
            while remaining:
                chunk = self.file.read(min(remaining,budget["max_read_bytes"],65536))
                require(bool(chunk), "Private stock input store truncated")
                raw.extend(chunk); remaining -= len(chunk)
            packed = json.loads(raw); raw = chunk = None
            value = unpack(packed); packed = None
            stage.release(3*length+2*self.expanded[start])
            return value,stage
        except BaseException: stage.close(); raise

    def read(self, number, memory, budget): return self._read_record(number,memory,budget)

    def close(self):
        self.file.close(); self.offsets = array("Q"); self.expanded.clear(); self.lengths.clear(); self.records.clear()
