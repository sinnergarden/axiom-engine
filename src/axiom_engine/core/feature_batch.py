"""Bounded numeric reuse of the original Feature path, with independent views."""
import struct
import sys
import time

from .contracts import canonical, require
from .execution import _execute_feature_plan

_POINTER_BYTES = struct.calcsize('P')
_MINIMUM_REUSE_VALUES = 64


def _tree_bytes(value):
    """Conservative retained graph charge, including borrowed immutable keys."""
    return sys.getsizeof(value) + (sum(_tree_bytes(v) for v in value) if type(value) is tuple else 0)


class _BatchProfile:
    def __init__(self):
        self.semantic = self.universe = self.previous_day = None

    def observe(self, p, c, keys, outputs):
        require(not p['event_schema'], 'FEATURE_BATCH_EVENTS_UNSUPPORTED')
        require(p['observation_domain'] == 'sessions', 'FEATURE_BATCH_SESSION_DOMAIN')
        days = {k[1] for k in outputs}
        require(len(days) == 1, 'FEATURE_BATCH_ONE_OUTPUT_SESSION')
        day = next(iter(days))
        universe = tuple(sorted({k[0] for k in keys}))
        require(set(keys) == {(s,d) for s in universe for d in c['sessions']}
                and set(outputs) == {(s,day) for s in universe}, 'FEATURE_BATCH_COMPLETE_GRID')
        semantic = canonical({k:p[k] for k in ('abi','semantics','recipe_ref','input_schema',
            'event_schema','observation_domain','history_policy','nodes','outputs','obligations')})
        if self.semantic is not None:
            require(semantic == self.semantic and universe == self.universe, 'FEATURE_BATCH_SEMANTIC_UNIVERSE_MISMATCH')
            require(day > self.previous_day, 'FEATURE_BATCH_OUTPUT_SESSION_ORDER')
        self.semantic, self.universe, self.previous_day = semantic, universe, day


class _NumericReuse:
    """Call-local exact numeric IDs and results; never cells or view provenance.

    The charge includes the mapping, keys/results, and reserved construction /
    insertion workspace. Counting immutable borrowed keys again is conservative.
    This is a cache-workspace bound, not a process RSS or input/output bound.
    """
    def __init__(self, budget):
        self.budget = budget
        enabled = budget >= 512 and sys.implementation.name == 'cpython'
        self.entries, self.intern, self.current = ({},{},{}) if enabled else (None,None,None)
        self.entry_bytes = self.intern_bytes = self.current_bytes = self.next_id = 0
        self.generation = 0
        self.key_bound = 0
        self.disabled = set()
        self.previous = {}
        self.stats = dict(reuse_budget_bytes=budget, views=0, helpers={}, key_attempts=0,
                          keys_built=0, key_charge_bytes=0, key_build_ns=0, numeric_compute_ns=0,
                          reuse_overhead_ns=0, peak_reuse_bytes=self._retained(), retained_reuse_bytes=0,
                          budget_fallbacks=0, evictions=0, numeric_ids_built=0, disabled_ops={})

    def _retained(self):
        if self.entries is None: return 0
        return self.entry_bytes+self.intern_bytes+self.current_bytes+self._tables()

    def _tables(self):
        return sum(sys.getsizeof(v) for v in (self.entries,self.intern,self.current))

    def begin_view(self, universe):
        if self.entries is not None:
            self.key_bound = sys.getsizeof((None,None))+max(sys.getsizeof(s) for s in universe)+sys.getsizeof('0000-00-00')

    def _discard(self):
        self.stats['evictions'] += len(self.entries)
        self.entries.clear(); self.intern.clear(); self.current.clear()
        self.entry_bytes = self.intern_bytes = self.current_bytes = 0
        self.generation += 1

    def _compute(self, count, function, args):
        count['computed'] += 1
        start = time.perf_counter_ns()
        try:
            return function(*args)
        finally:
            elapsed = time.perf_counter_ns()-start
            self.stats['numeric_compute_ns'] += elapsed
            count['numeric_compute_ns'] += elapsed

    def _reserve(self, required):
        # Growth workspace includes previous tables; required additionally
        # reserves 512 bytes for each potential new mapping entry.
        size = self._retained()+required+2*self._tables()+512
        if size > self.budget: self._discard()
        size = self._retained()+required+2*self._tables()+512
        if size > self.budget: return False
        self.stats['peak_reuse_bytes'] = max(self.stats['peak_reuse_bytes'], size)
        return True

    def _construction_bound(self, scope, dependency_scope, source_name, refs):
        misses = sum((0,source_name,k) not in self.current for k in dependency_scope)
        ref_misses = sum((1,None,k) not in self.current for k in scope) if refs is not None else 0
        # New ID keys/current lookup keys contain the original immutable key,
        # exact bits/validity or member/industry. Charge borrowed strings again.
        industry = max((sys.getsizeof(refs[k]['industry']) for k in scope),default=0) if refs is not None else 0
        leaf_bound = 512+2*self.key_bound+2*sys.getsizeof(source_name)+industry
        count = len(dependency_scope)+(len(scope) if refs is not None else 0)
        return (misses+ref_misses)*(leaf_bound+1024)+count*(2*_POINTER_BYTES+sys.getsizeof(0))+1024

    def _id(self, tag, source_name, key, cell=None, reference=None):
        lookup = tag,source_name,key
        code = self.current.get(lookup)
        if code is not None: return code
        leaf = ((tag,source_name,key,struct.pack('<d',cell.value) if cell.value is not None else None,bool(cell.issues))
                if tag == 0 else (tag,source_name,key,reference['member'],reference['industry']))
        code = self.intern.get(leaf)
        if code is None:
            code = self.next_id; self.next_id += 1
            self.intern[leaf] = code
            self.intern_bytes += _tree_bytes(leaf)+sys.getsizeof(code)
            self.stats['numeric_ids_built'] += 1
        self.current[lookup] = code
        self.current_bytes += _tree_bytes(lookup)+sys.getsizeof(code)
        return code

    def _key_charge(self, key):
        kind,node_index,values,domain = key
        # IDs are immutable integers bounded by next_id. Do not walk every
        # window again merely to charge its already-interned scalar backing.
        code_bytes = sys.getsizeof(self.next_id)
        return (sys.getsizeof(key)+sys.getsizeof(kind)+sys.getsizeof(node_index)+
                sys.getsizeof(values)+sys.getsizeof(domain)+(len(values)+len(domain))*code_bytes)

    def evaluate(self, kind, node_index, scope, cells, function, args, *, refs=None,
                 source_name=None, dependency_scope=None):
        count = self.stats['helpers'].setdefault(kind, dict(requested=0,computed=0,reused=0,disabled=0,
                                                           key_attempts=0,numeric_compute_ns=0,reuse_overhead_ns=0))
        start, before = time.perf_counter_ns(), count['numeric_compute_ns']
        try:
            return self._evaluate(kind,node_index,scope,cells,function,args,refs,count,
                                  source_name,scope if dependency_scope is None else dependency_scope)
        finally:
            overhead = time.perf_counter_ns()-start-(count['numeric_compute_ns']-before)
            count['reuse_overhead_ns'] += overhead
            self.stats['reuse_overhead_ns'] += overhead

    def _evaluate(self, kind, node_index, scope, cells, function, args, refs, count, source_name, dependency_scope):
        count['requested'] += 1
        # Cheap operators and short reductions deliberately do not build keys.
        if self.entries is None or kind in self.disabled or len(args[0]) < _MINIMUM_REUSE_VALUES:
            count['disabled'] += 1
            return self._compute(count, function, args)
        key = values = domain = None
        self.stats['key_attempts'] += 1
        count['key_attempts'] += 1
        start = time.perf_counter_ns()
        try:
            # Make room for even the temporary lookup tuples used by sizing.
            if not self._reserve(256):
                self.stats['budget_fallbacks'] += 1
            else:
                bound = self._construction_bound(scope,dependency_scope,source_name,refs)
                generation = self.generation
                room = self._reserve(bound)
                if room and self.generation != generation:
                    # Clearing the tables invalidates the sizing pass's hit
                    # counts. Reserve again for the now-cold complete keys.
                    room = self._reserve(self._construction_bound(scope,dependency_scope,source_name,refs))
                if not room:
                    self.stats['budget_fallbacks'] += 1
                else:
                    values = tuple(self._id(0,source_name,k,cell=c) for k,c in zip(dependency_scope,cells))
                    domain = tuple(self._id(1,None,k,reference=refs[k]) for k in scope) if refs is not None else ()
                    key = kind,node_index,values,domain
                    self.stats['keys_built'] += 1
                    self.stats['key_charge_bytes'] += self._key_charge(key)
        finally:
            self.stats['key_build_ns'] += time.perf_counter_ns()-start
        if key is None: return self._compute(count, function, args)
        try:
            saved = self.entries.get(key)
            if saved is not None:
                count['reused'] += 1
                return saved[0]
            result = self._compute(count, function, args)
            # Only scalar/tuple numeric helper outputs may enter this mapping.
            require(type(result) in (float,int,bool,type(None)) or
                    type(result) is tuple and all(type(v) in (float,int,bool,type(None)) for v in result),
                    'INTERNAL_NONNUMERIC_REUSE_RESULT')
            charge = self._key_charge(key)+_tree_bytes(result)+sys.getsizeof((None,0))+sys.getsizeof(0)
            self.entries[key] = result, charge
            self.entry_bytes += charge
            require(self._retained() <= self.budget, 'INTERNAL_REUSE_BUDGET')
            return result
        finally:
            # A raised arithmetic error must not retain the temporary key in
            # its traceback after the call's cache has been cleared.
            key = saved = values = domain = None

    def finish_view(self):
        self.stats['views'] += 1
        # The first view fills the cache. On subsequent views only continue an
        # op when its measured overhead is below the estimated saved helper work.
        # This affects diagnostics/performance only; original math is unchanged.
        for kind, count in self.stats['helpers'].items():
            prior = self.previous.get(kind)
            if (prior is not None and count['key_attempts'] > prior[2]
                    and kind not in self.disabled and self.entries is not None):
                hits = count['reused']-prior[0]
                overhead = count['reuse_overhead_ns']-prior[1]
                saved = hits*count['numeric_compute_ns']/count['computed'] if count['computed'] else 0
                if not hits or saved <= overhead:
                    self.disabled.add(kind)
                    self.stats['disabled_ops'][kind] = 'no_hits' if not hits else 'overhead_exceeds_estimated_saving'
            self.previous[kind] = count['reused'],count['reuse_overhead_ns'],count['key_attempts']
        if self.current is not None:
            self.current.clear(); self.current_bytes = 0

    def clear(self):
        if self.entries is not None:
            self.entries.clear(); self.intern.clear(); self.current.clear()
        self.entries = self.intern = self.current = None
        self.entry_bytes = self.intern_bytes = self.current_bytes = 0
        self.stats['retained_reuse_bytes'] = 0


def execute_feature_plan_batch(requests: tuple, *, reuse_budget_bytes: int) -> dict:
    """Execute a fixed tuple of original one-day views through the sole path.

    Requests contain (FeaturePlan, FactBatch, ExecutionContext), in strictly
    increasing output-day order, with the same semantic DAG and complete
    universe. Original source/clock/visibility inputs are independently admitted.
    Budget is a strict nonnegative int; zero disables all reuse/key construction.
    Frames remain ordinary original-identity FeatureFrames. Stats are performance
    diagnostics, not business evidence. Caller bounds its inputs and returned
    Frames independently; this cache budget does not bound their combined RSS.
    """
    start = time.perf_counter_ns()
    require(type(requests) is tuple and bool(requests), 'FEATURE_BATCH_FIXED_NONEMPTY_REQUESTS')
    require(type(reuse_budget_bytes) is int and reuse_budget_bytes >= 0, 'FEATURE_BATCH_REUSE_BUDGET')
    for request in requests:
        require(type(request) is tuple and len(request) == 3, 'FEATURE_BATCH_REQUEST_TRIPLE')
    reuse, profile, frames = _NumericReuse(reuse_budget_bytes), _BatchProfile(), []
    try:
        for request in requests:
            frames.append(_execute_feature_plan(*request, reuse=reuse, profile=profile))
            reuse.finish_view()
    finally:
        reuse.clear()
        reuse.stats['total_ns'] = time.perf_counter_ns()-start
    return dict(frames=tuple(frames), stats=reuse.stats)
