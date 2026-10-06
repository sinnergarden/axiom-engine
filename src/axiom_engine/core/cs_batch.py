"""The opt-in, neutral, one-column packed CS profile. No I/O or learned state."""
import hashlib
import json
import struct
import sys

from .._implementation import IMPLEMENTATION_REF
from .contracts import (ABI, SEMANTICS, ContractError, canonical, check_value,
                        digest, fields, require, schema, session, text)
from .execution import _Cell, _cs_zscore_scale, _cs_zscore_value, _merge

_INPUT_FIELDS = ('contract_version calendar_ref schema output_schema sessions security_ids '
                 'values value_validity value_reason_codes reason_dictionary fact_available_at_utc_us '
                 'reference_member reference_available_at_utc_us selection_cutoff_utc_us '
                 'source_bindings_by_session fact_source_codes reference_source_codes')
_PARAMS = dict(group='session', unknown_group='reject', missing='skip', ddof=0,
               epsilon=1e-12, constant='missing', clip=None, excluded='missing')
_BUFFER_TYPES = {
    'values': ('float64', 'd', 8), 'value_validity': ('bool', '?', 1),
    'value_reason_codes': ('int32', 'i', 4),
    'fact_available_at_utc_us': ('int64', 'q', 8), 'reference_member': ('bool', '?', 1),
    'reference_available_at_utc_us': ('int64', 'q', 8), 'selection_cutoff_utc_us': ('int64', 'q', 8),
    'fact_source_codes': ('int32', 'i', 4), 'reference_source_codes': ('int32', 'i', 4),
}
_SOURCE_FIELDS = 'id data_ref view_ref revision_policy qualification availability_basis'


def _ref(value):
    return 'sha256:' + hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _descriptor(raw, dtype, count):
    return dict(dtype=dtype, shape=[count], bytes_digest='sha256:'+hashlib.sha256(raw).hexdigest())


def _snapshot_buffer(value, name, count):
    dtype, code, size = _BUFFER_TYPES[name]
    try:
        view = memoryview(value)
    except (TypeError, ValueError) as error:
        raise ContractError('BATCH_BUFFER: '+name) from error
    require(view.readonly and view.ndim == 1 and view.c_contiguous,
            'BATCH_BUFFER_READONLY_1D_CONTIGUOUS: '+name)
    require(view.shape == (count,) and view.itemsize == size and view.nbytes == count*size,
            'BATCH_BUFFER_SHAPE: '+name)
    fmt = view.format
    require(not fmt.startswith(('>', '!')), 'BATCH_BUFFER_ENDIAN: '+name)
    fmt = fmt[1:] if fmt[:1] in ('<', '@', '=') else fmt
    formats = {'d'} if code == 'd' else {'?'} if code == '?' else {'i', 'l'} if code == 'i' else {'q', 'l'}
    require(fmt in formats, 'BATCH_BUFFER_DTYPE: '+name)
    # Capture owned bytes once. Readonly views may still have writable aliases;
    # calculation and identity both use this same immutable snapshot.
    raw = view.tobytes()
    if code == '?': require(all(v in (0, 1) for v in raw), 'BATCH_BOOL_DOMAIN: '+name)
    return memoryview(raw).cast(code), _descriptor(raw, dtype, count)


def _sources(value, days):
    require(type(value) is dict and set(value) == set(days), 'BATCH_SOURCE_SESSIONS')
    for day in days:
        item = value[day]
        fields(item, 'bindings source_sets')
        bindings, sets = item['bindings'], item['source_sets']
        require(type(bindings) is list and bool(bindings), 'BATCH_SOURCE_BINDINGS')
        ids = []
        for binding in bindings:
            fields(binding, _SOURCE_FIELDS)
            for name in ('id', 'revision_policy', 'availability_basis'): text(binding[name])
            digest(binding['data_ref']); digest(binding['view_ref'])
            require(binding['qualification'] in ('verified', 'observed', 'best_effort', 'synthetic'),
                    'Unknown source qualification')
            ids.append(binding['id'])
        require(ids == sorted(set(ids)), 'BATCH_SOURCE_BINDING_ORDER')
        require(type(sets) is list and bool(sets), 'BATCH_SOURCE_SETS')
        tuples = []
        for group in sets:
            require(type(group) is list and bool(group), 'BATCH_SOURCE_SET')
            for source in group: text(source)
            require(group == sorted(set(group)) and set(group) <= set(ids), 'BATCH_SOURCE_SET_IDS')
            tuples.append(tuple(group))
        require(tuples == sorted(set(tuples)), 'BATCH_SOURCE_SET_ORDER')


def _ceil_second(value):
    projected = -((-value)//1_000_000)*1_000_000
    require(-(2**63) <= projected < 2**63, 'CLOCK_PROJECTION_OVERFLOW')
    return projected


def _owned_buffer(values, dtype):
    codes = {'float64': 'd', 'bool': '?', 'int32': 'i', 'int64': 'q'}
    code = codes[dtype]
    raw = struct.pack('<'+str(len(values))+code, *values)
    return memoryview(raw).cast(code), _descriptor(raw, dtype, len(values))


def execute_cs_zscore_batch(batch: dict, *, params: dict,
                            clock_projection: str = 'ceil_to_core_second_v1') -> dict:
    """Validate a full D×U one-column grid and execute the frozen CS profile.

    Buffer dtype/shape/order follow core_cs_zscore_batch_input_v1. No carrier
    field, cached result, caller identity or admission bypass is trusted.
    Returned buffers are owned immutable snapshots; metadata binds native input
    clocks, source sets, exact output bytes and the actual arithmetic backend.
    """
    require(sys.byteorder == 'little', 'BATCH_BUFFER_ENDIAN: little-endian host required')
    fields(batch, _INPUT_FIELDS)
    fields(params, 'group unknown_group missing ddof epsilon constant clip excluded')
    require(canonical(params) == canonical(_PARAMS), 'UNSUPPORTED_CS_BATCH_PROFILE')
    require(type(clock_projection) is str and clock_projection == 'ceil_to_core_second_v1',
            'UNSUPPORTED_CLOCK_PROJECTION')
    # Freeze small metadata independently of caller dictionaries/lists.
    meta = json.loads(canonical({k: v for k, v in batch.items() if k not in _BUFFER_TYPES}))
    require(meta['contract_version'] == 'core_cs_zscore_batch_input_v1', 'BATCH_CONTRACT_VERSION')
    digest(meta['calendar_ref'])
    for field in ('schema', 'output_schema'):
        schema(meta[field])
        require(len(meta[field]) == 1 and meta[field][0]['dtype'] == 'float64', 'BATCH_SINGLE_FLOAT64_COLUMN')
    column, output_column = meta['schema'][0], meta['output_schema'][0]
    require(column['stage'] == 'fact', 'Input stage must be fact')
    require(output_column['unit'] == 'dimensionless' and output_column['stage'] == 'cross_sectional'
            and output_column['missing'] == 'preserve', 'BATCH_OUTPUT_SCHEMA')
    days, securities = meta['sessions'], meta['security_ids']
    require(type(days) is list and bool(days), 'BATCH_SESSIONS')
    for day in days: session(day)
    require(days == sorted(set(days)), 'BATCH_SESSION_ORDER')
    require(type(securities) is list and bool(securities), 'BATCH_SECURITY_IDS')
    for security in securities: text(security)
    require(securities == sorted(set(securities)), 'BATCH_SECURITY_ORDER')
    reasons = meta['reason_dictionary']
    require(type(reasons) is list and bool(reasons) and reasons[0] is None, 'BATCH_REASON_DICTIONARY')
    for reason in reasons[1:]: text(reason)
    require(reasons[1:] == sorted(set(reasons[1:])), 'BATCH_REASON_ORDER')
    _sources(meta['source_bindings_by_session'], days)
    width, count = len(securities), len(securities)*len(days)
    buffers, descriptions = {}, {}
    for name in _BUFFER_TYPES:
        buffers[name], descriptions[name] = _snapshot_buffer(batch[name], name, len(days) if name == 'selection_cutoff_utc_us' else count)
    # Validate native microseconds before any ceil. An unavailable fact cannot
    # be admitted just because it and the cutoff round to the same second.
    for day_index, day in enumerate(days):
        cutoff = buffers['selection_cutoff_utc_us'][day_index]
        source_sets = meta['source_bindings_by_session'][day]['source_sets']
        for index in range(day_index*width, (day_index+1)*width):
            valid = buffers['value_validity'][index]
            code = buffers['value_reason_codes'][index]
            require(0 <= code < len(reasons) and (code == 0) == valid, 'BATCH_REASON_CODE')
            value = buffers['values'][index]
            if not valid:
                require(struct.pack('<d', value) == b'\0'*8, 'BATCH_NULL_CANONICAL_ZERO')
            check_value(value if valid else None, column)
            for name in ('fact_source_codes', 'reference_source_codes'):
                source_code = buffers[name][index]
                require(0 <= source_code < len(source_sets), 'BATCH_SOURCE_CODE: '+name)
            require(len(source_sets[buffers['reference_source_codes'][index]]) == 1, 'BATCH_REFERENCE_SOURCE_SINGLETON')
            require(buffers['fact_available_at_utc_us'][index] <= cutoff
                    and buffers['reference_available_at_utc_us'][index] <= cutoff,
                    'UNAVAILABLE_AT_NATIVE_CUTOFF')
    cutoffs = [_ceil_second(v) for v in buffers['selection_cutoff_utc_us']]
    out_values, out_flags, out_reasons, out_clocks, out_source_codes = [], [], [], [], []
    output_sources = {}
    for day_index, day in enumerate(days):
        source_info = meta['source_bindings_by_session'][day]
        source_sets = source_info['source_sets']
        indices = range(day_index*width, (day_index+1)*width)
        facts, references, members = [], [], []
        for index in indices:
            facts.append(_Cell(buffers['values'][index] if buffers['value_validity'][index] else None,
                               _ceil_second(buffers['fact_available_at_utc_us'][index]),
                               tuple(source_sets[buffers['fact_source_codes'][index]])))
            references.append(_Cell(None, _ceil_second(buffers['reference_available_at_utc_us'][index]),
                                    tuple(source_sets[buffers['reference_source_codes'][index]])))
            members.append(buffers['reference_member'][index])
        deps = [cell for cell, member in zip(facts, members) if member]
        vals = [cell.value for cell in deps if cell.value is not None]
        std, undefined = _cs_zscore_scale(vals, _PARAMS)
        dependency = _merge(None, deps + references)
        mean, day_sources = None, []
        for cell, member in zip(facts, members):
            value = None
            if member and (undefined or vals and cell.value is not None):
                value, mean = _cs_zscore_value(cell.value, vals, std, undefined, _PARAMS, mean)
            merged = _merge(value, [dependency, cell], 'REFERENCE_MISSING')
            require(merged.available <= cutoffs[day_index], 'FUTURE_DEPENDENCY')
            present = merged.value is not None
            out_values.append(merged.value if present else 0.0)
            out_flags.append(present); out_reasons.append(0 if present else 1)
            out_clocks.append(merged.available); day_sources.append(merged.sources)
        unique_sources = sorted(set(day_sources))
        source_codes = {source: i for i, source in enumerate(unique_sources)}
        out_source_codes.extend(source_codes[source] for source in day_sources)
        output_sources[day] = dict(bindings=source_info['bindings'], source_sets=[list(s) for s in unique_sources])
    result = dict(contract_version='core_cs_zscore_batch_result_v1', sessions=days,
                  security_ids=securities, schema=meta['output_schema'],
                  reason_dictionary=[None, 'REFERENCE_MISSING'], source_bindings_by_session=output_sources)
    output_descriptions = {}
    for name, values, dtype in [('values',out_values,'float64'), ('value_validity',out_flags,'bool'),
                                ('value_reason_codes',out_reasons,'int32'), ('available_at_utc_us',out_clocks,'int64'),
                                ('source_codes',out_source_codes,'int32')]:
        result[name], output_descriptions[name] = _owned_buffer(values, dtype)
    input_document = dict(meta, **descriptions)
    keys_ref = _ref(dict(sessions=days, security_ids=securities, order='session_security'))
    spec_ref = _ref(dict(abi=ABI, semantics=SEMANTICS, operator='cs_zscore', operator_version='1',
                         params=_PARAMS, clock_projection=clock_projection))
    schema_ref = _ref(dict(schema=meta['schema'], output_schema=meta['output_schema']))
    numeric = dict(keys_ref=keys_ref, schema_ref=schema_ref, spec_ref=spec_ref,
                   reason_dictionary=reasons, **{k:descriptions[k] for k in
                   ('values','value_validity','value_reason_codes','reference_member')})
    source = dict(source_bindings_by_session=meta['source_bindings_by_session'],
                  **{k:descriptions[k] for k in ('fact_source_codes','reference_source_codes')})
    context = dict(calendar_ref=meta['calendar_ref'], keys_ref=keys_ref,
                   **{k:descriptions[k] for k in ('reference_member','reference_available_at_utc_us','selection_cutoff_utc_us')})
    implementation = _ref(dict(source_ref=IMPLEMENTATION_REF, arithmetic_backend='python-stdlib-statistics-fsum-v1',
                                python_implementation=sys.implementation.name, python_version=list(sys.version_info[:3])))
    result['metadata'] = dict(input_ref=_ref(input_document), numeric_input_ref=_ref(numeric),
                              spec_ref=spec_ref, keys_ref=keys_ref, schema_ref=schema_ref,
                              source_ref=_ref(source), context_ref=_ref(context),
                              values_digest=_ref(output_descriptions['values']),
                              flags_digest=_ref(dict(reason_dictionary=result['reason_dictionary'],
                                   **{k:output_descriptions[k] for k in ('value_validity','value_reason_codes')})),
                              clocks_digest=_ref(output_descriptions['available_at_utc_us']), implementation_ref=implementation)
    result['metadata']['result_ref'] = _ref({k:output_descriptions.get(k,v) for k,v in result.items()})
    return result
