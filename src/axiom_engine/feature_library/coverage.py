"""Rebuildable date coverage and compact selections; never Feature arithmetic."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile

from .codec import digest, seal, require, read_json, write_json, file_fingerprint
from .storage import write_part
from .store import VerifiedMatrixStore

INDEX_V1 = 'feature_library_coverage_v1'
INDEX_VERSION = 'feature_library_coverage_v2'
DATE_VERSION = 'feature_library_date_coverage_v2'
VIEW_V1 = 'feature_library_view_v1'
VIEW_V2 = 'feature_library_view_v2'
VIEW_VERSION = 'feature_library_view_v3'
PART_FIELDS = ('path', 'file_digest', 'partition_ref')


def partition_descriptor(summary):
    return {k: summary[k] for k in PART_FIELDS}


def coverage_summary(descriptor, part):
    """Source payload stays in the original sealed partition, once per block."""
    return {**partition_descriptor(descriptor), 'sessions': part['sessions'],
        'security_axis_ref': digest(part['security_ids']), 'columns': part['columns'],
        'semantic_refs': part['semantic_refs'], 'dependency_refs': part['dependency_refs']}


def read_index(root, maximum_parent_bytes, sessions=None):
    """Read bounded root/date parents, validating bytes before decoding each child."""
    path = Path(root) / 'index.json'
    if not path.exists(): return [], {}
    require(not path.is_symlink() and path.stat().st_size <= maximum_parent_bytes,
            'coverage index exceeds metadata budget')
    mark = file_fingerprint(path)
    value = read_json(path)
    require(file_fingerprint(path) == mark and value.get('index_ref') ==
            digest({k:v for k,v in value.items() if k != 'index_ref'}),
            'Feature coverage index mismatch')
    version = value.get('contract_version')
    if version == INDEX_V1:
        # Only transport metadata changes. Old immutable partitions are untouched.
        return [{k:v for k,v in p.items() if k != 'source_bindings'}
                for p in value['partitions']], {}
    require(version == INDEX_VERSION, 'unsupported Feature coverage index')
    partitions = []; shards = {}
    for entry in value['dates']:
        day = entry['session']
        require(day not in shards and set(entry) == {'session','coverage'},
                'unique date coverage descriptor required')
        descriptor = entry['coverage']
        shards[day] = descriptor
        if sessions is not None and day not in sessions:continue
        # A short-lived metadata owner prevents repeated checks of all previous
        # date files while still checking each actual file and its content ref.
        with VerifiedMatrixStore(maximum_parent_bytes=maximum_parent_bytes) as store:
            child = store.read_json(descriptor, 'coverage_ref')
            require(child['contract_version'] == DATE_VERSION and child['session'] == day and
                    all(p['sessions'] == [day] for p in child['partitions']),
                    'date coverage scope mismatch')
            partitions.extend(deepcopy(child['partitions']))
        shards[day] = descriptor
    require(file_fingerprint(path) == mark, 'coverage index changed during read')
    return partitions, shards


def write_date(root, day, partitions):
    require(all(p['sessions'] == [day] for p in partitions), 'date coverage scope mismatch')
    return write_part(root, seal({'contract_version': DATE_VERSION, 'session': day,
        'partitions': partitions}, 'coverage_ref'), 'coverage_ref')


def publish_index(root, shards):
    """Publish the small root once after a request; date children are immutable."""
    value = seal({'contract_version': INDEX_VERSION, 'dates':
        [{'session':day, 'coverage':shards[day]} for day in sorted(shards)]}, 'index_ref')
    with tempfile.NamedTemporaryFile(dir=root, prefix='.index-', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        write_json(temporary, value)
        os.replace(temporary, Path(root)/'index.json')
    finally:
        temporary.unlink(missing_ok=True)


def compact_view(coverage, hits):
    """One descriptor per selected block; no semantic or provenance copies."""
    blocks = {}; axes={}; axis_refs={}
    for day in coverage['sessions']:
        for column in coverage['definitions']:
            selections=hits[day,column]
            if isinstance(selections,dict): selections=[selections]
            for hit in sorted(selections,key=lambda h:tuple(h.get('security_ids',coverage['security_ids']))):
                descriptor = partition_descriptor(hit['partition'])
                securities=hit.get('security_ids',coverage['security_ids'])
                axis_ref=None
                if securities!=coverage['security_ids']:
                    axis=tuple(securities)
                    if axis not in axis_refs: axis_refs[axis]=digest(securities)
                    axis_ref=axis_refs[axis]; axes[axis_ref]=securities
                key = day, descriptor['partition_ref'], axis_ref
                if key not in blocks:
                    blocks[key] = {'session':day, 'columns':[], 'partition':descriptor}
                    if axis_ref is not None: blocks[key]['security_axis_ref']=axis_ref
                require(blocks[key]['partition'] == descriptor, 'conflicting Feature block descriptor')
                blocks[key]['columns'].append(column)
    return seal({'contract_version': VIEW_VERSION,
        **{k:coverage[k] for k in ('sessions','security_ids','sample_universe','definitions')},
        'columns':list(coverage['definitions']), 'security_axes':axes, 'blocks':list(blocks.values())}, 'view_ref')
