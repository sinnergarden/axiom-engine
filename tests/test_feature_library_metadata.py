"""Metadata scale and old immutable reader checks; no production Data run."""
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from axiom_engine.feature_library import FeatureLibrary, VerifiedMatrixStore
from axiom_engine.feature_library.codec import digest, seal, file_digest, write_json, read_json
from axiom_engine.feature_library.coverage import compact_view, write_date, publish_index, INDEX_VERSION
from axiom_engine.feature_library.storage import write_part
from test_incremental_feature_library import FixedSource, tiles, decoded


def metadata_scale(root, definitions=None):
    """2242 dates x 88 columns; synthetic refs stand in for sealed data partitions.

    Only new coverage/view transport is written. Values, Core partition headers
    and their original buffers are not calculated or fabricated by this check.
    Half/full checkpoints use the same immutable date children.
    """
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    days=[(date(2017,1,1)+timedelta(days=i)).isoformat() for i in range(2242)]
    if definitions is None:
        columns=[f'F{i:03d}' for i in range(88)]
        definitions={c:{'semantic_ref':digest(['semantics',c]),
            'schema':{'name':c,'dtype':'float64','unit':'dimensionless','stage':'raw','missing':'preserve'}}
            for c in columns}
    columns=list(definitions)
    assert len(columns)==88
    securities=[f'S{i:06d}' for i in range(3497)]
    shards={}; hits={}; checkpoints=[]; date_bytes=[]
    for ordinal,day in enumerate(days):
        parts=[]
        for start in range(0,88,16):
            selected=columns[start:start+16]; ref=digest(['synthetic metadata only',day,start])
            summary={'path':str(root/'partitions'/(ref[7:]+'.json')),
                'file_digest':digest(['synthetic physical bytes',day,start]),'partition_ref':ref,
                'sessions':[day],'security_axis_ref':digest(securities),'columns':selected,
                'semantic_refs':{c:definitions[c]['semantic_ref'] for c in selected},
                'dependency_refs':{c:digest(['local input',day,c]) for c in selected}}
            parts.append(summary)
            for c in selected: hits[day,c]={'partition':summary}
        shards[day]=write_date(root,day,parts)
        date_bytes.append(Path(shards[day]['path']).stat().st_size)
        if ordinal+1 in (1121,2242):
            publish_index(root,shards)
            coverage={'sessions':days[:ordinal+1],'security_ids':securities,
                'sample_universe':securities,'definitions':definitions}
            view=compact_view(coverage,hits); descriptor=write_part(root,view,'view_ref')
            with VerifiedMatrixStore() as store:
                admitted=store.read_json(descriptor,'view_ref')
                assert len(admitted['blocks'])==6*(ordinal+1)
            checkpoints.append({'dates':ordinal+1, 'root_bytes':(root/'index.json').stat().st_size,
                'view_bytes':Path(descriptor['path']).stat().st_size})
    files=list((root/'metadata').glob('*.json'))+[root/'index.json']
    return {'scope':{'dates':2242,'columns':88,'securities':3497,'column_block':16},
        'checkpoints':checkpoints,'maximum_root_bytes':max(c['root_bytes'] for c in checkpoints),
        'maximum_view_bytes':max(c['view_bytes'] for c in checkpoints),
        'maximum_date_parent_bytes':max(date_bytes),
        'total_retained_transport_metadata_bytes':sum(p.stat().st_size for p in files),
        'date_writes':len(date_bytes),'root_writes':2,'view_writes':2,
        'parent_budget_bytes':64*1024**2,
        'scope_note':'transport metadata only; immutable data partitions/headers/buffers excluded'}


class FeatureMetadataTests(unittest.TestCase):
    def test_2242_dates_88_columns_have_linear_transport_and_bounded_parents(self):
        from axiom_engine.feature_library import coverage, storage
        with tempfile.TemporaryDirectory() as temp:
            writes=[]
            def record(path,value):
                write_json(path,value); writes.append(Path(path).stat().st_size)
            with patch.object(coverage,'write_json',side_effect=record), \
                 patch.object(storage,'write_json',side_effect=record):
                result=metadata_scale(Path(temp)/'scale')
            first,last=result['checkpoints']
            self.assertLess(last['root_bytes'],2*first['root_bytes']+4096)
            self.assertLess(last['view_bytes'],2*first['view_bytes'])
            self.assertLess(max(result['maximum_root_bytes'],result['maximum_view_bytes'],
                                result['maximum_date_parent_bytes']),64*1024**2)
            self.assertEqual(len(writes),2242+2+2)
            self.assertLess(sum(writes),result['total_retained_transport_metadata_bytes']*1.02)

    def test_old_v1_cold_read_migrates_only_coverage_and_rejects_oversized_before_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            source=FixedSource(); library=FeatureLibrary(Path(temp)/'cache',maximum_working_bytes=2**20)
            request=source.request(['raw_x','mean_x'],source.c['sessions'][2:4])
            current,_=library.ensure_features(request,source)
            with library.open_feature_view(current) as view:
                expected=view.project(); selected=tiles(view); cap=view.describe()
                legacy_tiles=[]; summaries={}
                for t in selected:
                    part=view._store.read_json(t['partition'],'partition_ref')
                    summary=library._summary(t['partition'],part)
                    summaries[part['partition_ref']]=summary
                    legacy_tiles.append({**t,'partition':summary})
            legacy=seal({'contract_version':'feature_library_view_v1',
                **{k:cap[k] for k in ('sessions','security_ids','sample_universe','columns','definitions')},
                'tiles':legacy_tiles},'view_ref')
            old=write_part(library.root,legacy,'view_ref')
            write_json(library.root/'index.json',seal({'contract_version':'feature_library_coverage_v1',
                'partitions':list(summaries.values())},'index_ref'))
            with library.open_feature_view(old) as view:
                actual=view.project()
                for k in ('values','validity','available_at_utc_us'):
                    np.testing.assert_array_equal(actual[k],expected[k])
                self.assertEqual(decoded(actual,'reason_codes'),decoded(expected,'reason_codes'))
                self.assertEqual(actual['source_sets'],expected['source_sets'])
                self.assertEqual(actual['source_bindings'],expected['source_bindings'])
            with patch.object(library,'_publish_index',wraps=library._publish_index) as publish:
                migrated,stats=library.ensure_features(request,source)
                self.assertEqual(publish.call_count,1)
            self.assertEqual(stats['core_calls'],0)
            self.assertEqual(read_json(library.root/'index.json')['contract_version'],INDEX_VERSION)
            with library.open_feature_view(migrated) as view:
                self.assertEqual({t['partition']['partition_ref'] for t in tiles(view)},set(summaries))
                np.testing.assert_array_equal(view.project()['values'],expected['values'])
            self.assertEqual(read_json(old['path']),legacy)
            with VerifiedMatrixStore(maximum_parent_bytes=8) as store:
                with patch('axiom_engine.feature_library.store.file_digest',side_effect=AssertionError('must not hash')):
                    with self.assertRaisesRegex(ValueError,'parent byte budget'):
                        store.read_json(current,'view_ref')
                self.assertEqual(store.metrics['file_hash_calls'],0)


if __name__=='__main__': unittest.main()
