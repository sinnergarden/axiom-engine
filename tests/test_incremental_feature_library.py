"""One short vertical cache case through the original Core, not production evidence."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from axiom_engine.core import FeaturePlan
from axiom_engine.core.feature_columns import _needed
from axiom_engine.feature_library import FeatureLibrary, FeatureRequest
from axiom_engine.feature_library.codec import digest
from test_conformance import setup, node, rolling, cs, col
from test_feature_columns import packed


class FixedSource:
    """Explicit synthetic selected facts; no Data files, supplier or hidden math."""
    def __init__(self):
        nodes = [node('raw_x','identity',['x']), rolling('mean_x','x',2,1),
                 node('raw_y','identity',['y']), cs('z_x','cs_zscore','raw_x')]
        self.p,self.f,self.c=setup({'A':[[1,10],[2,20],[3,30],[4,40],[5,50],[6,60]],
            'B':[[2,20],[4,40],[6,60],[8,80],[10,100],[12,120]],
            'C':[[3,30],[6,60],[9,90],[12,120],[15,150],[18,180]]}, nodes,
            ['raw_x','mean_x','raw_y','z_x'],schema=[col('x',stage='fact'),col('y',stage='fact')])
        self.calls=[]

    def request(self, columns, days=None, sample=None):
        p=deepcopy(self.p)
        by_name={o['column']['name']:o for o in p['outputs']}
        p['outputs']=[by_name[c] for c in columns]
        return FeatureRequest(FeaturePlan.from_dict(p),tuple(days or self.c['sessions'][2:]),
            ('A','B','C'),{'x':{'clock':'explicit','anchor':'output_day'},
                         'y':{'clock':'explicit','unit':'dimensionless'}},
            None if sample is None else tuple(sample))

    def dependency_refs(self, definitions, days, securities):
        result={}
        for day in days:
            result[day]={}
            for name, definition in definitions.items():
                sem=definition['semantic']; needed=_needed({'input_schema':self.p['input_schema'],
                    'nodes':sem['nodes'],'outputs':[sem['output']]},[self.c['sessions'].index(day)])
                selected=[]
                for field in definition['inputs']:
                    j=[c['name'] for c in self.p['input_schema']].index(field)
                    for r in sorted(needed[field]):
                        d=self.c['sessions'][r]
                        for fact in self.f['rows']:
                            if fact['session']==d and fact['security_id'] in securities:
                                selected.append([field,d,fact['security_id'],fact['values'][j],
                                                 fact['availability'][j],fact['missing_reasons'][j]])
                receipt={'selected':selected,'cutoff':self.c['cutoffs'][day],'anchor':day}
                if definition['reference_dependent']:
                    receipt['reference']=[r for r in self.c['reference'] if r['session']==day]
                result[day][name]=digest(receipt)
        return result

    def prepare(self, plan, day, securities, dependencies):
        self.calls.append((day,[o['column']['name'] for o in plan.to_dict()['outputs']]))
        inputs=packed(plan.to_dict(),self.f,self.c)
        inputs['input_refs']['feature_dependency_ref']=digest(dependencies)
        return inputs

    def change_reference(self, day):
        self.p['reference_members'][day].pop('C')
        next(r for r in self.c['reference'] if r['session']==day and r['security_id']=='C')['member']=False


def decoded(projection, name):
    return [[projection['reason_dictionary'][int(i)] for i in row] for row in projection[name]]


class IncrementalFeatureLibraryTests(unittest.TestCase):
    def test_shared_admission_epoch_reuses_hashes_but_detects_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            library=FeatureLibrary(temp,maximum_working_bytes=2**20); source=FixedSource()
            request=source.request(['raw_x'],source.c['sessions'][2:3])
            with self.assertRaisesRegex(ValueError,'saved matrix source changed'):
                with library.execution_session():
                    descriptor,_=library.ensure_features(request,source)
                    with library.open_feature_view(descriptor) as view:
                        view.project(); hashes=view._store.metrics['file_hash_calls']
                        _,stats=library.ensure_features(request,source)
                        self.assertEqual(stats['core_calls'],0)
                        self.assertEqual(stats['admission_file_hash_calls'],0)
                        self.assertEqual(view._store.metrics['file_hash_calls'],hashes)
                        tile=view._view['tiles'][0]['partition']
                        part=view._store.read_json({k:tile[k] for k in ('path','file_digest','partition_ref')},'partition_ref')
                        path=Path(part['buffers']['values']['path'])
                        original=path.read_bytes(); path.write_bytes(original)
                        view.project()
            # New admission uses the actual unchanged bytes, without inheriting trust.
            descriptor,stats=library.ensure_features(request,source)
            self.assertGreater(stats['admission_file_hash_calls'],0)
            with library.open_feature_view(descriptor) as view:
                self.assertEqual(view.project()['values'][:,0].tolist(),[3,6,9])

    def test_rectangle_subset_append_pool_revision_and_one_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            source=FixedSource(); days=source.c['sessions']; library=FeatureLibrary(Path(temp)/'incremental',maximum_working_bytes=2**20,column_block=1)
            # First build; Core supplies all math, bytes are projected back exactly.
            first,stats=library.ensure_features(source.request(['raw_x','mean_x'],days[2:4]),source)
            self.assertEqual((stats['core_calls'],stats['computed_output_cells']),(2,12))
            with library.open_feature_view(first) as view:
                initial=view.project(); initial_refs=view._view['tiles']
                self.assertEqual(initial['values'][:,0].tolist(),[3,6,9,4,8,12])
                self.assertEqual(view.describe()['complete'],True)
            # Sample subset and feature subset change projection only.
            subset,stats=library.ensure_features(source.request(['raw_x'],days[2:4],sample=['B']),source)
            self.assertEqual(stats['core_calls'],0)
            with library.open_feature_view(subset) as view:
                self.assertEqual(view.project()['values'].reshape(-1).tolist(),[6,8])
            # Expand two old days by one column, add two days by all three columns.
            expanded,stats=library.ensure_features(source.request(['raw_x','mean_x','raw_y']),source)
            self.assertEqual((stats['core_calls'],stats['computed_output_cells']),(4,24))
            with library.open_feature_view(expanded) as view:
                expanded_values=view.project()
                for old in initial_refs:
                    new=next(t for t in view._view['tiles'] if (t['session'],t['column'])==(old['session'],old['column']))
                    self.assertEqual(new['partition']['partition_ref'],old['partition']['partition_ref'])
            fresh=FeatureLibrary(Path(temp)/'once',maximum_working_bytes=2**20)
            once,_=fresh.ensure_features(source.request(['raw_x','mean_x','raw_y']),source)
            with fresh.open_feature_view(once) as view:
                expected=view.project()
                for name in ('values','validity','available_at_utc_us'): np.testing.assert_array_equal(expanded_values[name],expected[name])
                self.assertEqual(decoded(expanded_values,'reason_codes'),decoded(expected,'reason_codes'))
            # Changing the reference pool changes only the CS output, on that date.
            library.ensure_features(source.request(['raw_x','mean_x','raw_y','z_x']),source)
            source.change_reference(days[3]); request=source.request(['raw_x','mean_x','raw_y','z_x'])
            coverage=library.plan_coverage(request,source)
            self.assertEqual(coverage['gaps'],[{'session':days[3],'column':'z_x'}])
            library.ensure_features(request,source)
            # Revising x at day 3 propagates only to x/mean/CS dependencies.
            next(r for r in source.f['rows'] if r['session']==days[3] and r['security_id']=='A')['values'][0]=100
            coverage=library.plan_coverage(source.request(['raw_x','mean_x','raw_y','z_x']),source)
            self.assertEqual({(g['session'],g['column']) for g in coverage['gaps']},
                {(days[3],'raw_x'),(days[3],'mean_x'),(days[4],'mean_x'),(days[3],'z_x')})
            # Interrupt once after sealing a tile, before updating the coverage index.
            interrupted=FeatureLibrary(Path(temp)/'resume',maximum_working_bytes=2**20)
            def stop(update): raise RuntimeError('one intentional interruption')
            with self.assertRaisesRegex(RuntimeError,'intentional interruption'):
                interrupted.ensure_features(source.request(['raw_y'],days[2:4]),source,progress=stop)
            restored,stats=interrupted.ensure_features(source.request(['raw_y'],days[2:4]),source)
            self.assertEqual((stats['recovered_partitions'],stats['core_calls']),(1,1))
            with interrupted.open_feature_view(restored) as view:
                self.assertEqual(view.project()['values'][:,0].tolist(),[30,60,90,40,80,120])


if __name__=='__main__': unittest.main()
