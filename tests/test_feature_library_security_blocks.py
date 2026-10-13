"""Synthetic security growth: original immutable blocks plus batched gaps."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from axiom_engine.core import FeaturePlan
from axiom_engine.feature_library import FeatureLibrary, FeatureRequest
from axiom_engine.feature_library.codec import digest, read_json, seal
from axiom_engine.feature_library.storage import write_part
from test_incremental_feature_library import FixedSource, decoded


class BlockSource(FixedSource):
    def request_axis(self, columns, days, securities):
        original=self.request(columns,days)
        p=original.plan.to_dict()
        p['reference_members']={d:{s:g for s,g in members.items() if s in securities}
                                for d,members in p['reference_members'].items()}
        self.physical=list(securities)
        return FeatureRequest(FeaturePlan.from_dict(p),tuple(days),tuple(securities),original.input_semantics)

    def dependency_refs(self, definitions, days, securities):
        result=super().dependency_refs(definitions,days,securities)
        for day in days:
            for name,definition in definitions.items():
                if definition['reference_dependent']:
                    result[day][name]=digest([result[day][name],self.physical])
        return result

    def prepare(self, plan, day, securities, dependencies):
        inputs=super().prepare(plan,day,securities,dependencies)
        positions=[inputs['security_ids'].index(s) for s in securities]
        for field in ('values','validity','reason_codes','available_at_utc_us','source_codes'):
            inputs[field]={name:value[:,positions] for name,value in inputs[field].items()}
        for field in ('reference_member','reference_industry','reference_available_at_utc_us','reference_source_codes'):
            inputs[field]=inputs[field][:,positions]
        inputs['security_ids']=list(securities)
        self.calls[-1]=(*self.calls[-1],list(securities))
        return inputs


class SecurityBlockTests(unittest.TestCase):
    def test_growth_batches_new_securities_and_recomputes_complete_cs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=BlockSource(); days=source.c['sessions'][2:4]
            columns=['raw_x','mean_x','raw_y','z_x']
            library=FeatureLibrary(root/'cache',maximum_working_bytes=2**20)
            old,_=library.ensure_features(source.request_axis(columns,days,['A']),source)
            with library.open_feature_view(old) as view:
                before=view.project(); receipts={d:view.column_receipts(d) for d in days}
            old_bytes={r['partition']['path']:Path(r['partition']['path']).read_bytes()
                       for group in receipts.values() for r in group.values()}
            source.calls.clear(); request=source.request_axis(columns,days,['A','B','C'])
            grown,stats=library.ensure_features(request,source)
            self.assertEqual((stats['core_calls'],stats['computed_output_cells']),(4,18))
            self.assertEqual(source.calls,[(day,names,axis) for day in days for names,axis in
                [(['raw_x','mean_x','raw_y'],['B','C']),(['z_x'],['A','B','C'])]])
            with library.open_feature_view(grown) as view:
                actual=view.project()
                for day in days:
                    for name in columns[:3]:
                        blocks=view.column_receipts(day,[name])[name]['blocks']
                        self.assertEqual({tuple(b['security_ids']) for b in blocks},{('A',),('B','C')})
                        self.assertIn(receipts[day][name]['partition'],[b['partition'] for b in blocks])
            fresh=FeatureLibrary(root/'fresh',maximum_working_bytes=2**20)
            expected_ref,_=fresh.ensure_features(request,source)
            with fresh.open_feature_view(expected_ref) as view: expected=view.project()
            for name in ('values','validity','available_at_utc_us'):
                np.testing.assert_array_equal(actual[name],expected[name])
            self.assertEqual(decoded(actual,'reason_codes'),decoded(expected,'reason_codes'))
            warm_view,warm=library.ensure_features(request,source)
            self.assertEqual(warm_view['view_ref'],grown['view_ref'])
            self.assertEqual((warm['core_calls'],warm['computed_output_cells']),(0,0))
            self.assertTrue(all(Path(path).read_bytes()==value for path,value in old_bytes.items()))
            with library.open_feature_view(old) as view:
                np.testing.assert_array_equal(view.project()['values'],before['values'])

    def test_lookback_change_rejects_only_affected_old_column(self):
        with tempfile.TemporaryDirectory() as temp:
            source=BlockSource(); day=source.c['sessions'][2]
            library=FeatureLibrary(Path(temp),maximum_working_bytes=2**20)
            columns=['raw_x','mean_x','raw_y']
            library.ensure_features(source.request_axis(columns,[day],['A']),source)
            prior=next(r for r in source.f['rows'] if r['security_id']=='A' and r['session']==source.c['sessions'][1])
            prior['availability'][0]=prior['availability'][0].replace('12:00:00','12:00:01')
            request=source.request_axis(columns,[day],['A','B','C'])
            coverage=library.plan_coverage(request,source)
            self.assertEqual({g['column']:g['security_ids'] for g in coverage['gaps']},
                {'raw_x':['B','C'],'mean_x':['A','B','C'],'raw_y':['B','C']})

    def test_multiblock_view_rejects_overlap_and_holes_and_reads_v2(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=BlockSource(); days=source.c['sessions'][2:3]
            library=FeatureLibrary(root,maximum_working_bytes=2**20)
            old,_=library.ensure_features(source.request_axis(['raw_x'],days,['A']),source)
            wire=read_json(old['path']); wire.pop('view_ref'); wire.pop('security_axes')
            wire['contract_version']='feature_library_view_v2'
            legacy=write_part(root,seal(wire,'view_ref'),'view_ref')
            with library.open_feature_view(legacy) as view: self.assertEqual(view.project()['values'].shape,(1,1))
            grown,_=library.ensure_features(source.request_axis(['raw_x'],days,['A','B','C']),source)
            wire=read_json(grown['path']); wire.pop('view_ref')
            for overlapping in (True,False):
                bad=deepcopy(wire)
                if overlapping: bad['blocks'].append(deepcopy(bad['blocks'][0]))
                else: bad['blocks'].pop()
                descriptor=write_part(root,seal(bad,'view_ref'),'view_ref')
                with self.assertRaisesRegex(ValueError,'duplicate|incomplete'):
                    library.open_feature_view(descriptor)


if __name__=='__main__': unittest.main()
