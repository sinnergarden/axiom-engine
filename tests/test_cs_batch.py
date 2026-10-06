"""Synthetic packed admission, exact row-wire equivalence and identity controls."""
import copy
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import struct
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError, execute_cs_zscore_batch
from test_conformance import SOURCES, col, cs, run, setup

PARAMS = dict(group='session', unknown_group='reject', missing='skip', ddof=0,
              epsilon=1e-12, constant='missing', clip=None, excluded='missing')
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def pack(values, code):
    return memoryview(struct.pack('<'+str(len(values))+code, *values)).cast(code)


def us(value):
    delta = datetime.fromisoformat(value.replace('Z','+00:00'))-EPOCH
    return (delta.days*86400+delta.seconds)*1_000_000+delta.microseconds


def clock(value):
    projected = -((-value)//1_000_000)*1_000_000
    return (EPOCH+timedelta(microseconds=projected)).strftime('%Y-%m-%dT%H:%M:%SZ')


def batch(data, members=None):
    p,f,c = setup(data,[cs('normalized_target','cs_zscore',src='raw_return',**PARAMS)],
                  members=members,schema=[col('raw_return',stage='fact')])
    days, securities = c['sessions'], sorted(data)
    rows = {(r['security_id'],r['session']):r for r in f['rows']}
    refs = {(r['security_id'],r['session']):r for r in c['reference']}
    keys = [(s,d) for d in days for s in securities]
    vals = [rows[k]['values'][0] for k in keys]
    return dict(contract_version='core_cs_zscore_batch_input_v1',calendar_ref=p['calendar_ref'],
                schema=p['input_schema'],output_schema=[p['nodes'][0]['column']],sessions=days,security_ids=securities,
                values=pack([v if v is not None else 0.0 for v in vals],'d'),
                value_validity=pack([v is not None for v in vals],'?'),
                value_reason_codes=pack([0 if v is not None else 1 for v in vals],'i'),
                reason_dictionary=[None,'NOT_REPORTED'],reference_member=pack([refs[k]['member'] for k in keys],'?'),
                fact_available_at_utc_us=pack([us(rows[k]['availability'][0]) for k in keys],'q'),
                reference_available_at_utc_us=pack([us(refs[k]['available_at']) for k in keys],'q'),
                selection_cutoff_utc_us=pack([us(c['cutoffs'][d]) for d in days],'q'),
                source_bindings_by_session={d:dict(bindings=copy.deepcopy(p['sources']),source_sets=[['facts']]) for d in days},
                fact_source_codes=pack([0]*len(keys),'i'),reference_source_codes=pack([0]*len(keys),'i'))


def clone(b):
    return {k:v if isinstance(v,memoryview) else copy.deepcopy(v) for k,v in b.items()}


def old_rows(b):
    """Map the same complete cohort to the public row ABI; no math in adapter."""
    out, width = [], len(b['security_ids'])
    for day_index,day in enumerate(b['sessions']):
        begin = day_index*width
        data = {s:[b['values'][begin+j] if b['value_validity'][begin+j] else None] for j,s in enumerate(b['security_ids'])}
        members = {s:[b['reference_member'][begin+j]] for j,s in enumerate(b['security_ids'])}
        p,f,c = setup(data,[cs(b['output_schema'][0]['name'],'cs_zscore',src=b['schema'][0]['name'],**PARAMS)],
                      members=members,schema=copy.deepcopy(b['schema']))
        original_day = c['sessions'][0]
        p['nodes'][0]['column'] = copy.deepcopy(b['output_schema'][0])
        p['outputs'][0]['column'] = copy.deepcopy(b['output_schema'][0])
        p['reference_members'] = {day:p['reference_members'][original_day]}
        p['sources'] = copy.deepcopy(b['source_bindings_by_session'][day]['bindings'])
        f['sources'] = copy.deepcopy(p['sources'])
        c.update(sessions=[day],cutoffs={day:clock(b['selection_cutoff_utc_us'][day_index])})
        sets = b['source_bindings_by_session'][day]['source_sets']
        for j,(row,ref) in enumerate(zip(f['rows'],c['reference'])):
            index = begin+j
            row.update(session=day,availability=[clock(b['fact_available_at_utc_us'][index])],
                       sources=[sets[b['fact_source_codes'][index]]],
                       missing_reasons=[b['reason_dictionary'][b['value_reason_codes'][index]]])
            ref.update(session=day,available_at=clock(b['reference_available_at_utc_us'][index]),
                       source=sets[b['reference_source_codes'][index]][0])
        for field in ('history_keys','output_keys'):c[field]=[[s,day] for s in b['security_ids']]
        out.extend(run(p,f,c)['rows'])
    return out


class CSBatch(unittest.TestCase):
    def assertRowExact(self,b,result):
        for index,row in enumerate(old_rows(b)):
            present = result['value_validity'][index]
            self.assertEqual(present,row['valid'][0])
            actual = struct.pack('<d',result['values'][index])
            self.assertEqual(actual,struct.pack('<d',row['values'][0]) if present else b'\0'*8)
            reason = result['reason_dictionary'][result['value_reason_codes'][index]]
            self.assertEqual([] if reason is None else [reason],row['reasons'][0])
            self.assertEqual(clock(result['available_at_utc_us'][index]),row['availability'][0])
            day = b['sessions'][index//len(b['security_ids'])]
            sets = result['source_bindings_by_session'][day]['source_sets']
            self.assertEqual(sets[result['source_codes'][index]],row['sources'][0])

    def test_exact_values_masks_reasons_clocks_and_multisource(self):
        samples = ([1,3],[None,3],[None,None],[5,5],[-0.0,1,2],[-1e-12,1e-12],
                   [-math.nextafter(1e-12,0),math.nextafter(1e-12,0)],
                   [-1e150,1e150],[-1e308,1e308],[1e-200,3e-200],[1,1,3])
        for vals in samples:
            for mask in ([True]*len(vals),[False]*len(vals),[i%2==0 for i in range(len(vals))]):
                with self.subTest(values=vals,mask=mask):
                    b = batch({str(i):[v] for i,v in enumerate(vals)},members={str(i):[m] for i,m in enumerate(mask)})
                    self.assertRowExact(b,execute_cs_zscore_batch(b,params=PARAMS))
        b = batch({'A':[1,None],'B':[3,5],'N':[9,8]},members={'A':[True,True],'B':[True,True],'N':[False,False]})
        for day in b['sessions']:
            b['source_bindings_by_session'][day]=dict(bindings=[dict(SOURCES[0],id=s) for s in 'abcd'],
                                                      source_sets=[['a'],['a','b'],['c'],['d']])
        b['fact_source_codes']=pack([1,2,3,1,2,3],'i')
        b['reference_source_codes']=pack([0,0,2,0,0,2],'i')
        refs=list(b['reference_available_at_utc_us'])
        refs[2]=b['selection_cutoff_utc_us'][0]-111
        refs[5]=b['selection_cutoff_utc_us'][1]-111
        b['reference_available_at_utc_us']=pack(refs,'q')
        self.assertRowExact(b,execute_cs_zscore_batch(b,params=PARAMS))

    def test_microsecond_gate_before_ceil_and_negative_epoch(self):
        for value in (1000000,1000001,-1,-1000001):
            b=batch({'A':[1],'B':[3]})
            b['fact_available_at_utc_us']=pack([value,value],'q')
            b['reference_available_at_utc_us']=pack([value,value],'q')
            b['selection_cutoff_utc_us']=pack([value],'q')
            self.assertRowExact(b,execute_cs_zscore_batch(b,params=PARAMS))
        b['fact_available_at_utc_us']=pack([1000001,1000000],'q')
        b['reference_available_at_utc_us']=pack([1000000]*2,'q')
        b['selection_cutoff_utc_us']=pack([1000000],'q')
        with self.assertRaisesRegex(ContractError,'NATIVE_CUTOFF'):execute_cs_zscore_batch(b,params=PARAMS)
        b['fact_available_at_utc_us']=pack([0,0],'q');b['reference_available_at_utc_us']=pack([0,0],'q')
        b['selection_cutoff_utc_us']=pack([2**63-1],'q')
        with self.assertRaisesRegex(ContractError,'PROJECTION_OVERFLOW'):execute_cs_zscore_batch(b,params=PARAMS)

    def test_immutable_inputs_and_owned_readonly_outputs(self):
        b=batch({'A':[1,None],'B':[3,4]})
        before={k:v.tobytes() for k,v in b.items() if isinstance(v,memoryview)}
        meta=copy.deepcopy({k:v for k,v in b.items() if not isinstance(v,memoryview)})
        result=execute_cs_zscore_batch(b,params=PARAMS)
        self.assertEqual(before,{k:v.tobytes() for k,v in b.items() if isinstance(v,memoryview)})
        self.assertEqual(meta,{k:v for k,v in b.items() if not isinstance(v,memoryview)})
        for name in ('values','value_validity','value_reason_codes','available_at_utc_us','source_codes'):
            view=result[name]
            self.assertTrue(view.readonly and view.c_contiguous and view.ndim==1)
            self.assertIsInstance(view.obj,bytes)
            with self.assertRaises(TypeError):view[0]=1
        b['output_schema'][0]['name']='changed'
        b['source_bindings_by_session'][b['sessions'][0]]['bindings'][0]['id']='changed'
        self.assertEqual(result['schema'][0]['name'],'normalized_target')
        self.assertEqual(result['source_bindings_by_session'][result['sessions'][0]]['bindings'][0]['id'],'facts')

    def test_buffer_layout_and_readonly_alias_snapshot(self):
        b=batch({'A':[1],'B':[3]})
        big_endian=(ctypes.c_double.__ctype_be__*2)(1,3)
        invalid=(pack([1,3],'d').cast('B').cast('d',shape=[1,2]),
                 pack([1,2,3,4],'d')[::2], memoryview(big_endian).toreadonly())
        for view in invalid:
            with self.subTest(format=view.format,shape=view.shape),self.assertRaises(ContractError):
                execute_cs_zscore_batch(dict(b,values=view),params=PARAMS)
        raw=bytearray(b['values'].tobytes())
        b['values']=memoryview(raw).cast('d').toreadonly()
        result=execute_cs_zscore_batch(b,params=PARAMS)
        saved=result['values'].tobytes(),result['metadata']['result_ref']
        raw[:]=struct.pack('<2d',100,100)
        self.assertEqual(saved,(result['values'].tobytes(),result['metadata']['result_ref']))

    def test_identity_closure_and_source_clock_changes(self):
        b=batch({'A':[1],'B':[3]});original=execute_cs_zscore_batch(b,params=PARAMS)
        changed=clone(b)
        changed['fact_available_at_utc_us']=pack([v+1 for v in b['fact_available_at_utc_us']],'q')
        moved=execute_cs_zscore_batch(changed,params=PARAMS)
        self.assertEqual(original['metadata']['numeric_input_ref'],moved['metadata']['numeric_input_ref'])
        for name in ('input_ref','result_ref'):self.assertNotEqual(original['metadata'][name],moved['metadata'][name])
        changed=clone(b)
        changed['source_bindings_by_session'][b['sessions'][0]]['bindings'][0]['view_ref']='sha256:'+'b'*64
        rebound=execute_cs_zscore_batch(changed,params=PARAMS)
        self.assertEqual(original['metadata']['numeric_input_ref'],rebound['metadata']['numeric_input_ref'])
        for name in ('source_ref','result_ref'):self.assertNotEqual(original['metadata'][name],rebound['metadata'][name])
        def ref(obj):
            raw=json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
            return 'sha256:'+hashlib.sha256(raw).hexdigest()
        formats={'values':'float64','value_validity':'bool','value_reason_codes':'int32','available_at_utc_us':'int64','source_codes':'int32'}
        wire={k:dict(dtype=formats[k],shape=[len(v)],bytes_digest='sha256:'+hashlib.sha256(v.tobytes()).hexdigest()) if k in formats else copy.deepcopy(v) for k,v in original.items()}
        saved_ref=wire['metadata'].pop('result_ref')
        self.assertEqual(saved_ref,ref(wire))
        self.assertEqual(original['metadata']['values_digest'],ref(wire['values']))
        self.assertEqual(original['metadata']['clocks_digest'],ref(wire['available_at_utc_us']))
        self.assertEqual(original['metadata']['flags_digest'],ref({k:wire[k] for k in ('value_validity','value_reason_codes','reason_dictionary')}))

    def test_bad_shapes_codes_schemas_and_no_io(self):
        b=batch({'A':[1],'B':[3]})
        mutations=[lambda x:x.update(values=[1,3]),lambda x:x.update(values=memoryview(bytearray(16)).cast('d')),
                   lambda x:x.update(values=pack([1],'d')),lambda x:x.update(values=pack([1,3],'q')),
                   lambda x:x.update(values=pack([float('inf'),3],'d')),
                   lambda x:x.update(value_validity=memoryview(b'\x02\x01').cast('?')),
                   lambda x:x.update(value_reason_codes=pack([1,0],'i')),
                   lambda x:x.update(fact_source_codes=pack([-1,0],'i')),
                   lambda x:x.update(reference_source_codes=pack([1,0],'i')),
                   lambda x:x.update(reason_dictionary=[None,'Z','A']),
                   lambda x:x['source_bindings_by_session'][x['sessions'][0]].update(source_sets=[['unknown']]),
                   lambda x:x['source_bindings_by_session'][x['sessions'][0]].update(
                       bindings=[dict(SOURCES[0],id=s) for s in ('facts','other')],
                       source_sets=[['facts','other']]),
                   lambda x:x['source_bindings_by_session'][x['sessions'][0]].update(source_sets=[['facts'],['facts']]),
                   lambda x:x.update(security_ids=['B','A']),lambda x:x.update(trusted=True),
                   lambda x:x.update(sessions=[]),lambda x:x['schema'][0].update(stage='cross_sectional'),
                   lambda x:x['output_schema'][0].update(unit='price'),lambda x:x.update(calendar_ref='latest')]
        for i,mutate in enumerate(mutations):
            x=clone(b);mutate(x)
            with self.subTest(case=i),self.assertRaises(ContractError):execute_cs_zscore_batch(x,params=PARAMS)
        x=batch({'A':[None],'B':[3]});x['schema'][0]['missing']='reject'
        with self.assertRaisesRegex(ContractError,'required column'):execute_cs_zscore_batch(x,params=PARAMS)
        x['values']=pack([-0.0,3],'d');x['schema'][0]['missing']='preserve'
        with self.assertRaisesRegex(ContractError,'CANONICAL_ZERO'):execute_cs_zscore_batch(x,params=PARAMS)
        for field,value in [('ddof',1),('ddof',False),('epsilon',0),('constant','zero'),('missing','fill_zero'),('group','industry')]:
            with self.assertRaisesRegex(ContractError,'PROFILE'):execute_cs_zscore_batch(b,params=dict(PARAMS,**{field:value}))
        with self.assertRaisesRegex(ContractError,'CLOCK_PROJECTION'):execute_cs_zscore_batch(b,params=PARAMS,clock_projection='floor')
        with patch('builtins.open',side_effect=AssertionError('I/O')),patch('socket.socket',side_effect=AssertionError('network')):
            execute_cs_zscore_batch(b,params=PARAMS)

    def test_six_and_three_hundred_columns_share_same_entry(self):
        from axiom_engine.core import cs_batch
        scale=cs_batch._cs_zscore_scale
        for columns in (6,300):
            with patch.object(cs_batch,'_cs_zscore_scale',wraps=scale) as calls:
                for index in range(columns):
                    b=batch({'A':[1,2],'B':[3,5]})
                    b['schema'][0]['name']='feature'+str(index)
                    b['output_schema'][0]['name']='normalized'+str(index)
                    self.assertRowExact(b,execute_cs_zscore_batch(b,params=PARAMS))
                self.assertEqual(calls.call_count,columns*2)

    def test_original_overflow_and_lazy_excluded_behavior(self):
        b=batch({'A':[1e308],'B':[9e307]})
        with self.assertRaises(OverflowError):old_rows(b)
        with self.assertRaises(OverflowError):execute_cs_zscore_batch(b,params=PARAMS)
        b['reference_member']=pack([False,False],'?')
        self.assertRowExact(b,execute_cs_zscore_batch(b,params=PARAMS))
