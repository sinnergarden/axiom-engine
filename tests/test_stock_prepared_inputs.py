"""Fresh-process input reuse and old business oracle; synthetic sources only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from axiom_engine.core import ContractError, stock_strategy_spec
from axiom_engine.core.contracts import canonical
from axiom_engine.runtime import (BacktestRequest, StockMarketSpec, StockInputSource,
    save_stock_market_inputs, load_stock_market_inputs, save_stock_prediction_inputs, load_stock_prediction_inputs)
from axiom_engine.runtime.stock_input_store import pack,unpack,encoded_size,measured_size
from test_stock_market_owner import market_for,bind
from test_stock_owned_inputs import request_for,run_owned
from test_stock_signal_inputs import v3_request,derived_request
from test_stock_stream_inputs import LIMITS
from test_stock_stream import execute
from test_stock_strategy import configured


def reopen(mr,pr,request,root):
    with load_stock_market_inputs(mr,StockMarketSpec.from_request(BacktestRequest.from_dict(request)),
            limits=LIMITS,max_market_bytes=2_000_000) as market:
        with load_stock_prediction_inputs(market,pr,BacktestRequest.from_dict(request),
                limits=LIMITS,max_signal_bytes=2_000_000) as inputs:
            return run_owned(root,request,inputs)


class PreparedInputTests(unittest.TestCase):
    def test_codec_preserves_types_missing_fields_order_and_bounded_expansion(self):
        value = {"rows":[{"s":"中\n\\\"😀","v":1,"b":True,"n":None,"refs":["a","b"]},
                         {"s":"中\n\\\"😀","v":1.0,"b":False,"refs":["a","b"]}],"empty":[{},{}]}
        encoded = pack(value); decoded = unpack(json.loads(canonical(encoded)))
        self.assertEqual(decoded,value); self.assertIs(type(decoded["rows"][0]["v"]),int)
        self.assertIs(type(decoded["rows"][1]["v"]),float); self.assertNotIn("n",decoded["rows"][1])
        self.assertEqual(encoded_size(value),len(canonical(value).encode()))
        self.assertEqual(measured_size(encoded,10000),encoded_size(value))
        with self.assertRaises(ContractError): measured_size(encoded,10)
        from axiom_engine.runtime.stock_owned_inputs import _input_key
        with tempfile.TemporaryDirectory() as tmp:
            request=request_for(Path(tmp)); changed=deepcopy(request)
            changed.update(risk_input={"reference":"sha256:"+"a"*64},risk_input_ref="sha256:"+"a"*64)
            self.assertNotEqual(_input_key(request),_input_key(changed))
            next_risk=deepcopy(changed); next_risk["risk_input_ref"]="sha256:"+"b"*64
            self.assertNotEqual(_input_key(changed),_input_key(next_risk))

    def test_raw_v3_and_derived_reopen_match_original_source_exactly(self):
        for kind in ("raw_v3","derived"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); request=v3_request(root)
                if kind=="derived": request=derived_request(root,request)
                expected,oracle,_=execute(root/"oracle",request)
                with market_for(request) as market,bind(market,request) as inputs:
                    mr=save_stock_market_inputs(market,root/"m",limits=LIMITS)
                    pr=save_stock_prediction_inputs(inputs,root/"p",limits=LIMITS)
                actual,view=reopen(mr,pr,request,root/"reopen")
                self.assertEqual(actual["run_id"],expected["run_id"])
                self.assertEqual(view.rows,oracle.rows)

    def test_original_native_locator_relocation_keeps_logical_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); request=request_for(root)
            with market_for(request) as market:
                mr=save_stock_market_inputs(market,root/"m",limits=LIMITS)
            relocated=deepcopy(request)
            for entry in relocated["market_input"]["native_inputs"]:
                entry["artifact"]["manifest_uri"] += ".relocated"
            with patch("axiom_engine.runtime.stock_stream_inputs._CanonicalIndex",side_effect=AssertionError("ancestor opened")):
                with load_stock_market_inputs(mr,StockMarketSpec.from_request(BacktestRequest.from_dict(relocated)),
                        limits=LIMITS,max_market_bytes=2_000_000):
                    pass

    def test_cache_hit_uses_closed_native_source_without_entering_its_scope(self):
        from test_stock_native_view import (open_native_view,synthetic_request,carrier,DATA_LIMITS,
            ADAPTER_LIMITS,ACTIVE)
        from axiom_engine.runtime import StockNativeViewSource,admit_stock_market_inputs,bind_stock_prediction_inputs
        if open_native_view is None: self.skipTest("pinned Data native consumer unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); request=synthetic_request(root); reference,_=carrier(root/"native",request)
            handle=open_native_view(root/"native",manifest_sha256=reference,limits=DATA_LIMITS)
            source=StockNativeViewSource(native_views=[handle],max_active_bytes=ACTIVE)
            try:
                with admit_stock_market_inputs(StockMarketSpec.from_request(BacktestRequest.from_dict(request)),
                        source=source,block_sessions=2,limits=ADAPTER_LIMITS,max_market_bytes=2_000_000) as market:
                    with bind(market,request,limits=ADAPTER_LIMITS): pass
                    source.close(); handle.close()
                    with patch.object(source,"execution_scope",side_effect=AssertionError("closed original scope entered")):
                        with bind_stock_prediction_inputs(market,BacktestRequest.from_dict(request),source=source,
                                limits=ADAPTER_LIMITS,max_signal_bytes=2_000_000) as inputs:
                            self.assertEqual(inputs.statistics["source_admissions"],0)
                            self.assertEqual(inputs.statistics["capture_seconds"],0)
                    self.assertEqual(market.statistics["signal_borrows"],0)
            finally:
                if not source._closed: source.close()
                handle.close()

    def test_risk_switch_reuses_projection_but_rebinds_request_capability(self):
        import inspect
        if "market_gate" not in inspect.signature(stock_strategy_spec).parameters:
            self.skipTest("runs with the separately reviewed risk integration")
        from test_stock_market_gate import GATE,dependency,bind_risk
        from axiom_engine.core.contracts import Document
        from axiom_engine.runtime.stock_owned_inputs import _projection_key,_input_key
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plain=configured(request_for(root),stock_strategy_spec(top_k=3))
            gated=bind_risk(configured(plain,stock_strategy_spec(top_k=3,market_gate=GATE)),dependency(plain["scope"]["calendar"]))
            changed_risk=deepcopy(gated["risk_input"]);changed_risk["view"]["records"][-1]["close"]=11.0
            changed_risk["native_ref"]=Document.from_dict(changed_risk["view"]).identity
            changed=bind_risk(gated,changed_risk)
            self.assertEqual(_projection_key(plain),_projection_key(gated))
            self.assertEqual(_projection_key(gated),_projection_key(changed))
            self.assertNotEqual(_input_key(gated),_input_key(changed))
            with market_for(plain) as market,bind(market,plain) as old:
                pr=save_stock_prediction_inputs(old,root/"p",limits=LIMITS)
                with self.assertRaises(ContractError): old.inventory(gated)
                with patch("axiom_engine.runtime.stock_stream_inputs._CanonicalIndex",side_effect=AssertionError("prediction reopened")):
                    with bind(market,gated) as current:
                        self.assertIs(current._AdmittedStockInputs__projection_store,old._AdmittedStockInputs__projection_store)
                        self.assertEqual(current.statistics["source_admissions"],0)
                        with self.assertRaises(ContractError): current.inventory(changed)
                    with bind(market,changed) as current:
                        self.assertEqual(current.statistics["capture_seconds"],0)
                    with load_stock_prediction_inputs(market,pr,BacktestRequest.from_dict(gated),
                            limits=LIMITS,max_signal_bytes=2_000_000) as reopened:
                        reopened.inventory(gated)
                        with self.assertRaises(ContractError): reopened.inventory(changed)
                    invalid=deepcopy(changed_risk)
                    invalid["view"]["field_meta"]["close"]["by_key"][0]["usable_from"]=plain["scope"]["calendar"][-1]+"T21:00:00+08:00"
                    invalid["native_ref"]=Document.from_dict(invalid["view"]).identity
                    before=market.statistics["signal_borrows"]
                    with self.assertRaises(ContractError): bind(market,bind_risk(gated,invalid))
                    self.assertEqual(market.statistics["signal_borrows"],before)

    def test_fresh_process_offline_reopen_and_second_strategy_zero_original_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); request=configured(request_for(root),stock_strategy_spec(top_k=3))
            second=configured(request,stock_strategy_spec(selection="rank_buffer",top_k=3,rank_buffer={"keep_rank":10}))
            oracles=[]
            for i,p in enumerate((request,second)):
                _,view,_=execute(root/f"oracle-{i}",p); oracles.append(view.rows)
            with market_for(request) as market:
                with bind(market,request) as inputs:
                    mr=save_stock_market_inputs(market,root/"m",limits=LIMITS)
                    pr=save_stock_prediction_inputs(inputs,root/"p",limits=LIMITS)
                with patch("axiom_engine.runtime.stock_stream_inputs._CanonicalIndex",side_effect=AssertionError("rescanned")):
                    with bind(market,second) as reused:
                        self.assertEqual(reused.statistics["source_admissions"],0)
                        self.assertEqual(reused.statistics["capture_seconds"],0)
            # Keep the explicit small profile used by the unchanged result
            # loader. All Native/fold/model/prediction ancestors are unavailable.
            originals={i["artifact"]["manifest_uri"] for i in request["market_input"]["native_inputs"]}
            for frame in request["prediction_input"]["frames"]:
                originals.update(frame[n]["manifest_uri"] for n in
                    ("fold_spec_artifact","model_metadata_artifact","prediction_artifact"))
            for path in originals: Path(path).rename(Path(path+".offline"))
            (root/"handoff.json").write_text(json.dumps(dict(mr=mr,pr=pr,requests=[request,second])))
            script = '''
import json,sys
from pathlib import Path
from unittest.mock import patch
from axiom_engine.runtime import *
from test_stock_owned_inputs import run_owned
from test_stock_stream_inputs import LIMITS
root=Path(sys.argv[1]); d=json.loads((root/'handoff.json').read_text()); results=[]
with patch('axiom_engine.runtime.stock_stream_inputs._CanonicalIndex',side_effect=AssertionError('original ancestor opened')):
 with load_stock_market_inputs(d['mr'],StockMarketSpec.from_request(BacktestRequest.from_dict(d['requests'][0])),limits=LIMITS,max_market_bytes=2_000_000) as market:
  with load_stock_prediction_inputs(market,d['pr'],BacktestRequest.from_dict(d['requests'][0]),limits=LIMITS,max_signal_bytes=2_000_000) as inputs:
   _,view=run_owned(root/'fresh-0',d['requests'][0],inputs); results.append(view.rows)
  with bind_stock_prediction_inputs(market,BacktestRequest.from_dict(d['requests'][1]),source=StockInputSource(),limits=LIMITS,max_signal_bytes=2_000_000) as inputs:
   assert inputs.statistics['source_admissions']==0 and inputs.statistics['capture_seconds']==0
   _,view=run_owned(root/'fresh-1',d['requests'][1],inputs); results.append(view.rows)
(root/'fresh-results.json').write_text(json.dumps(results))
'''
            env=dict(os.environ); here=Path(__file__).resolve().parents[1]
            env["PYTHONPATH"]=str(here/"src")+os.pathsep+str(here/"tests")
            result=subprocess.run([sys.executable,"-c",script,str(root)],env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((root/"fresh-results.json").read_text()),oracles)

    def test_reopened_carried_delisted_holding_preserves_both_stops(self):
        from test_csi300 import DAYS
        from test_csi300_runtime import full_request
        from test_stock_native_lifecycle import event_rules
        from test_stock_member_lifecycle import carry,carry_frame,RETIRED
        from test_stock_equity import fixture as equity_fixture
        from test_stock_stream_inputs import source_fixture
        for equity,reason in ((False,"HELD_MISSING_STOCK_LIFECYCLE_CAPABILITY"),(True,"HELD_EQUITY_FACT_GAP")):
            with self.subTest(equity=equity),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                plan=full_request(rules=event_rules(),k=1,mutate_native=carry,mutate_frame=carry_frame).to_dict()
                if equity:
                    def missing(a):
                        a.update(security_id=RETIRED,stock_listing_date=None)
                        for alias in a["aliases"]: alias["security_id"]=RETIRED
                    request=equity_fixture(root,plan=plan,stock=None,change=missing)[0]
                else:
                    request=source_fixture(root,plan)[0]
                expected,oracle,_=execute(root/"oracle",request)
                with market_for(request) as market,bind(market,request) as inputs:
                    mr=save_stock_market_inputs(market,root/"m",limits=LIMITS)
                    pr=save_stock_prediction_inputs(inputs,root/"p",limits=LIMITS)
                actual,view=reopen(mr,pr,request,root/"reopen")
                self.assertEqual(actual["stopped"]["reason"],reason)
                self.assertEqual(actual["stopped"]["session"],DAYS[2])
                self.assertGreater(actual["final_account"]["positions"][RETIRED]["quantity"],0)
                self.assertEqual(actual["final_account"],expected["final_account"])
                self.assertEqual(view.rows,oracle.rows)

    def test_corruption_scope_semantics_and_mutation_fail_before_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); request=request_for(root)
            with market_for(request) as market,bind(market,request) as inputs:
                mr=save_stock_market_inputs(market,root/"m",limits=LIMITS)
                pr=save_stock_prediction_inputs(inputs,root/"p",limits=LIMITS)
            with patch("axiom_engine.runtime.accounting.AccountLedger",side_effect=AssertionError("ledger allocated")):
                from axiom_engine.runtime.stock_stream_inputs import _grid
                def invalid_factor(view,days,universe):
                    rows,metadata=_grid(view,days,universe)
                    if "factor" in view["field_meta"]:
                        rows[next(iter(rows))]["factor"]=-1
                    return rows,metadata
                with patch("axiom_engine.runtime.stock_stream_inputs._grid",side_effect=invalid_factor):
                    with self.assertRaises(ContractError): load_stock_market_inputs(mr,StockMarketSpec.from_request(BacktestRequest.from_dict(request)),limits=LIMITS,max_market_bytes=2_000_000)
                other=deepcopy(request); other["clock_policy"]["decision_local_time"]="08:56:00"
                with self.assertRaises(ContractError): StockMarketSpec.from_request(BacktestRequest.from_dict(other))
                with patch("axiom_engine.runtime.stock_prepared_inputs.INPUT_SEMANTICS","different"):
                    with self.assertRaises(ContractError): load_stock_market_inputs(mr,StockMarketSpec.from_request(BacktestRequest.from_dict(request)),limits=LIMITS,max_market_bytes=2_000_000)
                part=next((root/"p"/"parts").iterdir()); raw=part.read_bytes(); part.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
                with load_stock_market_inputs(mr,StockMarketSpec.from_request(BacktestRequest.from_dict(request)),limits=LIMITS,max_market_bytes=2_000_000) as market:
                    with self.assertRaises(ContractError): load_stock_prediction_inputs(market,pr,BacktestRequest.from_dict(request),limits=LIMITS,max_signal_bytes=2_000_000)
                part.write_bytes(raw)
            with load_stock_market_inputs(mr,StockMarketSpec.from_request(BacktestRequest.from_dict(request)),limits=LIMITS,max_market_bytes=2_000_000) as market:
                part=next((root/"m"/"parts").iterdir()); part.write_bytes(part.read_bytes()+b"x")
                with self.assertRaises(ContractError):
                    with market._scope(): market._market_block(0,__import__('axiom_engine.runtime.stock_stream_inputs',fromlist=['_DecodedMemory'])._DecodedMemory(500000),dict(max_read_bytes=4096,max_decoded_bytes=500000))


if __name__ == "__main__": unittest.main()
