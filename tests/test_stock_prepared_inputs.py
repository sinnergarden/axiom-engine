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
