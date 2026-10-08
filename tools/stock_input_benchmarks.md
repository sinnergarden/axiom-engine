# Stock input benchmark and test notes

These notes explain the software probes, recorded measurements and their limits.
The public API and input-reuse contract are maintained in
[Axiom design 04](https://github.com/sinnergarden/axiom-docs/blob/af702babcfbead8cf0850aaf00738067b497fd35/docs/design/04_axiom_trade.md#66-已保存股票输入的一次准入与顺序账户复用).
Implementation increments were integrated through [PR21](https://github.com/sinnergarden/axiom-engine/pull/21)
and [PR22](https://github.com/sinnergarden/axiom-engine/pull/22).
Original inputs, account metrics, local locations and detailed receipts remain private.

## Synthetic probes

Use CPython 3.12 with its standard `_json` backend and permitted read-only RSS
monitoring. Both probes create temporary synthetic inputs; they do not call
Data, suppliers, feature construction, fitting or prediction.

```sh
PYTHONPATH=src:tests python3 -B tools/bench_stock_owned_inputs.py --coverage-rows 10000
PYTHONPATH=src:tests python3 -B tools/bench_stock_cjson.py --coverage-rows 20000
```

The owned probe compares cached/uncached admission, then five independent TopK
accounts against the original source oracle. The CJSON probe uses mixed repeated
strings/keys and unique ordinal/hash/float/UTF-8 values. It checks complete saved
headers and ledger rows against stream, and distinct Top3/Top5 accounts over the
same Signal. A synthetic PASS does not establish real input speed or memory bounds.

## Recorded evidence and source binding

| Frozen source | Scope | Recorded result |
|---|---|---|
| `26c49df` | Synthetic repeated coverage, 2,147,694 bytes | Cache off/on cold 0.849/0.540s; decode/canonical calls 105,725/291; five owned accounts 0.146s; saved outputs exact |
| `b5dc2f4` | Existing bounded real input, 594,205,431 bytes | Owned reuse exact; cold audit 472.639197s, complete cold 476.202292s; scalar-cache acceleration not accepted |
| `327cbd4` | Mixed synthetic coverage, 4,065,531 bytes | Stream/CJSON complete cold 1.805626/1.541866s; nine native helpers; complete saved outputs exact |
| `2842ca9` | Same original bounded real input, one audit/capture and two accounts | Complete cold 43.221188s; nine native CJSON, 17 expected nonnative stream; both saved business oracles exact |

The `b5dc2f4` scalar trial recorded a 5.12% hit rate and 45,648,342 evictions.
Its audit was slower than the earlier `21cc1d5` 349.007s observation. The later
`da10733` therefore keeps scalar cache disabled by default while retaining opt-in.
Synthetic cache gains do not justify changing this default or repeated real tuning.

The 59-test synthetic suite and mixed cold measurement belong to `327cbd4`.
Six targeted checks on `2842ca9` passed in 11.372s after the two review fixes;
the earlier suite was not rerun or rebound. The final real evidence binds to
`2842ca992bb7c557945866ef3a48be80b21d7a75` and implementation
`sha256:bd17832943f391e14dd7b3ae78cb5c314e05698fc0f369d8368cbcfc0ed66978`.
The integrated `182be5c` tree equals that frozen source tree; integration did not
rerun a benchmark or change its binding. This notes-only change does not alter it.

The final real cold audit was 39.658387s, capture 3.544879s and controller window
45.407932s. An initial harness stop cost 0.991207s and produced no complete account
acceptance; cumulative actual window cost is 46.399139s. The final admission used
the original input and saved business oracles, without rerunning a stream baseline.
The old 476.202292s versus new 43.221188s observations give a ratio of about 11.02;
source/cache version, machine state and OS file-cache conditions were not controlled,
so this is not a same-head A/B result. "Cold" means a fresh Source admission and
owned capture, including helper launch; it does not claim flushed OS caches.

## Timing and resource accounting

Do not add overlapping counters. Native `scan_seconds` includes helper launch/exit
and parent indexing; `cjson_helper_seconds` includes its internal read/hash, UTF-8
parse, full validation, canonical comparison and private spool work. The comparison
counter includes C encoding, UTF-8 conversion and the extra original-byte read;
the current counters do not separately measure every one of those operations.
Source selected-row reads/decode and owned capture are reported separately from
subsequent account consumption. Original-source opens after successful capture were
zero; audit/capture each occurred once and ten owned blocks served both accounts.

| Native file kind | Actual backend | Scan seconds | Helper seconds |
|---|---|---:|---:|
| states | CJSON | 0.866264 | 0.834612 |
| market | CJSON | 2.337932 | 2.303671 |
| limits | CJSON | 5.127539 | 5.095928 |
| factor | CJSON | 0.631599 | 0.602397 |
| actions-ex_date | CJSON | 7.423011 | 7.395639 |
| actions-record_date | CJSON | 7.569583 | 7.541549 |
| membership | CJSON | 0.764689 | 0.735081 |
| warmup-market | CJSON | 0.276322 | 0.247982 |
| warmup-factor | CJSON | 0.165258 | 0.139195 |

All nine native complete events were CJSON; no native preflight stream choice
occurred. Profile and four folds' fold/manifest/model/predictions files comprise
the 17 expected nonnative stream files. Raw scan/hash covered 594,205,431 bytes;
canonical comparison reread 580,218,841 native bytes. Helper time totaled 24.896053s:
UTF-8/parse 2.246812s, whole validation 14.515282s, comparison 2.157597s and spool
3.598419s. These measured subphases do not replace complete cold admission time.

The real plan explicitly used 256MiB/file, 8GiB/process tree, 128MiB/private spool,
128MiB/owned, 64MiB/new named results and 120s/helper caps. The sampled controller
tree peak was 1,799,520,256 bytes. Combining native helper peak evidence closes
sampling gaps and gives an observed peak of 2,016,051,200 bytes (1.87759GiB).
Spool used 46,206,192 bytes and owned used 15,102,120 bytes. Every declared limit
passed, handles/helpers closed, and original file/stat/hash/source guards passed.
These are observations for one bounded input, not hard per-allocation guarantees.

The planning estimate for file size S is `34S + 64MiB + max(512MiB, parentRSS) +
256MiB`: graph/pairs/key storage estimates 24S and encoder growth 10S. The largest
195.95MiB file planned about 7.32GiB. The graph multiplier is not a universal bound;
input text/raw and canonical bytes are released/stream-compared rather than kept
as additional full copies. Budget or monitoring ineligibility selects stream only
before helper launch. Started failure discards uncommitted views without retry.

## Regression coverage and monitor lessons

The focused `2842ca9` checks cover 17 metadata shapes and complete grid rejection,
growth/equal-size/inode races, the file cap before payload read, complete saved
Top3/Top5 equality, started MemoryError/RSS/timeout/spool cleanup and source freeze.
See `tests/test_stock_cjson.py` and `tests/test_implementation_ref.py`; broader
counterexamples include duplicate/unsorted keys, noncanonical numbers/whitespace/
Unicode, NaN/Infinity/overflow, reserved Unknown and nesting depth over 128.

Metadata object/list child selection retains original scanner wildcard semantics;
malformed nonempty lists must reach the existing grid gate rather than disappear.
Preflight/index/helper use the same dev/inode/size/mtime/ctime snapshot. Both path
and fd cap checks precede the admitted-size bulk read, and EOF/stat are checked
before hash/parse. Standard `json.load` still reads the whole document; ordinary
`iterencode` follows the Python encoder. Neither is evidence of C streaming.

The private real harness originally misclassified transient macOS `(ps)` by name.
The correction removes runtime name/argv allowlists: sample the own process tree
plus declared helper PID/independent-PGID/birth identities. Recheck those identities
before TERM/KILL, retain declared helpers after worker reparenting, tolerate short
exit between snapshot and lookup, defer incomplete declaration tails, and avoid
signaling a reused PID with a different birth. Five small control checks passed
before the parent-authorized continuation; they launch gate-blocked helpers without
real input payloads. Detailed process receipts and control fixtures remain private.

Real acceptance compares all business headers and eight result row groups with
zero numeric tolerance. Only declared source/output provenance and run-prefix IDs
are normalized. Complete finite/Unknown/depth/canonical-byte and original business
gates remain in use; no Data/Research/Feature/fit/predict/supplier calls occurred.
No private account metrics or original locations are reproduced here.
