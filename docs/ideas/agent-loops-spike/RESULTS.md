# Phase 0 results (gemma4:26b via Ollama, think=False, judge json_format=True)
Run 1 (default judge) was invalid: ~60% of judge replies unparsable (truncated JSON / leaked reasoning). Kept as results_v1_think_judge.json.
Run 2 (results.json): judge stats ok=99 unparsable=3 failed_after_retry=0.

| # | Assumption | Rule | Result | Verdict |
|---|---|---|---|---|
| A | Judge flip rate | <= 10% | 0% (coding, writing, research; 5 reruns each) | PASS |
| B | Fresh vs shared | fresh <= shared + 1 iter | coding 2.0 vs 2.0; writing 4.0 vs 2.0; combined 3.0 vs 2.0 (+1.0) | BORDERLINE (writing fails) |
| C | Batch accuracy | >= 90% at 10 reqs | coding 90%, writing 80% (avg 85%); batch-of-3 avg 85% | FAIL (groups of 3 not better) |
| D | Command catches what judge misses | >= 1 catch | judge alone accepted 0/15 seeded bugs; pytest caught 3/3; LLM wrongly passed 54-word 'needs 70-110' 1/5 | NOT MET (weak evidence) |
| E | Stall N | smallest N with <= 1 false stall | N=2:1, N=3:1, N=4:0 over 12 runs -> keep 3 | PASS (default 3) |
| F | Tool chaining | one execute_task writes 2 files | both files written in 25s | PASS |

Extra findings: judge must use JSON mode + no thinking + one retry; judge-error iterations must not count toward stall; countable constraints need code checks; fresh-context prompt must include the current artifact; shared runs were deterministic (n=3 ~ 1 sample); spike never exceeded 6 iterations, so long-loop context bloat is untested.
