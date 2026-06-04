
The scenarios for which world state patcher fails should be tracked, and dependent scenarios to that should be skipped
- world_state_patcher.py  lines 150 151

- orchestrator.py / _should_continue_after_patcher is a dead branch and continues advance all times


- orchestrator.py   run()  silently returns empty dict if reporter never fires

- orchestrator.py   _should_continue_after_generate  cheks status == "failed" but nothing is setting that

- orchestrator.py    run()  the _adapter is never closed on error paths


- orchestrator.py   initialize()  can partially succeed and leave state inconsistent. catch specific exceptions from each step and wrap them in a structured error with clear message


- orchestrator.py   the progress events lose information when node_output has no status. define a fixed progress event schema and always emit that shape.

