# Best-of-8 probe incomplete-result amendment

The preregistered `move_rel` best-of-8 probe has 86 infrastructure-valid cells
out of 96 intended cells.  The initial seed-503 attempt published valid results
for canonical task indices 0 and 1, then failed while fetching the screenshot
after an action on canonical index 2.  A continuation restricted to canonical
indices 2 through 11 failed at the same screenshot fetch before publishing any
result.

The frozen OSWorld screenshot client already made three attempts of the same
live HTTP GET (10-second timeout and 5-second interval).  The two jobs ended
with the same HTTP 500 / disconnected / reset sequence and the same sealed
`failure.json`.  The screenshot loss occurred after the action was executed.
No live VM memory or task-state checkpoint and no persisted model/action trace
exists at that boundary; teardown terminated the VM.  Therefore another job
could only reset the VM and repeat model inference and actions, not retry the
same observation request against the exact state.  No further GPU retry is
authorized.

The scientific output is consequently an incomplete aggregate.  The
preregistered yield gate remains formally unresolved.  Deterministic bounds
and a clearly labelled post-hoc Bayesian sensitivity diagnostic may be
reported, but neither substitutes for the gate.
